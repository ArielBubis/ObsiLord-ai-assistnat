"""Google Cloud Vision web detection implementation of ImageSearchClient.

This is the only module that talks to the Vision API. Looked up 2026-11-14
(docs.cloud.google.com/vision/docs/detecting-web):
- POST https://vision.googleapis.com/v1/images:annotate with
  {"requests": [{"image": {"content": BASE64}, "features": [{"type":
  "WEB_DETECTION", "maxResults": N}]}]}.
- The answer is responses[0].webDetection with bestGuessLabels[].label,
  webEntities[].description and .score, pagesWithMatchingImages[].pageTitle
  (which may hold <b> markup), and image URL lists that are not used here.
  A failed request has responses[0].error.message or a top-level error.
- Pricing page: the first 1,000 units a month are free, then $3.50 per 1,000.
  One request with one feature is one unit.
- The key goes in the X-Goog-Api-Key header, so it is never part of a URL.
  API key auth for this endpoint is unverified until the live check (the docs
  page only shows OAuth); see planning/PLAN.md.

The upload is a blocking stdlib HTTP call run in a worker thread. Nothing
about the image or the results is logged.
"""

from __future__ import annotations

import asyncio
import base64
import http.client
import json
import urllib.error
import urllib.request
from typing import Any

from obsilord.imagesearch.base import (
    ImageMatches,
    ImageSearchError,
    clean_field,
    unique,
)

ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
PROVIDER = "google_vision"
MAX_ERROR_CHARS = 200


def build_request(jpeg: bytes, max_results: int) -> dict[str, Any]:
    return {
        "requests": [
            {
                "image": {"content": base64.b64encode(jpeg).decode("ascii")},
                "features": [{"type": "WEB_DETECTION", "maxResults": max_results}],
            }
        ]
    }


def _error_message(data: Any) -> str | None:
    """Google's error text from a response body, if it has one."""
    if not isinstance(data, dict):
        return None
    error = data.get("error")
    if not isinstance(error, dict):
        responses = data.get("responses")
        first = responses[0] if isinstance(responses, list) and responses else None
        error = first.get("error") if isinstance(first, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return " ".join(error["message"].split())[:MAX_ERROR_CHARS]
    return None


def parse_response(data: Any, max_results: int) -> ImageMatches:
    """The matches from an images:annotate answer. Raises ImageSearchError on
    an error answer or an unexpected shape. No web detection at all is an
    empty result, not an error."""
    message = _error_message(data)
    if message is not None:
        raise ImageSearchError(f"Vision API error: {message}")
    try:
        first = data["responses"][0]
    except (KeyError, IndexError, TypeError):
        raise ImageSearchError("unexpected response from the Vision API") from None
    if not isinstance(first, dict):
        raise ImageSearchError("unexpected response from the Vision API")
    web = first.get("webDetection")
    if not isinstance(web, dict):
        return ImageMatches()

    def items(key: str) -> list[dict[str, Any]]:
        value = web.get(key)
        return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []

    entities = sorted(
        items("webEntities"),
        key=lambda e: e.get("score") if isinstance(e.get("score"), (int, float)) else 0.0,
        reverse=True,
    )
    return ImageMatches(
        best_guesses=unique(
            [clean_field(g.get("label")) for g in items("bestGuessLabels")], max_results
        ),
        entities=unique([clean_field(e.get("description")) for e in entities], max_results),
        page_titles=unique(
            [clean_field(p.get("pageTitle")) for p in items("pagesWithMatchingImages")],
            max_results,
        ),
    )


class GoogleVisionSearch:
    provider = PROVIDER

    def __init__(
        self,
        api_key: str,
        max_results: int,
        timeout_s: float,
        opener: urllib.request.OpenerDirector | None = None,
    ) -> None:
        self._api_key = api_key
        self._max_results = max_results
        self._timeout_s = timeout_s
        self._opener = opener if opener is not None else urllib.request.build_opener()

    def _post(self, jpeg: bytes) -> Any:
        body = json.dumps(build_request(jpeg, self._max_results)).encode("utf-8")
        request = urllib.request.Request(
            ENDPOINT,
            data=body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "X-Goog-Api-Key": self._api_key,
            },
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self._timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            try:
                message = _error_message(json.loads(exc.read()))
            except (ValueError, OSError):
                message = None
            detail = f": {message}" if message else ""
            raise ImageSearchError(f"Vision API HTTP {exc.code}{detail}") from None

    async def search(self, jpeg: bytes) -> ImageMatches:
        try:
            data = await asyncio.to_thread(self._post, jpeg)
        except ImageSearchError:
            raise
        except (OSError, ValueError, http.client.HTTPException) as exc:  # URLError, timeout, bad JSON
            raise ImageSearchError(f"Vision API request failed: {type(exc).__name__}") from None
        return parse_response(data, self._max_results)
