"""calculate(): the safe arithmetic evaluator and the unit table."""

from __future__ import annotations

import asyncio

import pytest

from obsilord.tools.calculate import (
    MSG_BAD_EXPRESSION,
    MSG_TOO_BIG,
    MSG_TOO_LONG,
    UNITS,
    calculate,
    convert,
    evaluate,
    format_number,
)


@pytest.mark.parametrize(
    ("expr", "value"),
    [
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("2^10", 1024),
        ("2 ** 3 ** 2", 512),
        ("-3 + +5", 2),
        ("7 / 2", 3.5),
        ("7 // 2", 3),
        ("7 % 3", 1),
        ("17% of 2400", 408),
        ("2400 * 17%", 408),
        ("50%", 0.5),
        ("2,400 * 0.5", 1200),
        ("1,234,567 + 1", 1234568),
        ("3 x 4", 12),
        ("3 × 4 ÷ 2", 6),
        ("10 plus 5 minus 3", 12),
        ("6 times 7", 42),
        ("84 divided by 2", 42),
        ("sqrt(16) + abs(-2)", 6),
        ("round(3.14159, 2)", 3.14),
        ("max(1, 5, 3) - min(4, 2)", 3),
        ("floor(2.7) + ceil(2.1)", 5),
        ("pi", 3.141592653589793),
        ("1e3 + 1", 1001),
    ],
)
def test_evaluate(expr: str, value: float) -> None:
    assert evaluate(expr) == pytest.approx(value)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",
        "().__class__",
        "open('x')",
        "a + 1",
        "lambda: 1",
        "[1, 2]",
        "'text'",
        "1 if 1 else 2",
        "2 < 3",
        "",
        "sqrt(x=4)",
    ],
)
def test_evaluate_refuses_anything_but_arithmetic(expr: str) -> None:
    with pytest.raises(ValueError, match=MSG_BAD_EXPRESSION):
        evaluate(expr)


def test_evaluate_limits() -> None:
    with pytest.raises(ValueError, match=MSG_TOO_BIG):
        evaluate("9 ** 9 ** 9")
    with pytest.raises(ValueError, match=MSG_TOO_BIG):
        evaluate("10 ** 400")
    with pytest.raises(ValueError, match=MSG_TOO_BIG):
        evaluate("0.1 ** -400")
    with pytest.raises(ValueError, match=MSG_TOO_BIG):
        evaluate("0.5 ** -2000")
    # A negative base with a fractional exponent has no real result.
    with pytest.raises(ValueError, match=MSG_BAD_EXPRESSION):
        evaluate("(-8) ** 0.5")
    with pytest.raises(ValueError, match=MSG_BAD_EXPRESSION):
        evaluate("(-8) ** (1 / 3)")
    with pytest.raises(ValueError, match=MSG_TOO_LONG):
        evaluate("1 + " * 100 + "1")
    with pytest.raises(ValueError, match="divide by zero"):
        evaluate("1 / 0")
    with pytest.raises(ValueError, match="divide by zero"):
        evaluate("5 % 0")
    with pytest.raises(ValueError, match="sqrt"):
        evaluate("sqrt(-1)")


@pytest.mark.parametrize(
    ("text", "out"),
    [
        ("5 km to miles", "5 km = 3.10686 mi"),
        ("3 ounces in grams", "3 oz = 85.0486 g"),
        ("350 F to C", "350 °F = 176.667 °C"),
        ("100 celsius in fahrenheit", "100 °C = 212 °F"),
        ("0 K to C", "0 K = -273.15 °C"),
        ("2 cups to ml", "2 cup = 473.176 ml"),
        ("1 tbsp to tsp", "1 tbsp = 3 tsp"),
        ("1.5 kg to pounds", "1.5 kg = 3.30693 lb"),
        ("90 minutes to hours", "90 min = 1.5 h"),
        ("100 km/h to mph", "100 km/h = 62.1371 mph"),
        ("6 feet in cm", "6 ft = 182.88 cm"),
        ("1/2 gallon to liters", "1/2 gal = 1.89271 l"),
        ("2 fl oz to ml", "2 fl oz = 59.1471 ml"),
    ],
)
def test_convert(text: str, out: str) -> None:
    assert convert(text) == out


def test_convert_refuses_unknown_and_mismatched_units() -> None:
    assert convert("5 parsecs to km") is None  # not a conversion this table knows
    with pytest.raises(ValueError, match="kg and km"):
        convert("5 kg to km")


def test_every_unit_has_a_dimension_and_a_factor() -> None:
    for unit in UNITS.values():
        assert unit.dimension in {"length", "mass", "volume", "temperature", "time", "speed"}
        assert unit.factor > 0


def test_format_number() -> None:
    assert format_number(408.0) == "408"
    assert format_number(1 / 3) == "0.333333"
    assert format_number(1234567.891) == "1234567.891"
    assert format_number(1e20) == "1e+20"


def test_tool_answers() -> None:
    assert asyncio.run(calculate.fn("17% of 2400")) == "17% of 2400 = 408"
    assert asyncio.run(calculate.fn("5 km to miles", "convert")) == "5 km = 3.10686 mi"
    # A conversion is found even when the model said math.
    assert asyncio.run(calculate.fn("5 km to miles", "math")) == "5 km = 3.10686 mi"


def test_tool_errors_are_readable() -> None:
    from obsilord.tools.registry import ToolError

    with pytest.raises(ToolError, match=MSG_BAD_EXPRESSION):
        asyncio.run(calculate.fn("open the door"))
    with pytest.raises(ToolError, match="Unknown units"):
        asyncio.run(calculate.fn("5 parsecs to km", "convert"))


def test_declaration() -> None:
    spec = calculate.spec
    assert spec.name == "calculate"
    assert spec.parameters["required"] == ["expression"]
    assert spec.parameters["properties"]["action"]["enum"] == ["math", "convert"]
    assert calculate.policy.private_data is False and calculate.policy.per_turn == 3
