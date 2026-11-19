"""calculate(): arithmetic and unit conversion, so the model does not do sums
in its head (T1, the first family tool).

The expression is parsed with the ast module and only numbers, + - * / // %
**, unary signs, a few math functions and pi and e are evaluated: no eval, no
names, no attributes. Before parsing, a few spoken forms are rewritten:
"17% of 2400", "17%" (a fraction), "x" and the times sign, the division
sign, "^", "plus", "minus", "times", "multiplied by", "divided by", and
thousands commas ("2,400"; so a list like max(100,200) needs a space after the
comma). Powers and results past about 1e300 are refused.

A conversion is "<amount> <unit> to|in|into|as <unit>"; the amount may be an
expression ("1/2 gallon to liters"). Units come from a fixed table with exact
definitions: metric, US customary (cup, fluid ounce, pint, quart and gallon
are US measures), temperature (C, F, K), time and speed. Every unit is linear
in its dimension's base unit (base = value * factor + offset), which covers
the temperature scales too.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

from obsilord.tools.registry import ToolError, ToolPolicy, tool

MAX_CHARS = 200
MAX_MAGNITUDE = 1e300
MSG_BAD_EXPRESSION = "I can only work out arithmetic and unit conversions"
MSG_TOO_BIG = "The number is too big"
MSG_TOO_LONG = "The expression is too long"
MSG_ZERO = "Cannot divide by zero"


@dataclass(frozen=True)
class Unit:
    symbol: str
    dimension: str
    factor: float
    offset: float = 0.0


def _units() -> dict[str, Unit]:
    table: list[tuple[Unit, tuple[str, ...]]] = [
        (Unit("mm", "length", 0.001), ("millimeter", "millimeters", "millimetre", "millimetres")),
        (Unit("cm", "length", 0.01), ("centimeter", "centimeters", "centimetre", "centimetres")),
        (Unit("m", "length", 1.0), ("meter", "meters", "metre", "metres")),
        (Unit("km", "length", 1000.0), ("kilometer", "kilometers", "kilometre", "kilometres")),
        (Unit("in", "length", 0.0254), ("inch", "inches")),
        (Unit("ft", "length", 0.3048), ("foot", "feet")),
        (Unit("yd", "length", 0.9144), ("yard", "yards")),
        (Unit("mi", "length", 1609.344), ("mile", "miles")),
        (Unit("mg", "mass", 0.001), ("milligram", "milligrams")),
        (Unit("g", "mass", 1.0), ("gram", "grams")),
        (Unit("kg", "mass", 1000.0), ("kilo", "kilos", "kilogram", "kilograms")),
        (Unit("oz", "mass", 28.349523125), ("ounce", "ounces")),
        (Unit("lb", "mass", 453.59237), ("lbs", "pound", "pounds")),
        (Unit("ml", "volume", 1.0), ("milliliter", "milliliters", "millilitre", "millilitres")),
        (Unit("l", "volume", 1000.0), ("liter", "liters", "litre", "litres")),
        (Unit("tsp", "volume", 4.92892159375), ("teaspoon", "teaspoons")),
        (Unit("tbsp", "volume", 14.78676478125), ("tablespoon", "tablespoons")),
        (Unit("fl oz", "volume", 29.5735295625), ("fluid ounce", "fluid ounces")),
        (Unit("cup", "volume", 236.5882365), ("cups",)),
        (Unit("pint", "volume", 473.176473), ("pints", "pt")),
        (Unit("quart", "volume", 946.352946), ("quarts", "qt")),
        (Unit("gal", "volume", 3785.411784), ("gallon", "gallons")),
        (Unit("°C", "temperature", 1.0, 273.15), ("c", "celsius", "degrees celsius", "degrees c")),
        (
            Unit("°F", "temperature", 5 / 9, 273.15 - 32 * 5 / 9),
            ("f", "fahrenheit", "degrees fahrenheit", "degrees f"),
        ),
        (Unit("K", "temperature", 1.0), ("k", "kelvin")),
        (Unit("s", "time", 1.0), ("sec", "secs", "second", "seconds")),
        (Unit("min", "time", 60.0), ("mins", "minute", "minutes")),
        (Unit("h", "time", 3600.0), ("hr", "hrs", "hour", "hours")),
        (Unit("day", "time", 86400.0), ("days",)),
        (Unit("week", "time", 604800.0), ("weeks",)),
        (Unit("m/s", "speed", 1.0), ("meters per second", "metres per second")),
        (Unit("km/h", "speed", 1000 / 3600), ("kph", "kmh", "kilometers per hour", "kilometres per hour")),
        (Unit("mph", "speed", 0.44704), ("miles per hour",)),
        (Unit("knot", "speed", 1852 / 3600), ("knots", "kn")),
    ]
    out: dict[str, Unit] = {}
    for unit, aliases in table:
        for key in (unit.symbol.lower(), unit.symbol.lower().replace("°", ""), *aliases):
            out[key] = unit
    return out


UNITS = _units()

_BINARY: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _rewrite(text: str) -> str:
    """Spoken and typed forms as Python arithmetic."""
    s = " ".join(text.split())
    s = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", lambda m: m.group(0).replace(",", ""), s)
    s = s.replace("×", "*").replace("÷", "/").replace("^", "**")
    s = re.sub(r"(?<=[\d)\s])x(?=[\s\d(])", "*", s)
    s = re.sub(r"\bmultiplied by\b|\btimes\b", "*", s, flags=re.I)
    s = re.sub(r"\bdivided by\b", "/", s, flags=re.I)
    s = re.sub(r"\bplus\b", "+", s, flags=re.I)
    s = re.sub(r"\bminus\b", "-", s, flags=re.I)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\b", r"(\1/100)*", s, flags=re.I)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\d(.])", r"(\1/100)", s)
    return s


def _checked(value: Any) -> Any:
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        raise ValueError(MSG_TOO_BIG)
    if abs(value) > MAX_MAGNITUDE:
        raise ValueError(MSG_TOO_BIG)
    return value


def _power(base: Any, exp: Any) -> Any:
    if base != 0 and abs(base) != 1:
        if abs(exp) * math.log10(abs(base)) > math.log10(MAX_MAGNITUDE):
            raise ValueError(MSG_TOO_BIG)
    try:
        value = base**exp
    except ZeroDivisionError:
        raise ValueError(MSG_ZERO) from None
    except OverflowError:
        # A base below 1 with a large negative exponent.
        raise ValueError(MSG_TOO_BIG) from None
    if isinstance(value, complex):
        # A negative base with a fractional exponent has no real result.
        raise ValueError(MSG_BAD_EXPRESSION)
    return value


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return _checked(node.value)
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval(node.operand)
        return -value if isinstance(node.op, ast.USub) else +value
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            return _checked(_power(left, right))
        op = _BINARY.get(type(node.op))
        if op is None:
            raise ValueError(MSG_BAD_EXPRESSION)
        try:
            return _checked(op(left, right))
        except ZeroDivisionError:
            raise ValueError(MSG_ZERO) from None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
        and node.args
    ):
        name = node.func.id
        args = [_eval(a) for a in node.args]
        if name == "round" and len(args) == 2:
            args[1] = int(args[1])
        try:
            return _checked(_FUNCTIONS[name](*args))
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f"{name} of that is not defined") from None
    raise ValueError(MSG_BAD_EXPRESSION)


def evaluate(expression: str) -> float | int:
    """The value of an arithmetic expression. Raises ValueError with a
    readable message for anything else."""
    if len(expression) > MAX_CHARS:
        raise ValueError(MSG_TOO_LONG)
    text = _rewrite(expression)
    if not text:
        raise ValueError(MSG_BAD_EXPRESSION)
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError):
        raise ValueError(MSG_BAD_EXPRESSION) from None
    return _eval(tree)


def format_number(value: float | int) -> str:
    """value without float noise: whole numbers as integers, others with at
    least six significant digits (more for large numbers)."""
    if isinstance(value, int):
        return str(value) if abs(value) < 10**15 else f"{value:.6g}"
    if abs(value) >= 1e15:
        return f"{value:.6g}"
    if value.is_integer():
        return str(int(value))
    digits = len(str(int(abs(value)))) if abs(value) >= 1 else 0
    return f"{value:.{min(15, max(6, digits + 3))}g}"


_CONVERSION = re.compile(
    r"^\s*(?P<amount>.*?[\d)])\s*(?P<src>[a-z°][a-z°/ .]*?)\s+(?:to|in|into|as)\s+"
    r"(?P<dst>[a-z°][a-z°/ .]*?)\s*\.?\s*$",
    re.I,
)


def _unit(name: str) -> Unit | None:
    key = " ".join(name.lower().replace("degrees", "degrees ").split()).rstrip(".")
    return UNITS.get(key)


def convert(text: str) -> str | None:
    """"5 km = 3.10686 mi" for a conversion, None when text is not one this
    table knows. Raises ValueError for units of different dimensions or a bad
    amount."""
    if len(text) > MAX_CHARS:
        raise ValueError(MSG_TOO_LONG)
    match = _CONVERSION.match(text)
    if match is None:
        return None
    src, dst = _unit(match["src"]), _unit(match["dst"])
    if src is None or dst is None:
        return None
    if src.dimension != dst.dimension:
        raise ValueError(f"{src.symbol} and {dst.symbol} measure different things")
    amount_text = " ".join(match["amount"].split())
    amount = evaluate(amount_text)
    value = _checked(((amount * src.factor + src.offset) - dst.offset) / dst.factor)
    return f"{amount_text} {src.symbol} = {format_number(value)} {dst.symbol}"


@tool(ToolPolicy(per_turn=3, timeout_s=2.0))
async def calculate(expression: str, action: Literal["math", "convert"] = "math") -> str:
    """Work out arithmetic or convert units exactly, instead of doing it in your
    head. Use it for any sum, percentage or unit conversion with numbers, for
    example '17% of 2400', '1234 * 5678' or '3 oz to g'. Not for counting,
    dates or questions without numbers to work with.

    Args:
        expression: The arithmetic, for example '2400 * 0.17' or '17% of 2400',
            or a conversion as '<amount> <unit> to <unit>', for example
            '350 F to C' or '2 cups to ml'.
        action: math for arithmetic, convert for a unit conversion.
    """
    try:
        converted = convert(expression)
        if converted is not None:
            return converted
        if action == "convert":
            raise ToolError(f"Unknown units in '{' '.join(expression.split())[:80]}'.")
        return f"{' '.join(expression.split())} = {format_number(evaluate(expression))}"
    except ValueError as exc:
        raise ToolError(f"{exc}.") from None
