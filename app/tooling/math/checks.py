"""Symbolic verification utilities used by the math service."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence

import sympy  # type: ignore[import-untyped]
from sympy import Matrix
from sympy.parsing.sympy_parser import parse_expr  # type: ignore[import-untyped]

from app.tooling.math.models import (
    DerivativeCheckRequest,
    DerivativeCheckResponse,
    EqualityCheckRequest,
    EqualityCheckResponse,
)

_NUMERIC_TOLERANCE = 1e-6
_RANDOM_RANGE = (-3.0, 3.0)


def _build_symbol_map(symbol_names: Sequence[str]) -> dict[str, sympy.Symbol]:
    """Return a deterministic mapping of symbol names to SymPy symbols."""

    if not symbol_names:
        return {}
    return {name: sympy.symbols(name, real=True) for name in symbol_names}


def _parse_expression(expression: str, symbol_map: dict[str, sympy.Symbol]) -> sympy.Expr:
    """Parse an expression string into a SymPy expression with restricted locals."""

    try:
        return parse_expr(expression, local_dict=symbol_map, evaluate=True)
    except (sympy.SympifyError, SyntaxError) as exc:  # pragma: no cover - defensive
        msg = f"Unable to parse expression '{expression}': {exc}."
        raise ValueError(msg) from exc


def _format_matrix(matrix: Matrix) -> list[list[str]]:
    """Convert a SymPy matrix into a JSON-friendly nested string list."""

    return [[sympy.simplify(item).__str__() for item in row] for row in matrix.tolist()]


def _random_assignment(symbols: Iterable[sympy.Symbol]) -> dict[sympy.Symbol, float]:
    """Generate a random assignment for each symbol within a bounded range."""

    return {symbol: random.uniform(*_RANDOM_RANGE) for symbol in symbols}


def _matrix_abs_max(matrix: Matrix, subs: dict[sympy.Symbol, float]) -> float:
    """Return the maximum absolute value of a matrix after substitution."""

    evaluated = matrix.subs(subs)
    flat_values = []
    for item in evaluated:
        # SymPy may return complex results when expressions are undefined; take magnitude.
        value = complex(item.evalf())
        flat_values.append(abs(value))
    return max(flat_values) if flat_values else 0.0


def _is_zero_matrix(matrix: Matrix) -> bool:
    """Return True when every entry in the matrix simplifies to zero."""

    return all(sympy.simplify(item) == 0 for item in matrix)


def check_symbolic_equality(request: EqualityCheckRequest) -> EqualityCheckResponse:
    """Determine whether two expressions are mathematically equivalent."""

    symbol_map = _build_symbol_map(request.symbols)
    canonical_expr = _parse_expression(request.canonical, symbol_map)
    candidate_expr = _parse_expression(request.candidate, symbol_map)

    combined_symbols = canonical_expr.free_symbols | candidate_expr.free_symbols
    if symbol_map:
        combined_symbols |= set(symbol_map.values())

    difference = sympy.simplify(canonical_expr - candidate_expr)
    if difference == sympy.Integer(0):
        detail = "Expressions are identical symbolically."
        return EqualityCheckResponse(equivalent=True, detail=detail)

    for _ in range(request.random_trials):
        subs = _random_assignment(combined_symbols)
        magnitude = abs(complex(difference.evalf(subs=subs)))
        if math.isnan(magnitude) or magnitude > _NUMERIC_TOLERANCE:
            detail = "Expressions differ either symbolically or under numeric evaluation."
            return EqualityCheckResponse(equivalent=False, detail=detail)

    detail = (
        "Expressions differ symbolically but matched across random numeric trials; treatment as "
        "equivalent requires contextual judgement."
    )
    return EqualityCheckResponse(equivalent=True, detail=detail)


def check_jacobian(request: DerivativeCheckRequest) -> DerivativeCheckResponse:
    """Verify that a supplied Jacobian matches the true Jacobian of a function."""

    symbol_map = _build_symbol_map(request.variables)
    symbols = [symbol_map.get(name, sympy.symbols(name, real=True)) for name in request.variables]

    function_components = [
        _parse_expression(component, symbol_map) for component in request.function
    ]
    candidate_rows = [
        [_parse_expression(entry, symbol_map) for entry in row] for row in request.candidate
    ]
    candidate_matrix = Matrix(candidate_rows)

    jacobian = Matrix(function_components).jacobian(symbols)
    difference = sympy.simplify(jacobian - candidate_matrix)

    if _is_zero_matrix(difference):
        detail = "Jacobian matches symbolically."
        evaluated = [[0.0 for _ in request.variables] for _ in request.function]
        return DerivativeCheckResponse(
            equivalent=True,
            detail=detail,
            symbolic_difference=_format_matrix(difference),
            evaluated_difference=evaluated,
        )

    evaluation_point = request.evaluation_point
    if evaluation_point:
        subs = {
            symbol_map.get(name, sympy.symbols(name, real=True)): value
            for name, value in evaluation_point.items()
        }
        abs_max = _matrix_abs_max(difference, subs)
        if abs_max > _NUMERIC_TOLERANCE:
            detail = (
                "Jacobian mismatch detected based on symbolic simplification and evaluation at the "
                "provided point."
            )
            evaluated = difference.subs(subs)
            numeric_diff = [[float(sympy.N(item)) for item in row] for row in evaluated.tolist()]
            return DerivativeCheckResponse(
                equivalent=False,
                detail=detail,
                symbolic_difference=_format_matrix(difference),
                evaluated_difference=numeric_diff,
            )

    for _ in range(request.random_trials):
        subs = _random_assignment(symbols)
        abs_max = _matrix_abs_max(difference, subs)
        if abs_max > _NUMERIC_TOLERANCE:
            detail = "Jacobian mismatch detected during random numeric evaluation."
            evaluated = difference.subs(subs)
            numeric_diff = [[float(sympy.N(item)) for item in row] for row in evaluated.tolist()]
            return DerivativeCheckResponse(
                equivalent=False,
                detail=detail,
                symbolic_difference=_format_matrix(difference),
                evaluated_difference=numeric_diff,
            )

    detail = (
        "Jacobian differs algebraically but matched within tolerance across numeric evaluations; "
        "manual inspection recommended."
    )
    evaluated = difference.subs({symbol: 0.0 for symbol in symbols})
    numeric_diff = [[float(sympy.N(item)) for item in row] for row in evaluated.tolist()]
    return DerivativeCheckResponse(
        equivalent=True,
        detail=detail,
        symbolic_difference=_format_matrix(difference),
        evaluated_difference=numeric_diff,
    )
