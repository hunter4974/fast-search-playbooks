"""Golden section search for minimizing a unimodal scalar function.

The algorithm maintains a bracketing interval [a, b] containing the minimum
of a unimodal function f. Each iteration reduces the interval by a factor of
(1 - 1/phi) ≈ 0.382 by sampling one interior point and reusing a previously
sampled point, so that exactly one new function evaluation is needed per
iteration after the first two.

This implementation returns the midpoint of the final interval as the minimizer
estimate (rather than one of the endpoints), because for a unimodal function
the true minimum lies strictly inside the final bracket and the midpoint is an
unbiased estimator with error bounded by half the interval width.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple


class SearchResult(NamedTuple):
    """Outcome of a golden section search.

    x is the minimizer estimate; fx is f(x); iterations is the number of
    interval reductions performed (not counting the initial two samples);
    interval is the final bracketing interval as a (low, high) tuple where
    low < high.
    """

    x: float
    fx: float
    iterations: int
    interval: tuple[float, float]


# Golden ratio conjugate; the reduction factor per iteration.
# 1/phi == phi - 1 == (sqrt(5) - 1) / 2 ≈ 0.3819660112501051
_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0
# Complementary factor so that (low, high) is split into two segments whose
# ratio matches the whole. _INV_PHI2 == 1 - 1/phi == 1/phi**2.
_INV_PHI2 = _INV_PHI * _INV_PHI


def golden_section_search(
    func: Callable[[float], float],
    a: float,
    b: float,
    *,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> SearchResult:
    """Minimize a unimodal function on the interval [a, b].

    Args:
        func: A unimodal scalar function to minimize. It must be defined on
            all of [a, b] and must not return NaN for finite inputs in that
            range.
        a: Left endpoint of the initial bracketing interval.
        b: Right endpoint of the initial bracketing interval. Must satisfy
            b > a.
        tol: Stop when the interval width is <= tol. Must be a positive,
            finite number.
        max_iter: Hard cap on the number of interval reductions. Guards
            against pathological or non-unimodal inputs that would otherwise
            prevent convergence.

    Returns:
        A SearchResult with the minimizer estimate x, the function value at
        x, the number of reductions performed, and the final interval.

    Raises:
        ValueError: If the interval is degenerate, endpoints are non-finite,
            tol is non-positive or non-finite, or max_iter < 1.

    Notes:
        This implementation does NOT verify unimodality. If the supplied
        function has multiple local minima, the algorithm will converge to
        some point within the final interval, but not necessarily the global
        minimum. The caller is responsible for ensuring unimodality.
    """
    if not math.isfinite(a):
        raise ValueError(f"a must be finite, got {a!r}")
    if not math.isfinite(b):
        raise ValueError(f"b must be finite, got {b!r}")
    if b <= a:
        raise ValueError(f"require b > a, got a={a!r}, b={b!r}")
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be a positive finite number, got {tol!r}")
    if not isinstance(max_iter, int):
        raise TypeError(f"max_iter must be int, got {type(max_iter).__name__}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")

    low, high = a, b

    # Initial interior samples. x1 is the left probe, x2 the right probe.
    # Using the golden ratio ensures that as the interval shrinks, one of the
    # old probes can be reused as a new probe, so each iteration after the
    # first needs only one new function evaluation.
    x1 = high - _INV_PHI * (high - low)
    x2 = low + _INV_PHI * (high - low)
    f1 = func(x1)
    f2 = func(x2)

    if math.isnan(f1):
        raise ValueError(f"func({x1!r}) returned NaN")
    if math.isnan(f2):
        raise ValueError(f"func({x2!r}) returned NaN")

    iterations = 0
    width = high - low

    while width > tol and iterations < max_iter:
        iterations += 1
        if f1 < f2:
            # Minimum lies in [low, x2]; discard the right segment.
            high = x2
            x2 = x1
            f2 = f1
            x1 = high - _INV_PHI * (high - low)
            f1 = func(x1)
        else:
            # Minimum lies in [x1, high]; discard the left segment.
            low = x1
            x1 = x2
            f1 = f2
            x2 = low + _INV_PHI * (high - low)
            f2 = func(x2)

        if math.isnan(f1):
            raise ValueError(f"func({x1!r}) returned NaN")
        if math.isnan(f2):
            raise ValueError(f"func({x2!r}) returned NaN")

        width = high - low

    # Midpoint of the final bracket: for a truly unimodal function the minimum
    # is interior, so the midpoint bounds error symmetrically by width/2.
    x = 0.5 * (low + high)
    fx = func(x)
    return SearchResult(x=x, fx=fx, iterations=iterations, interval=(low, high))
