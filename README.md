# golden-section-search

Minimizes a unimodal scalar function on a bounded interval using the golden section method, with no derivative information required.

## Usage

```python
from golden_section_search import golden_section_search

f = lambda x: (x - 3.0) ** 2
result = golden_section_search(f, 0.0, 10.0, tol=1e-10)
print(result.x)        # ≈ 3.0
print(result.fx)       # ≈ 0.0
print(result.interval) # final bracketing interval, e.g. (2.99999..., 3.00000...)
print(result.iterations)
```

`golden_section_search(func, a, b, *, tol=1e-8, max_iter=100)` returns a `SearchResult` (a `NamedTuple`) with fields `x`, `fx`, `iterations`, and `interval`.

## Why

Derivative-free 1D minimization is needed when the objective is a black-box or noisy function whose gradient is unavailable or unreliable. The golden section search converges reliably for any unimodal function at a fixed linear rate (~0.382 per iteration) and needs only one new function evaluation per step after the initial two. The trade-off versus Brent's method is simplicity and robustness: no parabolic interpolation steps that can fail on flat or noisy functions, at the cost of slower convergence on smooth functions.

## Edges and limitations

The function must be **unimodal** on the interval. This implementation does not verify that property; if the function has multiple local minima, the result is some point in the final interval but not necessarily the global minimum. The caller is responsible for ensuring unimodality.

The returned minimizer estimate `x` is the **midpoint of the final bracketing interval**, not one of the sampled points. For a truly unimodal function the minimum is interior to the bracket, so the midpoint is an unbiased estimate with error bounded by half the final interval width. If the true minimum sits at an endpoint (a non-unimodal or boundary case), the midpoint will never exactly equal it.

`max_iter` caps the number of interval reductions. If the interval has not converged to `tol` by then, the result reflects the best bracket found so far rather than raising.
