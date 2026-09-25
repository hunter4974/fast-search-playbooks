import math
import unittest

from golden_section_search import golden_section_search
from golden_section_search.core import SearchResult


class TestGoldenSectionSearch(unittest.TestCase):
    def test_finds_minimum_of_parabola(self):
        # f(x) = (x - 3)^2, minimum at x = 3.
        f = lambda x: (x - 3.0) ** 2
        result = golden_section_search(f, 0.0, 10.0, tol=1e-10, max_iter=200)

        self.assertIsInstance(result, SearchResult)
        self.assertAlmostEqual(result.x, 3.0, places=6)
        self.assertAlmostEqual(result.fx, 0.0, places=6)
        # Final interval should straddle the true minimum.
        low, high = result.interval
        self.assertLess(low, 3.0)
        self.assertGreater(high, 3.0)
        self.assertLess(high - low, 1e-8 + 1e-12)
        self.assertGreaterEqual(result.iterations, 1)

    def test_finds_minimum_of_shifted_parabola_negative_interval(self):
        # f(x) = (x + 2)^2, minimum at x = -2, interval entirely negative.
        f = lambda x: (x + 2.0) ** 2
        result = golden_section_search(f, -5.0, 0.0, tol=1e-10)
        self.assertAlmostEqual(result.x, -2.0, places=6)

    def test_minimum_at_endpoint_is_unsupported_behavior(self):
        # If the true minimum is at an endpoint, the interval still shrinks
        # toward that endpoint. The returned x should be close to it but not
        # exactly equal because the midpoint of a non-degenerate interval is
        # never an endpoint. We assert the documented behavior: x is the
        # midpoint of the final interval.
        f = lambda x: x  # decreasing on [0, 1], minimum at 0
        result = golden_section_search(f, 0.0, 1.0, tol=1e-6, max_iter=100)
        low, high = result.interval
        expected_x = 0.5 * (low + high)
        self.assertAlmostEqual(result.x, expected_x, places=12)
        self.assertLess(result.x, 1e-5)

    def test_custom_tolerance_controls_interval_width(self):
        f = lambda x: (x - 1.0) ** 2
        result = golden_section_search(f, 0.0, 5.0, tol=0.01, max_iter=200)
        low, high = result.interval
        self.assertLessEqual(high - low, 0.01 + 1e-12)

    def test_default_tolerance(self):
        f = lambda x: (x - 2.0) ** 2
        result = golden_section_search(f, -1.0, 5.0)
        low, high = result.interval
        self.assertLessEqual(high - low, 1e-8 + 1e-12)
        self.assertAlmostEqual(result.x, 2.0, places=5)

    def test_max_iter_caps_iterations(self):
        f = lambda x: (x - 3.0) ** 2
        result = golden_section_search(f, 0.0, 10.0, tol=1e-50, max_iter=5)
        self.assertEqual(result.iterations, 5)
        # Interval will not have converged to 1e-50.
        # After 5 reductions the width is 10 * 0.618**5 ≈ 0.902.
        low, high = result.interval
        self.assertGreater(high - low, 0.9)

    def test_invalid_interval_raises(self):
        f = lambda x: x ** 2
        with self.assertRaises(ValueError):
            golden_section_search(f, 1.0, 1.0)
        with self.assertRaises(ValueError):
            golden_section_search(f, 2.0, 1.0)

    def test_non_finite_endpoints_raise(self):
        f = lambda x: x ** 2
        with self.assertRaises(ValueError):
            golden_section_search(f, float("inf"), 1.0)
        with self.assertRaises(ValueError):
            golden_section_search(f, 0.0, float("nan"))

    def test_invalid_tolerance_raises(self):
        f = lambda x: x ** 2
        with self.assertRaises(ValueError):
            golden_section_search(f, 0.0, 1.0, tol=0.0)
        with self.assertRaises(ValueError):
            golden_section_search(f, 0.0, 1.0, tol=-1.0)
        with self.assertRaises(ValueError):
            golden_section_search(f, 0.0, 1.0, tol=float("inf"))

    def test_invalid_max_iter_raises(self):
        f = lambda x: x ** 2
        with self.assertRaises(ValueError):
            golden_section_search(f, 0.0, 1.0, max_iter=0)
        with self.assertRaises(TypeError):
            golden_section_search(f, 0.0, 1.0, max_iter=1.5)

    def test_nan_function_value_raises(self):
        f = lambda x: float("nan")
        with self.assertRaises(ValueError):
            golden_section_search(f, 0.0, 1.0)

    def test_nan_during_iteration_raises(self):
        # Returns NaN only for interior points the algorithm will probe.
        call_count = [0]
        def f(x):
            call_count[0] += 1
            if call_count[0] > 2:
                return float("nan")
            return (x - 0.5) ** 2
        with self.assertRaises(ValueError):
            golden_section_search(f, 0.0, 1.0, tol=1e-12, max_iter=50)

    def test_flat_function_converges(self):
        # Constant function is unimodal (every point is a minimum). The
        # algorithm should still shrink the interval and return the midpoint.
        f = lambda x: 7.0
        result = golden_section_search(f, 0.0, 1.0, tol=1e-6, max_iter=100)
        self.assertAlmostEqual(result.fx, 7.0, places=12)
        low, high = result.interval
        self.assertLessEqual(high - low, 1e-6 + 1e-12)
        self.assertAlmostEqual(result.x, 0.5 * (low + high), places=12)

    def test_quartic_near_minimum(self):
        # f(x) = (x - 1)^4, flat near minimum; harder for the algorithm.
        f = lambda x: (x - 1.0) ** 4
        result = golden_section_search(f, -2.0, 4.0, tol=1e-10, max_iter=200)
        self.assertAlmostEqual(result.x, 1.0, places=5)
        self.assertGreaterEqual(result.fx, 0.0)

    def test_iterations_count_is_consistent(self):
        f = lambda x: (x - 2.0) ** 2
        result = golden_section_search(f, 0.0, 4.0, tol=1e-10, max_iter=200)
        # The interval starts at width 4 and shrinks by ~0.382 each iteration.
        # log(1e-10 / 4) / log(0.382) ≈ 50. We just check it is reasonable.
        self.assertGreaterEqual(result.iterations, 20)
        self.assertLessEqual(result.iterations, 200)

    def test_narrow_interval(self):
        # Minimum is inside a very narrow initial interval.
        f = lambda x: (x - 0.5000001) ** 2
        result = golden_section_search(f, 0.4999999, 0.5000003, tol=1e-12, max_iter=200)
        low, high = result.interval
        self.assertLess(low, 0.5000001)
        self.assertGreater(high, 0.5000001)
        self.assertLessEqual(high - low, 1e-12 + 1e-15)


if __name__ == "__main__":
    unittest.main()
