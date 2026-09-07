"""Small, dependency-free stats helpers shared by run_eval.py and compare.py.

Wilson score interval for a binomial proportion and the Newcombe (1998, method 10)
hybrid-score interval for the difference of two independent proportions.
"""
import math

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95):
    """Wilson 95% CI for k successes out of n. Returns (p_hat, lo, hi)."""
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def newcombe_diff(k1: int, n1: int, k2: int, n2: int, z: float = Z95):
    """Newcombe hybrid-score 95% CI for p1 - p2 (independent samples).

    Returns (diff, lo, hi). p1 = k1/n1 ("post"), p2 = k2/n2 ("pre") by convention.
    """
    if n1 <= 0 or n2 <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p1, l1, u1 = wilson(k1, n1, z)
    p2, l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (d, max(-1.0, lo), min(1.0, hi))


def two_prop_z_pvalue(k1: int, n1: int, k2: int, n2: int):
    """Two-sided pooled two-proportion z-test p-value (normal approx)."""
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    zval = (p1 - p2) / se
    # two-sided normal tail via erfc
    return math.erfc(abs(zval) / math.sqrt(2))


def fisher_exact_pvalue(k1: int, n1: int, k2: int, n2: int):
    """Two-sided Fisher exact test p-value; uses scipy if available, else None."""
    try:
        from scipy.stats import fisher_exact  # type: ignore
    except Exception:
        return None
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    return float(fisher_exact(table, alternative="two-sided")[1])
