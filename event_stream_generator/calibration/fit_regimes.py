"""Fit simple temporal regimes to inter-event-time samples."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple


def fit_best_regime(
    samples: Iterable[float],
    *,
    min_samples: int = 5,
) -> Dict[str, Any]:
    values = [float(item) for item in samples if float(item) > 0.0 and math.isfinite(float(item))]
    if len(values) < min_samples:
        return {
            "regime": "generic_backoff",
            "params": {},
            "sample_count": len(values),
            "fit": {
                "status": "insufficient_samples",
                "min_samples": min_samples,
            },
        }
    if max(values) - min(values) <= 1e-12:
        return {
            "regime": "deterministic_backoff",
            "params": {"value": round(values[0], 9)},
            "sample_count": len(values),
            "fit": {
                "status": "zero_variance_samples",
                "min_samples": min_samples,
            },
        }
    try:
        from scipy import stats
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for calibration fitting. Install with: pip install scipy"
        ) from exc

    candidates = []
    for fitter in (
        _fit_exponential,
        _fit_gamma,
        _fit_lognormal,
        _fit_weibull,
    ):
        try:
            candidates.append(fitter(values, stats))
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
            continue
    if not candidates:
        return {
            "regime": "generic_backoff",
            "params": {},
            "sample_count": len(values),
            "fit": {
                "status": "all_fit_attempts_failed",
                "min_samples": min_samples,
            },
        }
    best = min(candidates, key=lambda item: (item["fit"]["ks_statistic"], item["fit"]["aic"]))
    best["sample_count"] = len(values)
    return best


def _fit_exponential(values: List[float], stats: Any) -> Dict[str, Any]:
    loc, scale = stats.expon.fit(values, floc=0.0)
    del loc
    params = {"lambda": round(1.0 / max(float(scale), 1e-12), 9)}
    return _fit_record("exponential", params, stats.expon, (0.0, scale), values)


def _fit_gamma(values: List[float], stats: Any) -> Dict[str, Any]:
    shape, loc, scale = stats.gamma.fit(values, floc=0.0)
    del loc
    params = {"shape": round(float(shape), 9), "scale": round(float(scale), 9)}
    return _fit_record("gamma", params, stats.gamma, (shape, 0.0, scale), values)


def _fit_lognormal(values: List[float], stats: Any) -> Dict[str, Any]:
    sigma, loc, scale = stats.lognorm.fit(values, floc=0.0)
    del loc
    mu = math.log(max(float(scale), 1e-12))
    params = {"mu": round(mu, 9), "sigma": round(float(sigma), 9)}
    return _fit_record("lognormal", params, stats.lognorm, (sigma, 0.0, scale), values)


def _fit_weibull(values: List[float], stats: Any) -> Dict[str, Any]:
    shape, loc, scale = stats.weibull_min.fit(values, floc=0.0)
    del loc
    params = {"shape": round(float(shape), 9), "scale": round(float(scale), 9)}
    return _fit_record("weibull", params, stats.weibull_min, (shape, 0.0, scale), values)


def _fit_record(
    regime: str,
    params: Dict[str, float],
    distribution: Any,
    distribution_args: Tuple[float, ...],
    values: List[float],
) -> Dict[str, Any]:
    from scipy import stats

    ks = stats.kstest(values, distribution.cdf, args=distribution_args)
    log_likelihood = sum(
        math.log(max(float(distribution.pdf(value, *distribution_args)), 1e-300))
        for value in values
    )
    param_count = len(distribution_args) - 1
    n = len(values)
    aic = 2 * param_count - 2 * log_likelihood
    bic = math.log(n) * param_count - 2 * log_likelihood
    return {
        "regime": regime,
        "params": params,
        "fit": {
            "status": "fitted",
            "ks_statistic": round(float(ks.statistic), 9),
            "ks_pvalue": round(float(ks.pvalue), 9),
            "aic": round(float(aic), 6),
            "bic": round(float(bic), 6),
        },
    }
