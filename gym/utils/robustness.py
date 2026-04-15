import numpy as np


def _extract_robustness_bound(robustness, bound: str) -> float:
    """Extract 'upper' (.u) or 'lower' (.l) bound from a pacSTL result."""
    attr = bound  # "u" or "l"
    agg = max if bound == "u" else min
    worst = -np.inf if bound == "u" else np.inf

    if robustness is None:
        return None

    if hasattr(robustness, attr):
        return float(getattr(robustness, attr))

    if hasattr(robustness, "__getitem__"):
        try:
            if len(robustness) > 0:
                trace = robustness[0] if isinstance(robustness[0], list) else robustness
                best = worst
                for item in trace:
                    if hasattr(item, "__getitem__") and len(item) >= 2:
                        _, rob_interval = item[0], item[1]
                        if hasattr(rob_interval, attr):
                            best = agg(best, float(getattr(rob_interval, attr)))
                    elif hasattr(item, attr):
                        best = agg(best, float(getattr(item, attr)))
                if np.isfinite(best):
                    return best
        except (IndexError, TypeError):
            pass

    try:
        return float(robustness)
    except (TypeError, ValueError):
        return None


def extract_robustness_upper(robustness) -> float:
    """Extract the upper bound (.u) from a pacSTL robustness result.

    For specs where positive = danger (e.g. crossing_detection), .u is the
    worst-case robustness across obstacle realisations.
    """
    return _extract_robustness_bound(robustness, "u")


def extract_robustness_lower(robustness) -> float:
    """Extract the lower bound (.l) at the initial time point.

    For PAC-STL specs with future-looking operators (e.g. eventually[T, T]),
    RTAMT fills every position after the first with -inf padding.  Only the
    time-0 entry carries the full-trajectory worst-case robustness, so we
    read the first entry directly instead of aggregating with min (which would
    always collapse to -inf).

    A positive lower bound certifies that the spec is satisfied under all
    obstacle realisations (worst-case safety guarantee).
    """
    if robustness is None:
        return None

    # Direct interval object
    if hasattr(robustness, "l"):
        return float(robustness.l)

    # Nested trace format: ([[t0, rob0], [t1, rob1], ...], ...)
    if hasattr(robustness, "__getitem__"):
        try:
            if len(robustness) > 0:
                trace = robustness[0] if isinstance(robustness[0], list) else robustness
                if len(trace) > 0:
                    first = trace[0]
                    if hasattr(first, "__getitem__") and len(first) >= 2:
                        rob_interval = first[1]
                        if hasattr(rob_interval, "l"):
                            return float(rob_interval.l)
                    elif hasattr(first, "l"):
                        return float(first.l)
        except (IndexError, TypeError):
            pass

    try:
        return float(robustness)
    except (TypeError, ValueError):
        return None