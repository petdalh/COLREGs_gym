import numpy as np



def extract_robustness_upper(robustness) -> float:
    """Extract the upper bound from a pacSTL robustness result.

    The evaluator returns different formats depending on the spec
    structure. This normalizes them all to a single float (or None).
    """
    if robustness is None:
        return None

    # Direct interval object with .u attribute
    if hasattr(robustness, "u"):
        return float(robustness.u)

    # List of (time, interval) tuples — take the worst (max) upper bound
    if hasattr(robustness, "__getitem__"):
        try:
            # robustness is [(trace_list)] or similar nested structure
            if len(robustness) > 0:
                trace = robustness[0] if isinstance(robustness[0], list) else robustness
                max_u = -np.inf
                for item in trace:
                    if hasattr(item, "__getitem__") and len(item) >= 2:
                        _, rob_interval = item[0], item[1]
                        if hasattr(rob_interval, "u"):
                            max_u = max(max_u, float(rob_interval.u))
                    elif hasattr(item, "u"):
                        max_u = max(max_u, float(item.u))
                if np.isfinite(max_u):
                    return max_u
        except (IndexError, TypeError):
            pass

    # Bare numeric
    try:
        return float(robustness)
    except (TypeError, ValueError):
        return None