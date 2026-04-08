import pickle
from pathlib import Path


REACHABLE_SETS_DIR = Path(__file__).resolve().parents[2] / "reachable_sets"


def _load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def preload_reachable_sets():
    """Load all 4 speed-band reachable set dicts from pacSTL pickle files."""
    prefix = "reachable_sets_u"
    return [
        _load_pickle(REACHABLE_SETS_DIR / f"{prefix}{i + 1}.pkl")
        for i in range(4)
    ]


def select_reachable_set(ellipsoids_Ab_dicts, speed):
    """Return the reachable set dict matching the target vessel's speed band."""
    if speed < 0.1:
        return ellipsoids_Ab_dicts[0]
    if speed < 0.3:
        return ellipsoids_Ab_dicts[1]
    if speed < 0.5:
        return ellipsoids_Ab_dicts[2]
    return ellipsoids_Ab_dicts[3]
