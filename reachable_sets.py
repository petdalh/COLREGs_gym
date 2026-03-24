import os
import pickle
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
home = os.getenv("PYTHONPATH")
if not home:
    raise EnvironmentError("PYTHONPATH not set in environment variables.")

PACSTL_ELLIPSOIDS_DIR = Path(home) / "COLREGs_gym" / "reachable_sets"

def _load_pickle(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def preload_reachable_sets():
    """Load all 4 speed-band reachable set dicts from pacSTL pickle files.

    vessel_type: "voyager" or "drill"
    Returns a list of 4 dicts indexed [0..3] for speed bands u1..u4.
    """
    prefix = "reachable_sets_u"
    return [
        _load_pickle(str(PACSTL_ELLIPSOIDS_DIR / f"{prefix}{i + 1}.pkl"))
        for i in range(4)
    ]


def select_reachable_set(ellipsoids_Ab_dicts, speed):
    """Return the reachable set dict matching the target vessel's speed band.

    Mirrors the threshold logic in pacSTL/sim_experiments/main.py:
      speed < 0.1  -> u1,  speed < 0.3 -> u2,
      speed < 0.5  -> u3,  else        -> u4
    """
    if speed < 0.1:
        return ellipsoids_Ab_dicts[0]
    elif speed < 0.3:
        return ellipsoids_Ab_dicts[1]
    elif speed < 0.5:
        return ellipsoids_Ab_dicts[2]
    else:
        return ellipsoids_Ab_dicts[3]
