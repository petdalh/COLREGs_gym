import numpy as np


ISTL_STATE_COMPONENTS = ("x", "y", "heading", "vx", "vy")
ISTL_SEMANTICS = "istl"
PACSTL_SEMANTICS = "pacstl"


def normalize_stl_semantics(value) -> str:
    semantics = str(value or PACSTL_SEMANTICS).strip().lower()
    if semantics not in {PACSTL_SEMANTICS, ISTL_SEMANTICS}:
        raise ValueError(
            "monitoring_configuration.stl_semantics must be one of "
            f"{[PACSTL_SEMANTICS, ISTL_SEMANTICS]}, got {value!r}."
        )
    return semantics


def default_istl_state_noise() -> dict:
    return {
        "ego": {component: 0.0 for component in ISTL_STATE_COMPONENTS},
        "obstacle": {component: 0.0 for component in ISTL_STATE_COMPONENTS},
    }


def normalize_istl_state_noise(config=None) -> dict:
    normalized = default_istl_state_noise()
    config = dict(config or {})
    for actor in ("ego", "obstacle"):
        actor_config = dict(config.get(actor, {}))
        for component in ISTL_STATE_COMPONENTS:
            value = float(actor_config.get(component, normalized[actor][component]))
            if value < 0.0:
                raise ValueError(
                    f"istl_state_noise.{actor}.{component} must be >= 0, got {value}."
                )
            normalized[actor][component] = value
    return normalized


def effective_istl_state_noise(base_noise: dict, growth_per_s: dict, time_step: float):
    effective = {}
    for component in ISTL_STATE_COMPONENTS:
        effective[component] = (
            float(base_noise.get(component, 0.0))
            + float(growth_per_s.get(component, 0.0)) * float(time_step)
        )
    return effective


def default_istl_time_steps(decision_depth: int, action_hold_dt: float) -> list[float]:
    return [
        float(depth_idx + 1) * float(action_hold_dt)
        for depth_idx in range(int(decision_depth))
    ]


def state_to_interval_state(
    time_step: float,
    nominal_state,
    noise: dict,
    noise_growth_per_s: dict | None = None,
):
    import interval
    from pacstl.common.interfaces import TimeStampedState

    nominal = np.asarray(nominal_state, dtype=float)
    if noise_growth_per_s is not None:
        noise = effective_istl_state_noise(noise, noise_growth_per_s, time_step)
    values = []
    for idx, component in enumerate(ISTL_STATE_COMPONENTS):
        half_width = float(noise.get(component, 0.0))
        value = float(nominal[idx])
        values.append(interval.interval(value - half_width, value + half_width))
    return TimeStampedState(
        time_step=time_step,
        state_array=np.array(values, dtype=object),
    )


def obstacle_interval_state(
    time_step: float,
    encounter_speed: float,
    noise: dict,
    noise_growth_per_s: dict | None = None,
):
    speed = float(encounter_speed)
    nominal = np.array(
        [
            speed * float(time_step),
            0.0,
            0.0,
            speed,
            0.0,
        ],
        dtype=float,
    )
    return state_to_interval_state(
        time_step,
        nominal,
        noise,
        noise_growth_per_s=noise_growth_per_s,
    )
