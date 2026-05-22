from ast import literal_eval
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from gym.utils.robustness import extract_robustness_lower, extract_robustness_upper

# ── Publication style ────────────────────────────────────────────────
# Use Computer Modern (LaTeX default) so figures match paper body text.
# Falls back gracefully if CM isn't installed.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "text.usetex": False,  # set True if LaTeX is available
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#333333",
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.labelpad": 4,
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.linewidth": 0.35,
        "grid.alpha": 0.45,
        "grid.color": "#bbbbbb",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 7.5,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#cccccc",
        "legend.fancybox": False,
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)

# ── Colour palette (muted, colour-blind friendly) ───────────────────
_EGO_CLR = "#2060a8"  # steel blue
_OBS_CLR = "#c44e52"  # muted red
_GOAL_CLR = "#4a9c5e"  # forest green
_CMAP = "viridis"
_DEFAULT_COMMAND_FRACTIONS = np.linspace(0.0, 1.0, 5, dtype=float)
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configuration" / "config.yaml"


def _parse_surge_command_bounds(config_path: Path) -> np.ndarray:
    for line in config_path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line.startswith("surge_accel_commands:"):
            continue

        surge_accel_commands = literal_eval(line.split(":", 1)[1].strip())
        if np.asarray(surge_accel_commands, dtype=float).size:
            return _DEFAULT_COMMAND_FRACTIONS

    raise KeyError("surge_accel_commands")


def _load_command_fractions(config_path: Path = _DEFAULT_CONFIG_PATH) -> np.ndarray:
    try:
        try:
            from gym.utils.config import load_config

            load_config(str(config_path))["environment_configuration"]["action_configuration"][
                "surge_accel_commands"
            ]
            command_fractions = _DEFAULT_COMMAND_FRACTIONS
        except ModuleNotFoundError:
            command_fractions = _parse_surge_command_bounds(config_path)
        command_fractions = np.asarray(command_fractions, dtype=float)
    except (FileNotFoundError, KeyError, SyntaxError, TypeError, ValueError):
        return _DEFAULT_COMMAND_FRACTIONS

    if command_fractions.size == 0 or not np.all(np.isfinite(command_fractions)):
        return _DEFAULT_COMMAND_FRACTIONS
    return command_fractions


_COMMAND_FRACTIONS = _load_command_fractions()
_SPEED_NORM = Normalize(
    vmin=float(np.min(_COMMAND_FRACTIONS)),
    vmax=float(np.max(_COMMAND_FRACTIONS)),
)


def plot_episode_trajectory(
    ego_traj: list,
    enc_traj: list,
    goal: tuple,
    dt: float = 0.5,
    episode_num: int = 1,
    dot_every: int = 120,
    collision_radius: float = 0.0,
    save_path: str | None = None,
    ego_speed_traj: list | None = None,
    obstacle_speed_mps: float | None = None,
    terminal_reason: str | None = None,
    final_goal_distance: float | None = None,
    final_encounter_distance: float | None = None,
):
    # ── Figure (single-column width ≈ 3.5 in for two-column papers) ──
    fig, ax = plt.subplots(figsize=(3.6, 3.6))

    ego_arr = np.array(ego_traj)
    enc_arr = np.array(enc_traj)
    ego_speed_arr = None
    if ego_speed_traj is not None:
        ego_speed_arr = np.asarray(ego_speed_traj, dtype=float)
        if len(ego_speed_arr) != len(ego_arr):
            n = min(len(ego_arr), len(ego_speed_arr))
            ego_arr = ego_arr[:n]
            ego_speed_arr = ego_speed_arr[:n]

    # ── Ego trajectory (speed-coloured or solid) ─────────────────────
    if ego_speed_arr is not None and len(ego_arr) > 1:
        segments = np.stack(
            [ego_arr[:-1, [1, 0]], ego_arr[1:, [1, 0]]],
            axis=1,
        )
        segment_values = ego_speed_arr[1:]
        finite_mask = np.isfinite(segment_values) & np.all(
            np.isfinite(segments.reshape(-1, 4)), axis=1
        )

        if np.any(finite_mask):
            finite_values = segment_values[finite_mask]

            lc = LineCollection(
                segments[finite_mask],
                cmap=_CMAP,
                norm=_SPEED_NORM,
                linewidth=1.2,
                zorder=3,
                capstyle="round",
            )
            lc.set_array(finite_values)
            ax.add_collection(lc)
            # invisible line just for the legend entry
            ax.plot([], [], color=_EGO_CLR, linewidth=1.2, label="Ego vessel")
            cbar = fig.colorbar(lc, ax=ax, pad=0.03, fraction=0.046, aspect=28)
            cbar.set_ticks(_COMMAND_FRACTIONS)
            cbar.ax.tick_params(labelsize=7, width=0.4, length=2)
            cbar.set_label(r"$u_d / v_{\max}$", fontsize=8, labelpad=3)
            cbar.outline.set_linewidth(0.4)
        else:
            ax.plot(
                ego_arr[:, 1], ego_arr[:, 0],
                color=_EGO_CLR, linewidth=1.2, label="Ego vessel",
            )
    else:
        ax.plot(
            ego_arr[:, 1], ego_arr[:, 0],
            color=_EGO_CLR, linewidth=1.2, label="Ego vessel",
        )

    # ── Obstacle trajectory ──────────────────────────────────────────
    if len(enc_arr) > 0:
        obstacle_label = "Obstacle vessel"
        if obstacle_speed_mps is not None and np.isfinite(obstacle_speed_mps):
            obstacle_label = rf"Obstacle vessel ($v_o={obstacle_speed_mps:.2f}$ m/s)"
        ax.plot(
            enc_arr[:, 1], enc_arr[:, 0],
            color=_OBS_CLR, linewidth=1.2, label=obstacle_label,
        )

    # ── Start / end markers ──────────────────────────────────────────
    ax.plot(ego_arr[0, 1], ego_arr[0, 0], "o",
            color=_EGO_CLR, markersize=5, markeredgecolor="white",
            markeredgewidth=0.5, zorder=6, label="Ego start")
    if len(enc_arr) > 0:
        ax.plot(enc_arr[0, 1], enc_arr[0, 0], "o",
                color=_OBS_CLR, markersize=5, markeredgecolor="white",
                markeredgewidth=0.5, zorder=6, label="Obstacle start")

    ax.plot(ego_arr[-1, 1], ego_arr[-1, 0], "x",
            color=_EGO_CLR, markersize=6, markeredgewidth=1.2, zorder=6)
    if len(enc_arr) > 0:
        ax.plot(enc_arr[-1, 1], enc_arr[-1, 0], "x",
                color=_OBS_CLR, markersize=6, markeredgewidth=1.2, zorder=6)

    # ── Time dots ────────────────────────────────────────────────────
    ego_dot_idx = np.arange(0, len(ego_arr), dot_every)
    enc_dot_idx = np.arange(0, len(enc_arr), dot_every) if len(enc_arr) > 0 else []

    if ego_speed_arr is not None and np.any(np.isfinite(ego_speed_arr)):
        dot_mask = np.isfinite(ego_speed_arr[ego_dot_idx])
        ego_dot_valid = ego_dot_idx[dot_mask]
        if len(ego_dot_valid) > 0:
            speed_vals = ego_speed_arr[ego_dot_valid]
            ax.scatter(
                ego_arr[ego_dot_valid, 1], ego_arr[ego_dot_valid, 0],
                c=speed_vals, cmap=_CMAP, norm=_SPEED_NORM,
                s=12, linewidths=0.3, edgecolors="white", zorder=5,
            )
    else:
        ax.scatter(
            ego_arr[ego_dot_idx, 1], ego_arr[ego_dot_idx, 0],
            color=_EGO_CLR, s=12, linewidths=0.3, edgecolors="white", zorder=5,
        )

    if len(enc_arr) > 0:
        ax.scatter(
            enc_arr[enc_dot_idx, 1], enc_arr[enc_dot_idx, 0],
            color=_OBS_CLR, s=12, linewidths=0.3, edgecolors="white", zorder=5,
        )
        for idx in enc_dot_idx:
            circle = plt.Circle(
                (enc_arr[idx, 1], enc_arr[idx, 0]),
                radius=collision_radius,
                color=_OBS_CLR,
                fill=False,
                linewidth=0.45,
                linestyle="--",
                alpha=0.35,
                zorder=4,
            )
            ax.add_patch(circle)

        final_circle = plt.Circle(
            (enc_arr[-1, 1], enc_arr[-1, 0]),
            radius=collision_radius,
            color=_OBS_CLR,
            fill=False,
            linewidth=0.8,
            linestyle="-",
            alpha=0.75,
            zorder=4,
        )
        ax.add_patch(final_circle)

    # ── Time annotations ─────────────────────────────────────────────
    for idx in ego_dot_idx:
        ax.annotate(
            f"{idx * dt:.0f}s",
            (ego_arr[idx, 1], ego_arr[idx, 0]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=5.5,
            color=_EGO_CLR,
        )
    if len(enc_arr) > 0:
        for idx in enc_dot_idx:
            ax.annotate(
                f"{idx * dt:.0f}s",
                (enc_arr[idx, 1], enc_arr[idx, 0]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=5.5,
                color=_OBS_CLR,
            )

    # ── Goal marker ──────────────────────────────────────────────────
    ax.plot(goal[1], goal[0], "*",
            color=_GOAL_CLR, markersize=10, markeredgecolor="white",
            markeredgewidth=0.4, zorder=7, label="Goal")

    # ── Axis labels & legend ─────────────────────────────────────────
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    title_parts = [f"Episode {episode_num}"]
    if terminal_reason:
        title_parts.append(str(terminal_reason))
    distance_parts = []
    if final_goal_distance is not None and np.isfinite(final_goal_distance):
        distance_parts.append(f"d_goal={final_goal_distance:.1f} m")
    if final_encounter_distance is not None and np.isfinite(final_encounter_distance):
        distance_parts.append(f"d_enc={final_encounter_distance:.1f} m")
    if distance_parts:
        title_parts.append(", ".join(distance_parts))
    ax.set_title(" | ".join(title_parts), fontweight="medium", pad=6)
    ax.set_aspect("equal")
    ax.tick_params(top=True, right=True, which="both")

    # Place legend as a compact horizontal strip below the x-axis label
    # so it never occludes trajectories.  We split into two rows (ncol=3)
    # to avoid running into the colourbar on the right.
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.4,
        borderaxespad=0.0,
        frameon=False,
    )

    if save_path:
        # bbox_extra_artists tells savefig to expand the bounding box
        # so the external legend is never clipped — no manual
        # subplots_adjust needed.
        fig.savefig(
            save_path, format="png",
            bbox_extra_artists=[leg], bbox_inches="tight",
        )
        plt.close(fig)
    else:
        plt.show()


def plot_surge_command_fraction(
    history_surge_command_fraction: list,
    dt: float = 0.5,
    save_path: str | None = None,
):
    arr = np.asarray(history_surge_command_fraction, dtype=float)
    times = np.arange(len(arr)) * dt
    valid = np.isfinite(arr)

    if not np.any(valid):
        return

    fig, ax = plt.subplots(figsize=(3.6, 1.6))
    fig.subplots_adjust(bottom=0.28, top=0.95, left=0.12, right=0.97)

    ax.step(times[valid], arr[valid], where="post", color=_EGO_CLR, linewidth=0.8)
    ax.set_ylabel(r"$u_d / v_{\max}$")
    ax.set_xlabel(r"$t$ (s)")
    ax.tick_params(top=True, right=True, which="both")

    if save_path:
        fig.savefig(save_path, format="png")
        plt.close(fig)
    else:
        plt.show()


def plot_control_timeseries(
    history_heading_deg: list,
    history_heading_cmd_deg: list,
    history_heading_error_deg: list,
    history_surge: list,
    history_surge_cmd: list,
    history_tau_surge: list,
    history_tau_yaw: list,
    dt: float = 0.5,
    save_path: str | None = None,
):
    n = len(history_heading_deg)
    if n == 0:
        return

    times = np.arange(n) * dt
    hdg = np.asarray(history_heading_deg, dtype=float)
    hdg_cmd = np.asarray(history_heading_cmd_deg, dtype=float)
    hdg_err = np.asarray(history_heading_error_deg, dtype=float)
    surge = np.asarray(history_surge, dtype=float)
    surge_cmd = np.asarray(history_surge_cmd, dtype=float)
    tau_s = np.asarray(history_tau_surge, dtype=float)
    tau_y = np.asarray(history_tau_yaw, dtype=float)

    _CMD_CLR = "#888888"
    _ERR_CLR = "#c44e52"

    fig, axes = plt.subplots(4, 1, figsize=(3.6, 6.0), sharex=True)
    fig.subplots_adjust(hspace=0.12, left=0.14, right=0.97, top=0.95, bottom=0.07)

    # Panel 1 — heading
    axes[0].plot(times, hdg, color=_EGO_CLR, linewidth=0.9, label="Actual")
    axes[0].plot(times, hdg_cmd, color=_CMD_CLR, linewidth=0.7, linestyle="--", label="Commanded")
    axes[0].set_ylabel("Heading (°)")
    axes[0].legend(loc="upper right")
    axes[0].tick_params(top=True, right=True, which="both")

    # Panel 2 — heading error
    axes[1].plot(times, hdg_err, color=_ERR_CLR, linewidth=0.9)
    axes[1].axhline(y=0, color="#555555", linewidth=0.45, linestyle="--", zorder=1)
    axes[1].fill_between(times, hdg_err, 0, alpha=0.15, color=_ERR_CLR)
    axes[1].set_ylabel(r"$\psi_d - \psi$ (°)")
    axes[1].tick_params(top=True, right=True, which="both")

    # Panel 3 — surge velocity
    alpha_ema = 0.1  # ~5 s time constant at dt=0.5, matching MRAC reference model
    def _ema(x):
        out = np.zeros_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = alpha_ema * x[i] + (1 - alpha_ema) * out[i - 1]
        return out
    surge_filtered = _ema(surge)
    surge_cmd_filtered = surge_cmd
    axes[2].plot(times, surge_filtered, color=_EGO_CLR, linewidth=0.9, label="Actual surge")
    axes[2].step(times, surge_cmd, where="post", color=_CMD_CLR,
                 linewidth=0.5, linestyle="--", alpha=0.35, label="_nolegend_")
    axes[2].plot(times, surge_cmd_filtered, color=_CMD_CLR,
                 linewidth=0.9, linestyle="--", label="Commanded surge (filtered)")
    axes[2].set_ylabel("Surge (m/s)")
    axes[2].legend(loc="upper right")
    axes[2].tick_params(top=True, right=True, which="both")

    # Panel 4 — actuator effort
    axes[3].plot(times, tau_s, color=_EGO_CLR, linewidth=0.9, label=r"$\tau_X$")
    axes[3].plot(times, tau_y, color=_OBS_CLR, linewidth=0.9, label=r"$\tau_N$")
    axes[3].axhline(y=0, color="#555555", linewidth=0.45, linestyle="--", zorder=1)
    axes[3].set_ylabel("Actuator effort")
    axes[3].set_xlabel(r"$t$ (s)")
    axes[3].legend(loc="upper right")
    axes[3].tick_params(top=True, right=True, which="both")

    if save_path:
        fig.savefig(save_path, format="png")
        plt.close(fig)
    else:
        plt.show()


def plot_robustness(
    ep_robustness: list,
    dt: float = 0.5,
    save_path: str | None = None,
    title: str | None = None,
):
    times, lowers, uppers = [], [], []

    for step_idx, rob in enumerate(ep_robustness):
        if rob is None:
            continue
        lower = extract_robustness_lower(rob)
        upper = extract_robustness_upper(rob)
        if lower is None or upper is None:
            continue
        times.append(step_idx * dt)
        lowers.append(lower)
        uppers.append(upper)

    if not times:
        print("No robustness data to plot.")
        return

    times = np.array(times)
    lowers = np.array(lowers)
    uppers = np.array(uppers)

    # ── Single-column width, short height (good for stacking) ────────
    fig, ax = plt.subplots(figsize=(3.6, 1.6))
    fig.subplots_adjust(bottom=0.28, top=0.95, left=0.12, right=0.97)

    ax.fill_between(times, lowers, uppers, color=_EGO_CLR, alpha=0.15, zorder=2)
    ax.plot(times, lowers, color=_EGO_CLR, linewidth=0.8, zorder=3)
    ax.plot(times, uppers, color=_EGO_CLR, linewidth=0.8, zorder=3)
    ax.axhline(y=0, color="#555555", linewidth=0.45, linestyle="--", zorder=1)

    ax.set_ylabel(r"$[\rho]$")
    ax.set_xlabel(r"$t$ (s)")
    if title:
        ax.set_title(title, fontweight="medium", pad=4)
    ax.tick_params(top=True, right=True, which="both")

    if save_path:
        fig.savefig(save_path, format="png")
        plt.close(fig)
    else:
        plt.show()
