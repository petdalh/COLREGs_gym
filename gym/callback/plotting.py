import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def plot_episode_trajectory(
    ego_traj: list,
    enc_traj: list,
    goal: tuple,
    dt: float = 0.5,
    episode_num: int = 1,
    dot_every: int = 40,
    collision_radius: float = 6.0,
    save_path: str | None = None,
    ego_speed_traj: list | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 8))

    ego_arr = np.array(ego_traj)
    enc_arr = np.array(enc_traj)
    ego_speed_arr = None
    if ego_speed_traj is not None:
        ego_speed_arr = np.asarray(ego_speed_traj, dtype=float)
        if len(ego_speed_arr) != len(ego_arr):
            n = min(len(ego_arr), len(ego_speed_arr))
            ego_arr = ego_arr[:n]
            ego_speed_arr = ego_speed_arr[:n]

    if ego_speed_arr is not None and len(ego_arr) > 1:
        segments = np.stack(
            [
                ego_arr[:-1, [1, 0]],
                ego_arr[1:, [1, 0]],
            ],
            axis=1,
        )
        segment_values = ego_speed_arr[1:]
        finite_mask = np.isfinite(segment_values) & np.all(
            np.isfinite(segments.reshape(-1, 4)), axis=1
        )

        if np.any(finite_mask):
            finite_values = segment_values[finite_mask]
            vmin = float(np.min(finite_values))
            vmax = float(np.max(finite_values))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-9

            norm = Normalize(vmin=vmin, vmax=vmax)
            lc = LineCollection(
                segments[finite_mask],
                cmap="viridis",
                norm=norm,
                linewidth=2.0,
                zorder=3,
            )
            lc.set_array(finite_values)
            ax.add_collection(lc)
            ax.plot([], [], color="black", linewidth=2.0, label="Ego vessel")
            cbar = fig.colorbar(lc, ax=ax, pad=0.02)
            cbar.set_label("Speed multiplier")
        else:
            ax.plot(
                ego_arr[:, 1],
                ego_arr[:, 0],
                "b-",
                linewidth=1.5,
                label="Ego vessel",
            )
    else:
        ax.plot(ego_arr[:, 1], ego_arr[:, 0], "b-", linewidth=1.5, label="Ego vessel")

    if len(enc_arr) > 0:
        ax.plot(enc_arr[:, 1], enc_arr[:, 0], "r-", linewidth=1.5, label="Obstacle vessel")

    ax.plot(ego_arr[0, 1], ego_arr[0, 0], "bo", markersize=8, label="Ego start")
    if len(enc_arr) > 0:
        ax.plot(enc_arr[0, 1], enc_arr[0, 0], "ro", markersize=8, label="Obstacle start")

    ax.plot(ego_arr[-1, 1], ego_arr[-1, 0], "bx", markersize=10, markeredgewidth=2)
    if len(enc_arr) > 0:
        ax.plot(enc_arr[-1, 1], enc_arr[-1, 0], "rx", markersize=10, markeredgewidth=2)

    ego_dot_indices = np.arange(0, len(ego_arr), dot_every)
    enc_dot_indices = np.arange(0, len(enc_arr), dot_every) if len(enc_arr) > 0 else []

    if ego_speed_arr is not None and np.any(np.isfinite(ego_speed_arr)):
        dot_mask = np.isfinite(ego_speed_arr[ego_dot_indices])
        ego_dot_indices_valid = ego_dot_indices[dot_mask]
        if len(ego_dot_indices_valid) > 0:
            speed_values = ego_speed_arr[ego_dot_indices_valid]
            finite_speed_values = ego_speed_arr[np.isfinite(ego_speed_arr)]
            vmin = float(np.min(finite_speed_values))
            vmax = float(np.max(finite_speed_values))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-9
            norm = Normalize(vmin=vmin, vmax=vmax)
            ax.scatter(
                ego_arr[ego_dot_indices_valid, 1],
                ego_arr[ego_dot_indices_valid, 0],
                c=speed_values,
                cmap="viridis",
                norm=norm,
                s=20,
                zorder=5,
            )
    else:
        ax.scatter(
            ego_arr[ego_dot_indices, 1],
            ego_arr[ego_dot_indices, 0],
            c="blue",
            s=20,
            zorder=5,
        )

    if len(enc_arr) > 0:
        ax.scatter(
            enc_arr[enc_dot_indices, 1],
            enc_arr[enc_dot_indices, 0],
            c="red",
            s=20,
            zorder=5,
        )
        # draw collision radius circle around each obstacle dot
        for idx in enc_dot_indices:
            circle = plt.Circle(
                (enc_arr[idx, 1], enc_arr[idx, 0]),
                radius=collision_radius,
                color="red",
                fill=False,
                linewidth=0.6,
                linestyle="--",
                alpha=0.4,
                zorder=4,
            )
            ax.add_patch(circle)

    for idx in ego_dot_indices:
        ax.annotate(
            f"{idx * dt:.0f}s",
            (ego_arr[idx, 1], ego_arr[idx, 0]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            color="blue",
        )
    if len(enc_arr) > 0:
        for idx in enc_dot_indices:
            ax.annotate(
                f"{idx * dt:.0f}s",
                (enc_arr[idx, 1], enc_arr[idx, 0]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color="red",
            )

    ax.plot(goal[1], goal[0], "g*", markersize=15, label="Goal")

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title(f"Vessel Trajectories - Episode {episode_num}")
    ax.legend()
    ax.grid(True)
    ax.set_aspect("equal")

    if save_path:
        fig.savefig(save_path, format="png", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_robustness(
    ep_robustness: list,
    dt: float = 0.5,
    save_path: str | None = None,
):
    times = []
    lowers = []
    uppers = []

    for step_idx, rob in enumerate(ep_robustness):
        if rob is None:
            continue

        trace_list = rob[0]
        if not trace_list:
            continue

        _, rob_interval = trace_list[0]

        t = step_idx * dt
        times.append(t)
        lowers.append(rob_interval.l)
        uppers.append(rob_interval.u)

    if not times:
        print("No robustness data to plot.")
        return

    times = np.array(times)
    lowers = np.array(lowers)
    uppers = np.array(uppers)

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.21, top=0.99, left=0.08, right=0.99)

    ax.plot(times, lowers, "b", linewidth=1.0)
    ax.plot(times, uppers, "b", linewidth=1.0)
    ax.fill_between(times, lowers, uppers, facecolor="b", alpha=0.25)
    ax.axhline(y=0, color="k", linewidth=0.5, linestyle="--")

    ax.set_ylabel(r"$[\rho]$")
    ax.set_xlabel(r"$t$ (s)")
    ax.grid(True)
    fig.set_figheight(2.0)

    if save_path:
        fig.savefig(save_path, format="png")
        plt.close(fig)
    else:
        plt.show()
