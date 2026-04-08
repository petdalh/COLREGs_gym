import numpy as np
import matplotlib.pyplot as plt


def plot_episode_trajectory(
    ego_traj: list,
    enc_traj: list,
    goal: tuple,
    dt: float = 0.5,
    episode_num: int = 1,
    dot_every: int = 40,
    collision_radius: float = 6.0,
    save_path: str | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 8))

    ego_arr = np.array(ego_traj)
    enc_arr = np.array(enc_traj)

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
