from matplotlib import pyplot as plt

def plot_robustness(robustness):
    """
    Plots the robustness values over episodes.
    """
    plt.figure(figsize=(10, 5))
    for ep_robustness in robustness:
        plt.plot(ep_robustness, alpha=0.3)

    plt.title("Robustness Over Episodes")
    plt.xlabel("Time Steps (sampled)")
    plt.ylabel("Robustness")
    plt.grid(True)
    plt.savefig("robustness_over_episodes.png")
    plt.show()