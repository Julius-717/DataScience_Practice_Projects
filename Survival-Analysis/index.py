import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def dframe(days):
    d = pd.DataFrame(index=days, columns=[
        "A", "B", "C", "D", "E", "ps", "sA", "sB", "sC", "sD", "sE"], data=1)
    d.index.name = "days"
    d.loc[3:, "A"] = 0
    d.loc[6:, "B"] = 0
    d.loc[5:, "C"] = 0
    d.loc[4:, "D"] = 0
    return d


def symbols(d):
    d.loc[d.A == 1, "sA"], d.loc[d.A == 0, "sA"] = "bo", "rs"
    d.loc[d.B == 1, "sB"], d.loc[d.B == 0, "sB"] = "bo", "rs"
    d.loc[d.C == 1, "sC"], d.loc[d.C == 0, "sC"] = "bo", "rs"
    d.loc[d.D == 1, "sD"], d.loc[d.D == 0, "sD"] = "bo", "rs"
    d.loc[d.E == 1, "sE"], d.loc[d.E == 0, "sE"] = "bo", "rs"
    return d


def survival_prob(d):
    d.ps = [np.sum([row[d.columns[:5]]]) / 5 for index, row in d.iterrows()]
    return d


def initialize_fig():
    plt.style.use("grayscale")
    fig, axs = plt.subplots(2, 1, False, False, figsize=(6, 8))
    fig.suptitle("Survival Analysis", fontsize=18, y=0.94)
    plt.subplots_adjust(hspace=0.3)
    axs[0].set_xlabel("$Days$", fontsize=14)
    axs[0].set_ylabel("$Subject$", fontsize=14)
    axs[0].set_yticklabels([None, "E", "D", "C", "B", "A", None])
    axs[0].tick_params(axis="both", which="major", labelsize=12)
    axs[0].grid(True, color="grey", linestyle="dotted", linewidth=0.3)
    axs[1].set_xlabel("$Days$", fontsize=14)
    axs[1].set_ylabel("Survival\probability$", fontsize=14)
    axs[1].tick_params(axis="both", which="major", labelsize=12)
    axs[1].grid(True, color="grey", linestyle="dotted", linewidth=0.3)
    return fig, axs

def plot_results(fig, axs, d):
    ys = [[i] * len(days) for i in np.arange(10, 0, -2)]
    markers = [i for i in [d.sA, d.sB, d.sC, d.sD, d.sE]]
    for a, b in zip(ys, markers):
        for i, j, k in zip(days, a, b):
            axs[0].plot(i, j, k, ms=20)
    axs[0].scatter(1, 10, marker="o", c="b", label="$Living$")
    axs[0].scatter(1, 10, marker="s", c="r", label="$Dead$")
    axs[0].set_yticks(np.arange(0, 13, 2))
    axs[0].legend(loc="upper right", fontsize=8)
    axs[1].plot(days, d.ps, lw=2, label="$Survival\prob.$")
    axs[1].scatter(days, d.ps, s=20)
    axs[1].legend(loc="best", fontsize=8)

    plt.show()

def main(days):
    d = dframe(days)
    d = symbols(d)
    d = survival_prob(d)
    fig, axs = initialize_fig()
    plot_results(fig, axs, d)
