import random
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def initialize_fig(subplots, nsteps):
    plt.style.use("grayscale")
    fig, axs = plt.subplots(3, 3, True, True, figsize = (9, 7))
    fig.suptitle("Random Walk", fontsize = 18, y = 0.97)
    for i in subplots:
        axs[i].tick_params(axis = "both", which = "major", labelsize = 8)
        axs[i].set_xticks(np.arange(0, nsteps + 1.1, int((nsteps + 1)/5)))
        axs[i].grid(True, color = "grey", ls = "dotted", lw = 0.2)
        if i in [(j, 0) for j in range(0, 3)]:
            axs[i].set_ylabel("$Walk$, [-]", fontsize = 10)
        if i in [(2, j) for j in range(0, 3)]:
            axs[i].set_xlabel("$Steps$, [-]", fontsize = 10)
    return fig, axs

def plot_results(nsteps, initial, fig, axs, x, ind, value, col):
    steps = [1 if random.randint(0, 1) else -1 for _ in np.arange(0, nsteps)]
    walk = np.concatenate((initial, np.array(steps).cumsum()))
    axs[value].set_title(f"walk{ind+1}(m:{walk.min()} | M:{walk.max()})")
    axs[value].fill_between(x, walk, walk.min(), color = "gray", alpha = .3)
    axs[value].fill_between(x, walk, walk.max(), color = "gray", alpha = .5)
    axs[value].axvline(x = walk.argmin(), ls = "dotted", lw = 1)
    axs[value].axvline(x = walk.argmax(), ls = "dotted", lw = 1)
    axs[value].plot(x, walk, lw = 1, c = col)
    

def main(nsteps, initial, subplots):
    colors = cm.jet(np.linspace(0, 1, len(subplots)))
    x = np.arange(0, nsteps + 1)
    fig, axs = initialize_fig(subplots, nsteps)
    for (ind, value), col in zip(enumerate(subplots), colors):
        plot_results(nsteps, initial, fig, axs, x, ind, value, col)
    plt.show()

if __name__ == "__main__":
    nsteps = 100
    initial = np.zeros(1)
    subplots = [(i, j) for i in range(0, 3) for j in range(0, 3)]
    main(nsteps, initial, subplots)
