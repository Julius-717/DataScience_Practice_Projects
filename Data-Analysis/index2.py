import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def get_function(xs, ys):
    zs =  np.sqrt(xs**2 + ys**2)
    return zs

def initialize_figure():
    plt.style.use("grayscale")
    fig, axs = plt.subplots(2, 1, True, False, figsize = (5, 7))
    for i in range(2):
        axs[i].set_ylabel("y, [-]")
        if i == 0:
            axs[i].set_title(r"Meshgrid(each $50^{th}$ point)")
        elif i == 0:
            axs[i].set_xlabel("x, [-]")
            axs[i].set_title(r"$\sqrt{x^2+y^2}$ for a grid of values")
    return fig, axs

def plot_results(fig, axs, xs, ys, f_xy):
    axs[0].scatter(xs[::50, ::50], ys[::50, ::50], s = .1, c = "k")
    cb = axs[1].scatter(xs, ys, s = .01, c = f_xy, cmap = plt.cm.jet)
    for i in range(2):
        axs[i].set_xticks(np.arange(-6, 6.1, 2))
        axs[i].set_yticks(np.arange(-6, 6.1, 2))
        axs[i].set_aspect(aspect = 1)
    plt.tight_layout(pad =1)
    plt.colorbar(cb)
    plt.show()

def main(points):
    xs, ys = np.meshgrid(points, points)
    f_xy = get_function(xs, ys)
    fig, axs = initialize_figure()
    plot_results(fig, axs, xs, ys, f_xy)

if __name__ == "__main__":
    points = np.arange(-5, 5.01, 0.01)
    main(points)