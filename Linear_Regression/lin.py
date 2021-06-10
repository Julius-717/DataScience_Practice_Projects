import numpy as np
import matplotlib.pyplot as plt

def get_data(a, b, x):
    
    y = a * x + np.random.randn(len(x)) + b
    return y

def get_fit(x, y):

    p, var = np.polyfit(x, y, 1, cov = True)
    return p, var # polynomial coeff and covariance

def get_poly_eval(x, p):

    y_=np.polyval(p, x)
    return y_

def initialize_fig(p):

    plt.style.use("grayscale")
    fig, ax = plt.subplots()
    fig.suptitle("Lin. Regr. and Max. Likeli. Est.")
    ax.set_title(f"Es. params: a={p[0]}, b={p[1]}.")
    ax.set_xlabel("$x,$[-]")
    ax.set_ylabel("$y,$[-]")
    ax.tick_params(axis="both",which="major")
    ax.grid(True)
    return fig, ax

def plot_figure(fig, ax, x, y, y_):

    ax.scatter(x, y, c="r")
    ax.plot(x, y_)
    plt.show()

def main(a, b, x):

    y = get_data(a, b, x)
    p, var = get_fit(x, y)
    y_ = get_poly_eval(x, p)
    fig, ax = initialize_fig(p)
    plot_figure(fig, ax, x, y, y_)

if __name__=="__main__":
    a = 6
    b = 1
    x = np.linspace(0, 1, 1000)
    main(a, b, x)