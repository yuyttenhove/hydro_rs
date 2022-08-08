import matplotlib.pyplot as plt
from numpy import block
import pandas as pd
import sys
from pathlib import Path


def plot_quantity(ax, xdata, ydata, title):
    ax.scatter(xdata, ydata, alpha=0.75)
    ax.set_title(title)
    ax.set_xlim([0.5, 1.5])
    mask = (xdata <= 1.5) & (xdata >= 0.5)
    ylim = [ydata[mask].min(), ydata[mask].max()]
    y_delta = ylim[1] - ylim[0]
    ax.set_ylim([ylim[0] - 0.1 * y_delta, ylim[1] + 0.1 * y_delta])


def main(fname: str):
    data = pd.read_csv(fname, sep="\t")
    data.columns = ["x", "rho", "v", "P", "a"]

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    plot_quantity(axes[0][0], data["x"], data["rho"], "Density")
    plot_quantity(axes[0][1], data["x"], data["v"], "Velocity")
    plot_quantity(axes[1][0], data["x"], data["P"], "Pressure")
    plot_quantity(axes[1][1], data["x"], data["a"], "Acceleration")

    plt.savefig("test.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    try:
        fname = sys.argv[1]
    except IndexError:
        pass
    else:
        main(fname)
