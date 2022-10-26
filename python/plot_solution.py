import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path


def plot_quantity(ax: plt.Axes, xdata: np.ndarray, ydata: np.ndarray, title: str):
    ax.scatter(xdata, ydata, s=3, alpha=0.75)
    ax.set_title(title)
    ax.set_xlim([0.5, 1.5])
    mask = (xdata <= 1.5) & (xdata >= 0.5)
    ylim = [ydata[mask].min(), ydata[mask].max()]
    y_delta = ylim[1] - ylim[0]
    ax.set_ylim([ylim[0] - 0.1 * y_delta, ylim[1] + 0.1 * y_delta])


def main(fname: str):
    data = pd.read_csv(fname, sep="\t")
    data.columns = ["x", "rho", "v", "P", "a", "u", "S"]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    plot_quantity(axes[0][0], data["x"], data["v"], "Velocity")
    plot_quantity(axes[0][1], data["x"], data["rho"], "Density")
    plot_quantity(axes[0][2], data["x"], data["P"], "Pressure")
    plot_quantity(axes[1][0], data["x"], data["u"], "Internal energy")
    plot_quantity(axes[1][1], data["x"], data["S"], "Entropy")
    plot_quantity(axes[1][2], data["x"], data["a"], "Acceleration")

    plt.savefig(Path(fname).parent.parent / "test.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    try:
        fname = sys.argv[1]
    except IndexError:
        fname = "../run/output/sod_shock_0007.txt"
    main(fname)
