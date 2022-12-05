import matplotlib.pyplot as plt
import sys
from pathlib import Path

from utils import read_data, plot_quantity


def main(fname: str, savename: str):
    data = read_data(fname)

    fig, axes = plt.subplots(1, 3, figsize=(9, 6))
    xlim = (0., 1.)
    plot_quantity(axes[0], data["x"].values, data["v"].values, xlim, "Velocity", logx=False)
    plot_quantity(axes[1], data["x"].values, data["rho"].values, xlim, "Density", logx=False, logy=False)
    plot_quantity(axes[2], data["x"].values, data["P"].values, xlim, "Pressure", logx=False, logy=False)

    plt.tight_layout()
    plt.savefig(Path(fname).parent.parent / savename, dpi=600)
    plt.show()


if __name__ == "__main__":
    try:
        fname = sys.argv[1]
        savename = sys.argv[2]
    except IndexError:
        fname = "../run/output/constant_0010.txt"
        savename = "test.png"
    main(fname, savename)
