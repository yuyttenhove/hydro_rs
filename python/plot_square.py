import matplotlib.pyplot as plt
from pathlib import Path
from utils import read_data, plot_quantity


def main(fname, savename):
    data = read_data(fname)
    fig, axes = plt.subplots(1, 2, figsize=(9, 6))
    xlim = (0., 1.)
    plot_quantity(axes[0], data["x"].values, data["rho"].values, xlim, "Density", logx=False)
    plot_quantity(axes[1], data["x"].values, data["t"].values, xlim, "Timestep", logx=False, logy=False)

    plt.tight_layout()
    plt.savefig(Path(fname).parent.parent / savename, dpi=600)
    plt.show()


if __name__ == "__main__":
    for i in range(11):
        main(f"../run/output/square_{i:04}.txt", f"square_{i:04}.png")
