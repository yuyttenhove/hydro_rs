from matplotlib import pyplot as plt
from utils import plot_quantity, read_particle_data, get_root


if __name__ == "__main__":
    root = get_root()
    fname = root / "run" / "output" / "sodshock_2D_0005.txt"
    save_name = root / "run" / "constant_2D.png"
    data = read_particle_data(fname)
    x_lim = (0, 2)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    plot_quantity(axes[0][0], data["x"].values, data["v_x"].values, x_lim, "Velocity")
    plot_quantity(axes[0][1], data["x"].values, data["rho"].values, x_lim, "Density")
    plot_quantity(axes[0][2], data["x"].values, data["P"].values, x_lim, "Pressure")
    plot_quantity(axes[1][0], data["x"].values, data["u"].values, x_lim, "Internal energy")
    plot_quantity(axes[1][1], data["x"].values, data["S"].values, x_lim, "Entropy")
    fig.tight_layout()
    fig.savefig(save_name, dpi=300)
    fig.show()