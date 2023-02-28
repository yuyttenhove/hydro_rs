import matplotlib.pyplot as plt
import numpy as np

from utils import get_slice, read_particle_data


def plot_slice(fname, savename):
    data = read_particle_data(fname)
    coords = np.column_stack([data["x"].values, data["y"].values, data["z"].values])
    values = data["rho"]
    x_lim = [0., 1.]
    y_lim = [0., 1.5]
    res = 1500

    interpolated = get_slice(values, coords, x_lim, y_lim, res)
    plt.imshow(interpolated, cmap="viridis")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(savename, dpi=300)


if __name__ == "__main__":
    from utils import get_root
    n = 150
    plot_slice(get_root() / "run" / "output" / f"rayleigh_taylor_2D_{n:04}.txt", get_root() / "run/slice_rt.png")
    
