import matplotlib.pyplot as plt
import numpy as np

from utils import get_slice, read_particle_data


def plot_slice(key, fname, savename):
    data, _ = read_particle_data(fname)
    coords = np.column_stack([data["x"].values, data["y"].values, data["z"].values])
    values = data[key]
    x_lim = [0., 1.]
    y_lim = [0., 1.]
    res = 1500

    interpolated = get_slice(values, coords, x_lim, y_lim, res)
    im = plt.imshow(interpolated, cmap="viridis")
    plt.colorbar(im)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(savename, dpi=300)


if __name__ == "__main__":
    from utils import get_root
    import sys

    try:
        key = sys.argv[1]
    except IndexError:
        key = "rho"

    try:
        fname = sys.argv[2]
    except IndexError:
        fname = get_root() / "run" / "output" / f"KelvinHelmholtz_HLLC_boosted_2D_0006.txt"

    try:
        savename = sys.argv[3]
    except IndexError:
        savename = get_root() / "run" / "output" / f"slice.png"
    plot_slice(key, fname, savename)
    
