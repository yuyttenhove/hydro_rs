from math import ceil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import h5py

from typing import Tuple


def get_start_count(fname, header, name):
    start = 0
    count = 0
    with open(fname) as f:
        i = 0
        while (l := f.readline().rstrip()) is not None:
            i += 1
            if l == header:
                start = i
                break
        if start == -1:
            raise ValueError(f"Found no {name} data!")
        while (l := f.readline().rstrip()) is not None:
            if len(l) == 0:
                break
            if l[:2] == "##":
                break
            if l[0] != "#":
                count += 1
    return start, count


def read_particle_data(fname: str) -> pd.DataFrame:
    start, count = get_start_count(fname, "## Particles:", "particle")
    data = pd.read_csv(fname, sep="\t", skiprows=start, nrows=count)
    data = data.iloc[:, 1:]
    data.columns = ["x", "y", "z", "rho", "v_x", "v_y", "v_z", "P", "u", "S", "t"]
    return data


def read_face_data(fname: str) -> pd.DataFrame:
    start, count = get_start_count(fname, "## Voronoi faces:", "voronoi face")
    data = pd.read_csv(fname, sep="\t", skiprows=start, nrows=count)
    data = data.iloc[:, 1:]
    data.columns = ["a_x", "a_y", "b_x", "b_y"]
    return data


def expand_lim(lim):
    diff = lim[1] - lim[0]
    return lim[0] - 0.1 * diff, lim[1] + 0.1 * diff


def plot_faces(fname: str, lw=0.5, dpi=300, show_ax=True):
    faces = read_face_data(fname)
    faces = [np.stack([row[:2], row[2:]]) for row in faces.values]
    lines = LineCollection(faces, color="r", lw=lw)
    xlim = (min([f[:, 0].min() for f in faces]), max([f[:, 0].max() for f in faces]))
    ylim = (min([f[:, 1].min() for f in faces]), max([f[:, 1].max() for f in faces]))
    x_diff = xlim[1] - xlim[0]
    y_diff = ylim[1] - ylim[0]
    width = 9
    fig, ax = plt.subplots(figsize=(width, ceil(width * y_diff / x_diff)))
    ax.add_collection(lines)
    ax.set_aspect("equal")
    ax.set_xlim(*expand_lim(xlim))
    ax.set_ylim(*expand_lim(ylim))
    if not show_ax:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("voronoi.png", dpi=dpi)

def plot_quantity(ax: plt.Axes, xdata: np.ndarray, ydata: np.ndarray, xlim: Tuple[float, float], title: str, logx=False, logy=False):
    windowsize = 9
    y_median = np.array([np.median(ydata[i: i + windowsize]) for i in range(len(ydata) - windowsize + 1)])
    y_delta = ydata[windowsize // 2: -windowsize // 2 + 1] - y_median
    y_delta = np.concatenate([[y_delta[0], y_delta[0]], y_delta, [y_delta[-1], y_delta[-1]]])
    y_delta = 3 * np.array([np.sqrt(np.sum(y_delta[i: i + windowsize] ** 2) / (windowsize * (windowsize - 1))) for i in
                            range(len(ydata) - windowsize + 1)])
    y_min = y_median - y_delta
    y_max = y_median + y_delta
    x_median = xdata[windowsize // 2: -windowsize // 2 + 1]
    # ax.fill_between(x_median, y_min, y_max, alpha=0.25)
    # ax.plot(x_median, y_median)
    ax.scatter(xdata, ydata, s=4, color="red", zorder=1000, alpha=0.33)
    ax.set_title(title)
    if not logx:
        ax.set_xlim(*xlim)
        if not logy:
            mask = (xdata <= xlim[1]) & (xdata >= xlim[0])
            ylim = [ydata[mask].min(), ydata[mask].max()]
            y_delta = ylim[1] - ylim[0]
            ax.set_ylim(ylim[0] - 0.1 * y_delta, ylim[1] + 0.1 * y_delta)
        else:
            ax.semilogy()
    else:
        if logy:
            ax.loglog()
        else:
            ax.semilogx()

def write_file(fname, box_size, num_part, coords, m, v, u, dimension):
    with h5py.File(fname, 'w') as file:
        # Header
        grp = file.create_group("/Header")
        grp.attrs["BoxSize"] = box_size
        grp.attrs["NumPart_Total"] = [num_part, 0, 0, 0, 0, 0]
        grp.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
        grp.attrs["NumPart_ThisFile"] = [num_part, 0, 0, 0, 0, 0]
        grp.attrs["Time"] = 0.0
        grp.attrs["NumFilesPerSnapshot"] = 1
        grp.attrs["MassTable"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        grp.attrs["Flag_Entropy_ICs"] = [0, 0, 0, 0, 0, 0]
        grp.attrs["NumPart_Total"] = num_part
        grp.attrs["Dimension"] = dimension

        # Units
        grp = file.create_group("/Units")
        grp.attrs["Unit length in cgs (U_L)"] = 1.
        grp.attrs["Unit mass in cgs (U_M)"] = 1.
        grp.attrs["Unit time in cgs (U_t)"] = 1.
        grp.attrs["Unit current in cgs (U_I)"] = 1.
        grp.attrs["Unit temperature in cgs (U_T)"] = 1.

        # Particle group
        grp = file.create_group("/PartType0")
        grp.create_dataset('Coordinates', data=coords, dtype='d')
        grp.create_dataset('Velocities', data=v, dtype='f')
        grp.create_dataset('Masses', data=m, dtype='f')
        grp.create_dataset('InternalEnergy', data=u, dtype='f')
        ids = np.arange(len(m)) + 1
        grp.create_dataset('ParticleIDs', data=ids, dtype='L')


def get_plane(width, height, nx, ny, pert = 0):
    coords = np.zeros((nx * ny, 2))
    for i in range(nx):
        for j in range(ny):
            coords[i * ny + j, 0] = (i + 0.5) / nx * width
            coords[i * ny + j, 1] = (j + 0.5) / ny * height
    coords += pert * np.array([width / nx, height / ny]) * np.random.random((nx * ny, 2))
    return np.concatenate([coords, np.zeros((len(coords), 1))], axis=1)

def internal_energy_ideal_gas(P, rho, gamma):
    return P / ((gamma - 1.) * rho)


def get_root():
    return Path(__file__).parent.parent


if __name__ == "__main__":
   plot_faces(Path(__file__).parent / "../run/output/sodshock_2D_0000.txt")
