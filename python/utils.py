import math
from math import ceil
from pathlib import Path
from typing import List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.interpolate import griddata

from riemann_solver import RiemannSolver


def read_particle_data(fname: str) -> Tuple[pd.DataFrame, float]:
    with h5py.File(fname, "r") as data:
        time = data["Header"].attrs["Time"][0]
        coordinates = data["PartType0/Coordinates"][:]
        try:
            centroids = data["PartType0/Centroids"][:]
        except KeyError:
            print("Warning, no centroids found, substituting with coordinates!")
            centroids = np.array(coordinates)
        velocities = data["PartType0/Velocities"][:]
        densities = data["PartType0/Densities"][:]
        pressures = data["PartType0/Pressures"][:]
        try:
            internal_energy = data["PartType0/InternalEnergy"][:]
        except KeyError:
            try:
                internal_energy = data["PartType0/InternalEnergies"][:]
            except KeyError:
                print("Warning! No internal energies found! Setting them to zero...")
                internal_energy = np.zeros_like(pressures)
        try:
            entropy = data["PartType0/Entropy"][:]
        except KeyError:
            try:
                entropy = data["PartType0/Entropies"][:]
            except KeyError:
                print("Warning! No internal energies found! Setting them to zero...")
                entropy = np.zeros_like(pressures)
        try:
            timestep = data["PartType0/Timestep"][:]
        except KeyError:
            timestep = np.zeros_like(pressures)

    data = pd.DataFrame(
        data=np.concatenate(
            [
                coordinates,
                centroids,
                densities.reshape((-1, 1)),
                velocities,
                pressures.reshape((-1, 1)),
                internal_energy.reshape((-1, 1)),
                entropy.reshape((-1, 1)),
                timestep.reshape((-1, 1)),
            ],
            axis=1
        ),
        columns=["x", "y", "z", "c_x", "c_y", "c_z", "rho",
                 "v_x", "v_y", "v_z", "P", "u", "S", "t"]
    )
    return data, time


def read_face_geometry(fname: str) -> pd.DataFrame:
    with h5py.File(fname, "r") as data:
        dimension = data["Header"].attrs["Dimension"]
        if dimension != 2:
            raise ValueError(
                f"Cannot read face geometry with dimension = {dimension}!")
        if "VoronoiFaces" not in data:
            raise ValueError("No Voronoi data found in file!")
        normals = data["VoronoiFaces/Normal"][:]
        areas = data["VoronoiFaces/Area"][:].reshape((-1, 1))
        centroids = data["VoronoiFaces/Centroid"][:]

    face_dir = np.stack([normals[:, 1], -normals[:, 0],
                        np.zeros(len(normals))], axis=1)
    a = centroids - 0.5 * areas * face_dir
    b = centroids + 0.5 * areas * face_dir

    data = pd.DataFrame(data=np.concatenate(
        [a[:, :2], b[:, :2]], axis=1), columns=["a_x", "a_y", "b_x", "b_y"])
    return data


def expand_lim(lim):
    diff = lim[1] - lim[0]
    return lim[0] - 0.1 * diff, lim[1] + 0.1 * diff


def plot_faces(fname: str, lw=0.5, dpi=300, show_ax=True):
    faces = read_face_geometry(fname)
    faces = [np.stack([row[:2], row[2:]]) for row in faces.values]
    lines = LineCollection(faces, color="r", lw=lw)
    xlim = (min([f[:, 0].min() for f in faces]),
            max([f[:, 0].max() for f in faces]))
    ylim = (min([f[:, 1].min() for f in faces]),
            max([f[:, 1].max() for f in faces]))
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
    fig.savefig(Path(fname).parent.parent / "voronoi.png", dpi=dpi)


def plot_analytic_solution(axes: List[List[plt.Axes]], gas_gamma: float, rho_L: float, v_L: float, P_L: float,
                           rho_R: float, v_R: float, P_R: float, time: float = 0.25,
                           x_min: float = 0.5, x_max: float = 1.5, N: int = 1000):
    solver = RiemannSolver(gas_gamma)

    delta_x = (x_max - x_min) / N
    x_s = np.arange(x_min, x_max, delta_x) - 1
    rho_s, v_s, P_s, _ = solver.solve(
        rho_L, v_L, P_L, rho_R, v_R, P_R, x_s / time)
    x_s += 1

    # Additional arrays
    u_s = P_s / (rho_s * (gas_gamma - 1.0))  # internal energy
    s_s = P_s / rho_s ** gas_gamma  # entropic function

    axes[0][0].plot(x_s, v_s, ls="--", c="black", lw=1, zorder=-1)
    axes[0][1].plot(x_s, rho_s, ls="--", c="black", lw=1, zorder=-1)
    axes[0][2].plot(x_s, P_s, ls="--", c="black", lw=1, zorder=-1)
    axes[1][0].plot(x_s, u_s, ls="--", c="black", lw=1, zorder=-1)
    axes[1][1].plot(x_s, s_s, ls="--", c="black", lw=1, zorder=-1)


def plot_quantity(ax: plt.Axes, xdata: np.ndarray, ydata: np.ndarray, xlim: Tuple[float, float], title: str, logx=False, logy=False):
    windowsize = 9
    y_median = np.array([np.median(ydata[i: i + windowsize])
                        for i in range(len(ydata) - windowsize + 1)])
    y_delta = ydata[windowsize // 2: -windowsize // 2 + 1] - y_median
    y_delta = np.concatenate(
        [[y_delta[0], y_delta[0]], y_delta, [y_delta[-1], y_delta[-1]]])
    y_delta = 3 * np.array([np.sqrt(np.sum(y_delta[i: i + windowsize] ** 2) / (windowsize * (windowsize - 1))) for i in
                            range(len(ydata) - windowsize + 1)])
    # y_min = y_median - y_delta
    # y_max = y_median + y_delta
    # x_median = xdata[windowsize // 2: -windowsize // 2 + 1]
    # ax.fill_between(x_median, y_min, y_max, alpha=0.25)
    # ax.plot(x_median, y_median)
    ax.scatter(xdata, ydata, s=4, color="red", zorder=1000, alpha=0.33)
    # ax.plot(xdata, ydata, color="red", zorder=1000)
    ax.set_title(title)
    if not logx:
        ax.set_xlim(*xlim)
        if not logy:
            mask = (xdata <= xlim[1]) & (xdata >= xlim[0]) & np.isfinite(ydata)
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


def write_file(fname, box_size, num_part, dimension, coords, m, v, u, smoothing_length=None):
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
        if smoothing_length is not None:
            grp.create_dataset('SmoothingLength', data=smoothing_length, dtype='d')


def get_plane(width, height, nx, ny, pert=0):
    coords = np.zeros((nx * ny, 2))
    for i in range(nx):
        for j in range(ny):
            coords[i * ny + j, 0] = (i + 0.5) / nx * width
            coords[i * ny + j, 1] = (j + 0.5) / ny * height
    coords += pert * np.array([width / nx, height / ny]) * \
        np.random.random((nx * ny, 2))
    return np.concatenate([coords, np.zeros((len(coords), 1))], axis=1)


def internal_energy_ideal_gas(P, rho, gamma):
    return P / ((gamma - 1.) * rho)


def get_root():
    return Path(__file__).parent.parent


def interpolate_nn(data, coordinates_from, coordinates_to):
    return griddata(coordinates_from, data, coordinates_to, method="nearest")


def get_slice(data, coordinates, x_lim, y_lim, res=1000, z=0):
    x = np.linspace(x_lim[0], x_lim[1], res) + \
        0.5 * (x_lim[1] - x_lim[0]) / res
    y_res = math.floor((y_lim[1] - y_lim[0]) / (x_lim[1] - x_lim[0]) * res)
    y = np.linspace(y_lim[0], y_lim[1], y_res) + \
        0.5 * (y_lim[1] - y_lim[0]) / y_res
    z = np.array([z])
    x, y, z = np.meshgrid(x, y, z)
    coords_to = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return interpolate_nn(data, coordinates, coords_to).reshape((y_res, res))[::-1, :]


def yee_quantities(coordinates: np.ndarray, T_inf, gamma, beta) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
     See appendix A from Hopkins 2016 (Has wrong sign in exponent of density!!)
     Coordinates are assumed to be centred around zero
    """
    radii_2 = np.sum(coordinates * coordinates, axis=1)
    temperature = T_inf - (gamma - 1) * beta**2 / \
        (8 * gamma * math.pi**2) * np.exp(1 - radii_2)
    density = np.power(temperature, 1 / (gamma - 1))
    v_fac = 0.5 * beta / math.pi * np.exp(0.5 * (1 - radii_2))
    velocity = np.stack([-coordinates[:, 1] * v_fac,
                        coordinates[:, 0] * v_fac], axis=1)
    internal_energy = temperature / (gamma - 1)

    return density, velocity, internal_energy
