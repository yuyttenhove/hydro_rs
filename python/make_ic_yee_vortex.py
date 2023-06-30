from typing import Tuple
import numpy as np
import math
from pathlib import Path
import h5py

from utils import write_file, get_root, yee_quantities

# Make initial conditions for a 2D yee vortex (see Pakmor 2016, appendix)

# Some global constants
GAMMA = 1.6666666667
T_INF = 1.
BETA = 5.

# Usefull functions
def concentric_seed_points(res: float, half_box_size: float) -> np.ndarray:
    box_diag = math.sqrt(2) * half_box_size
    points = []
    for n, r in enumerate(np.arange(0.5 * res, stop=box_diag, step=res)):
        circumference = 2 * math.pi * r
        n_points_on_ring = round(circumference / res)
        for i in range(n_points_on_ring):
            offset = 0.5 if n % 2 == 0 else 0.
            theta = (i + offset) / n_points_on_ring * 2 * math.pi
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            if abs(x) < half_box_size and abs(y) < half_box_size:
                points.append([x, y])
    return np.array(points)

def cartesian_seed_points(res: float, half_box_size: float) -> np.ndarray:
    points = []
    for x in np.arange(-half_box_size + 0.5 * res, half_box_size, res):
        for y in np.arange(-half_box_size + 0.5 * res, half_box_size, res):
            points.append([x, y])
    return np.array(points)


def gen_ic_from_coords(generators: np.ndarray, centroids: np.ndarray, volumes: np.ndarray, box_size: float, save_name: Path):
    """
    Compute the Yee vortex ICs for the given 2D 'centroids' (shape: '(N, 2)', centred around 0), 'volumes' (shape: '(N,)') and 'box_size'.
    """
    density, velocity, internal_energy = yee_quantities(centroids, T_INF, GAMMA, BETA)
    num_part = len(generators)
    box_size = np.array([box_size, box_size, 1])
    generators = np.concatenate([generators, np.zeros((num_part, 1))], axis=1)
    velocity = np.concatenate([velocity, np.zeros((num_part, 1))], axis=1)
    mass = volumes * density
    write_file(save_name, box_size, num_part, generators,
               mass, velocity, internal_energy, 2)


def gen_ic(box_size: float, n_1D: int, save_name: Path):
    res = box_size / n_1D
    centroids = concentric_seed_points(res, 0.5 * box_size)
    # centroids = cartesian_seed_points(res, 0.5 * box_size)
    coordinates = centroids + 0.5 * box_size
    volumes = res ** 2 * np.ones(len(centroids))
    gen_ic_from_coords(coordinates, centroids, volumes, box_size, save_name)


def gen_ic_from_file(fname: Path, save_name: Path):
    with h5py.File(fname, "r") as data:
        box_size = data["Header"].attrs["BoxSize"][0]
        coordinates = data["PartType0/Coordinates"][:][:, :2]
        centroids = data["PartType0/Centroids"][:][:, :2]
        volumes = data["PartType0/Volumes"][:]
    centroids -= 0.5 * box_size
    gen_ic_from_coords(coordinates, centroids, volumes, box_size, save_name)


def main():
    root = get_root()
    n = 400
    # gen_ic(10, n, root / f"run/ICs/yee_{n}.hdf5")
    gen_ic_from_file(root / f"run/output/yee_{n}_0000.hdf5", root / f"run/ICs/yee_{n}.hdf5")


if __name__ == "__main__":
    main()
