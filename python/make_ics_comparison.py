import sys

import numpy as np

from utils import write_file, get_root


def box(num_part: int) -> np.ndarray:
    x = np.linspace(0., 1., num_part, endpoint=False) + 0.5 / num_part
    return np.where((x <= 0.25) | (x >= 0.75), 0., 1.)


def triangle(num_part: int) -> np.ndarray:
    x = np.linspace(0., 1., num_part, endpoint=False) + 0.5 / num_part
    y = 1. - 4. * np.abs(0.5 - x)
    return np.maximum(0., y)


def wave(num_part: int) -> np.ndarray:
    x = np.linspace(0., 1., num_part, endpoint=False) + 0.5 / num_part
    f = lambda r: 2 * r ** 3 - 3 * r ** 2 + 1
    return np.where((x <= 0.25) | (x >= 0.75), 0., f(4. * np.abs(x - 0.5)))


def transform(vals: np.ndarray, min=1., max=2.):
    return vals * (max - min) + min


if __name__ == "__main__":
    root = get_root()
    try:
        num_part = sys.argv[1]
    except IndexError:
        num_part = 100
    boxsize = 1.
    dimension = 1

    x = np.linspace(0., 1., num_part, endpoint=False) + 0.5 / num_part
    x *= boxsize
    coords = np.zeros((x.size, 3))
    coords[:, 0] = x
    v = np.zeros_like(coords)
    u = np.zeros_like(x)
    h = np.zeros_like(x)
    rho = 1.
    m = rho * boxsize / num_part

    write_file(root / "run" / "ICs" / f"advection_box_{num_part}.hdf5", boxsize, num_part, dimension, coords,
               m * transform(box(num_part)), v, u, h)
    write_file(root / "run" / "ICs" / f"advection_triangle_{num_part}.hdf5", boxsize, num_part, dimension, coords,
               m * transform(triangle(num_part)), v, u, h)
    write_file(root / "run" / "ICs" / f"advection_wave_{num_part}.hdf5", boxsize, num_part, dimension, coords,
               m * transform(wave(num_part)), v, u, h)
