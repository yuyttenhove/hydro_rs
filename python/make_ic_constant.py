from pathlib import Path
from utils import internal_energy_ideal_gas, write_file, get_plane
import numpy as np


if __name__ == "__main__":
    nx = 5
    ny = 5
    box_size = np.array([1, 1, 1])
    volume = box_size.prod()
    num_part = nx * ny
    dimension = 2
    rho = 1
    P = 1
    velocity = 0.5
    gamma = 5 / 3

    coords = get_plane(box_size[0], box_size[1], nx, ny)
    m = volume / num_part * rho * np.ones(num_part)
    v = np.zeros_like(coords)
    v[:, 0] = velocity
    u = internal_energy_ideal_gas(P, rho, gamma) * np.ones_like(m)

    fname = Path(__file__).parent.parent / "run/ICs/constant_2D.hdf5"
    write_file(fname, box_size, num_part, coords, m, v, u, dimension)
