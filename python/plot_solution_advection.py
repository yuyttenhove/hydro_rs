from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np

from make_ics_comparison import box, triangle, wave, transform
from utils import get_root


def read(fname) -> Tuple:
    with h5py.File(fname, "r") as data:
        x = data["PartType0/Coordinates"][:]
        rho = data["PartType0/Densities"][:]
    return x[:, 0], rho


def plot_comparison(ax: plt.Axes, godunov, waf, muscl, exact, x, x_exact, labels=False):
    if labels:
        ax.plot(x, godunov, label="Godunov")
        ax.plot(x, waf, label="WAF")
        ax.plot(x, muscl, label="MUSCL-Hancock")
        ax.plot(x_exact, exact, ls="--", lw=1, c="k", label="Analytic")
    else:
        ax.plot(x, godunov)
        ax.plot(x, waf)
        ax.plot(x, muscl)
        ax.plot(x_exact, exact, ls="--", lw=1, c="k")


if __name__ == "__main__":
    root = get_root()

    fig, axes = plt.subplots(3, 3, figsize=(6, 6), sharex=True, sharey="row", layout="constrained")

    num_part_exact = 1000
    x_exact = np.linspace(0, 1, num_part_exact, endpoint=False) + 1. / num_part_exact
    box_exact = transform(box(num_part_exact))
    triangle_exact = transform(triangle(num_part_exact))
    wave_exact = transform(wave(num_part_exact))

    times = [0, 2, 10]
    for i, t in enumerate(times):
        x, box_godunov = read(root / "run" / "output" / f"advection_box_godunov_optimal_{t:04}.hdf5")
        _, box_waf = read(root / "run" / "output" / f"advection_box_waf_vanleer_optimal_{t:04}.hdf5")
        _, box_muscl = read(root / "run" / "output" / f"advection_box_muscl_minbee_optimal_{t:04}.hdf5")
        plot_comparison(axes[i, 0], box_godunov, box_waf, box_muscl, box_exact, x, x_exact, labels=i == 0)
        axes[i][0].set_ylabel(f"$t={t}$")

        _, triangle_godunov = read(root / "run" / "output" / f"advection_triangle_godunov_optimal_{t:04}.hdf5")
        _, triangle_waf = read(root / "run" / "output" / f"advection_triangle_waf_tvd_optimal_{t:04}.hdf5")
        _, triangle_muscl = read(root / "run" / "output" / f"advection_triangle_muscl_tvd_optimal_{t:04}.hdf5")
        plot_comparison(axes[i, 1], triangle_godunov, triangle_waf, triangle_muscl, triangle_exact, x, x_exact)

        _, wave_godunov = read(root / "run" / "output" / f"advection_wave_godunov_optimal_{t:04}.hdf5")
        _, wave_waf = read(root / "run" / "output" / f"advection_wave_waf_tvd_optimal_{t:04}.hdf5")
        _, wave_muscl = read(root / "run" / "output" / f"advection_wave_muscl_tvd_optimal_{t:04}.hdf5")
        plot_comparison(axes[i, 2], wave_godunov, wave_waf, wave_muscl, wave_exact, x, x_exact)

    fig.legend(loc="outside lower center", bbox_transform=fig.transFigure, ncol=2)
    # plt.tight_layout()
    plt.show()
