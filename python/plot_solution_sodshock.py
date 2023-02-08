import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from typing import List

from riemann_solver import RiemannSolver
from utils import read_particle_data


def plot_analytic_solution(axes: List[List[plt.Axes]], gas_gamma: float, rho_L: float, v_L: float, P_L: float,
                           rho_R: float, v_R: float, P_R: float, time: float = 0.25,
                           x_min: float = 0.5, x_max: float = 1.5, N: int = 1000):
    solver = RiemannSolver(gas_gamma)

    delta_x = (x_max - x_min) / N
    x_s = np.arange(x_min, x_max, delta_x) - 1
    rho_s, v_s, P_s, _ = solver.solve(rho_L, v_L, P_L, rho_R, v_R, P_R, x_s / time)
    x_s += 1

    # Additional arrays
    u_s = P_s / (rho_s * (gas_gamma - 1.0))  # internal energy
    s_s = P_s / rho_s ** gas_gamma  # entropic function

    axes[0][0].plot(x_s, v_s, ls="--", c="black", lw=1, zorder=-1)
    axes[0][1].plot(x_s, rho_s, ls="--", c="black", lw=1, zorder=-1)
    axes[0][2].plot(x_s, P_s, ls="--", c="black", lw=1, zorder=-1)
    axes[1][0].plot(x_s, u_s, ls="--", c="black", lw=1, zorder=-1)
    axes[1][1].plot(x_s, s_s, ls="--", c="black", lw=1, zorder=-1)


def plot_quantity(ax: plt.Axes, xdata: np.ndarray, ydata: np.ndarray, title: str):
    ax.scatter(xdata, ydata, s=4, color="red", zorder=1000, alpha=0.33)
    ax.set_title(title)
    ax.set_xlim(0.5, 1.5)
    mask = (xdata <= 1.5) & (xdata >= 0.5)
    ylim = [ydata[mask].min(), ydata[mask].max()]
    y_delta = ylim[1] - ylim[0]
    ax.set_ylim(ylim[0] - 0.1 * y_delta, ylim[1] + 0.1 * y_delta)


def main(fname: str, savename: str):
    # Parameters
    gas_gamma = 5.0 / 3.0  # Polytropic index
    rho_L = 1.0  # Density left state
    rho_R = 0.125  # Density right state
    v_L = 0.0  # Velocity left state
    v_R = 0.0  # Velocity right state
    P_L = 1.0  # Pressure left state
    P_R = 0.1  # Pressure right state

    # read data
    data = read_particle_data(fname)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    plot_analytic_solution(axes, gas_gamma, rho_L, v_L, P_L, rho_R, v_R, P_R)
    plot_quantity(axes[0][0], data["x"].values, data["v_x"].values, "Velocity")
    plot_quantity(axes[0][1], data["x"].values, data["rho"].values, "Density")
    plot_quantity(axes[0][2], data["x"].values, data["P"].values, "Pressure")
    plot_quantity(axes[1][0], data["x"].values, data["u"].values, "Internal energy")
    plot_quantity(axes[1][1], data["x"].values, data["S"].values, "Entropy")

    plt.tight_layout()
    plt.savefig(Path(fname).parent.parent / savename, dpi=600)
    plt.show()


if __name__ == "__main__":
    try:
        fname = sys.argv[1]
        savename = sys.argv[2]
    except IndexError:
        fname = "../run/output/sodshock_2D_0003.txt"
        savename = "test.png"
    main(fname, savename)
