import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from utils import plot_analytic_solution, plot_quantity, read_particle_data

def main(fname: str, savename: str, time=0.5):
    # Parameters
    gas_gamma = 5.0 / 3.0  # Polytropic index
    rho_L = 1.0  # Density left state
    rho_R = 1.0  # Density right state
    v_L = 1.0  # Velocity left state
    v_R = -1.0  # Velocity right state
    P_L = 1e-6  # Pressure left state
    P_R = 1e-6  # Pressure right state

    # read data
    data = read_particle_data(fname)

    # Plot
    x_lim = [0., 2.]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    plot_analytic_solution(axes, gas_gamma, rho_L, v_L, P_L, rho_R, v_R, P_R, time=time, x_min=x_lim[0], x_max=x_lim[1])
    plot_quantity(axes[0][0], data["x"].values, data["v_x"].values, x_lim, "Velocity")
    plot_quantity(axes[0][1], data["x"].values, data["rho"].values, x_lim, "Density")
    plot_quantity(axes[0][2], data["x"].values, data["P"].values, x_lim, "Pressure")
    plot_quantity(axes[1][0], data["x"].values, data["u"].values, x_lim, "Internal energy")
    plot_quantity(axes[1][1], data["x"].values, data["S"].values, x_lim, "Entropy")

    plt.tight_layout()
    plt.savefig(Path(fname).parent.parent / savename, dpi=600)
    plt.show()


if __name__ == "__main__":
    try:
        fname = sys.argv[1]
        savename = sys.argv[2]
    except IndexError:
        fname = "../run/output/noh_1D_0010.txt"
        savename = "test.png"
    main(fname, savename)
