import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from typing import List

from riemann_solver import RiemannSolver


def plot_analytic_solution(axes: List[List[plt.Axes]], gas_gamma: float, rho_L: float, v_L: float, P_L: float,
                           rho_R: float, v_R: float, P_R: float, time: float = 0.25,
                           x_min: float = 0., x_max: float = 2., N: int = 1000):
    solver = RiemannSolver(gas_gamma)

    delta_x = (x_max - x_min) / N
    x_s = 0.5 * (np.arange(x_min, x_max, delta_x) - 1)
    x_s2 = np.array(x_s + 1)
    rho_s, v_s, P_s, _ = solver.solve(rho_L, v_L, P_L, rho_R, v_R, P_R, x_s / time)
    rho_s2, v_s2, P_s2, _ = solver.solve(rho_R, v_R, P_R, rho_L, v_L, P_L, x_s / time)
    x_s += 1
    x_s2 += 1.0
    s2neg = x_s2 > 2.0
    s2pos = ~s2neg
    x_s2[s2neg] -= 2.0

    # Additional arrays
    u_s = P_s / (rho_s * (gas_gamma - 1.0))  # internal energy
    s_s = P_s / rho_s ** gas_gamma  # entropic function
    u_s2 = P_s2 / (rho_s2 * (gas_gamma - 1.0))  # internal energy
    s_s2 = P_s2 / rho_s2 ** gas_gamma  # entropic function

    axes[0][0].plot(x_s, v_s, ls="--", c="black", lw=1, zorder=-1)
    axes[0][0].plot(x_s2[s2pos], v_s2[s2pos], ls="--", c="black", lw=1, zorder=-1)
    axes[0][0].plot(x_s2[s2neg], v_s2[s2neg], ls="--", c="black", lw=1, zorder=-1)
    axes[0][1].plot(x_s, rho_s, ls="--", c="black", lw=1, zorder=-1)
    axes[0][1].plot(x_s2[s2pos], rho_s2[s2pos], ls="--", c="black", lw=1, zorder=-1)
    axes[0][1].plot(x_s2[s2neg], rho_s2[s2neg], ls="--", c="black", lw=1, zorder=-1)
    axes[0][2].plot(x_s, P_s, ls="--", c="black", lw=1, zorder=-1)
    axes[0][2].plot(x_s2[s2pos], P_s2[s2pos], ls="--", c="black", lw=1, zorder=-1)
    axes[0][2].plot(x_s2[s2neg], P_s2[s2neg], ls="--", c="black", lw=1, zorder=-1)
    axes[1][0].plot(x_s, u_s, ls="--", c="black", lw=1, zorder=-1)
    axes[1][0].plot(x_s2[s2pos], u_s2[s2pos], ls="--", c="black", lw=1, zorder=-1)
    axes[1][0].plot(x_s2[s2neg], u_s2[s2neg], ls="--", c="black", lw=1, zorder=-1)
    axes[1][1].plot(x_s, s_s, ls="--", c="black", lw=1, zorder=-1)
    axes[1][1].plot(x_s2[s2pos], s_s2[s2pos], ls="--", c="black", lw=1, zorder=-1)
    axes[1][1].plot(x_s2[s2neg], s_s2[s2neg], ls="--", c="black", lw=1, zorder=-1)


def plot_quantity(ax: plt.Axes, xdata: np.ndarray, ydata: np.ndarray, title: str):
    windowsize = 9
    y_median = np.array([np.median(ydata[i: i + windowsize]) for i in range(len(ydata) - windowsize + 1)])
    y_delta = ydata[windowsize // 2: -windowsize // 2 + 1] - y_median
    y_delta = np.concatenate([[y_delta[0], y_delta[0]], y_delta, [y_delta[-1], y_delta[-1]]])
    y_delta = 3 * np.array([np.sqrt(np.sum(y_delta[i: i + windowsize] ** 2) / (windowsize * (windowsize - 1))) for i in
                            range(len(ydata) - windowsize + 1)])
    y_min = y_median - y_delta
    y_max = y_median + y_delta
    x_median = xdata[windowsize // 2: -windowsize // 2 + 1]
    ax.fill_between(x_median, y_min, y_max, alpha=0.25)
    ax.plot(x_median, y_median)
    ax.scatter(xdata, ydata, s=4, color="red", zorder=1000, alpha=0.33)
    ax.set_title(title)
    xlim = [0., 2.]
    ax.set_xlim(*xlim)
    mask = (xdata <= xlim[1]) & (xdata >= xlim[0])
    ylim = [ydata[mask].min(), ydata[mask].max()]
    y_delta = ylim[1] - ylim[0]
    ax.set_ylim(ylim[0] - 0.1 * y_delta, ylim[1] + 0.1 * y_delta)


def main(fname: str, savename: str):
    # Parameters
    gas_gamma = 5.0 / 3.0  # Polytropic index
    rho_L = 1.0  # Density left state
    rho_R = 1.0  # Density right state
    v_L = 2.0  # Velocity left state
    v_R = -2.0  # Velocity right state
    P_L = 0.4  # Pressure left state
    P_R = 0.4  # Pressure right state

    # read data
    data = pd.read_csv(fname, sep="\t")
    data.columns = ["x", "rho", "v", "P", "a", "u", "S"]

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    plot_analytic_solution(axes, gas_gamma, rho_L, v_L, P_L, rho_R, v_R, P_R)
    plot_quantity(axes[0][0], data["x"].values, data["v"].values, "Velocity")
    plot_quantity(axes[0][1], data["x"].values, data["rho"].values, "Density")
    plot_quantity(axes[0][2], data["x"].values, data["P"].values, "Pressure")
    plot_quantity(axes[1][0], data["x"].values, data["u"].values, "Internal energy")
    plot_quantity(axes[1][1], data["x"].values, data["S"].values, "Entropy")
    plot_quantity(axes[1][2], data["x"].values, data["a"].values, "Acceleration")

    plt.tight_layout()
    plt.savefig(Path(fname).parent.parent / savename, dpi=600)
    plt.show()


if __name__ == "__main__":
    try:
        fname = sys.argv[1]
        savename = sys.argv[2]
    except IndexError:
        fname = "../run/output/toro_0005.txt"
        savename = "test.png"
    main(fname, savename)
