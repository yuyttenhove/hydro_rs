import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple


def read_data(fname: str) -> pd.DataFrame:
    data = pd.read_csv(fname, sep="\t")
    data.columns = ["x", "rho", "v", "P", "a", "u", "S", "t"]
    return data


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
