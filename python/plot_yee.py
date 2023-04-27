import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

from utils import read_particle_data, get_root, yee_quantities, get_slice

GAMMA = 1.666666667
T_INF = 1.
BETA = 5.


def load(data, center=5):
    coordinates = data.loc[:, ["x", "y", "z"]].values
    velocities = data.loc[:, ["v_x", "v_y", "v_z"]].values
    centroids = data.loc[:, ["c_x", "c_y", "c_z"]].values
    centred_centroids = centroids
    centred_centroids[:, :2] -= center
    density = data["rho"].values
    internal_energy = data["u"].values
    density_s, velocities_s, internal_energy_s = yee_quantities(
        centred_centroids, T_INF, GAMMA, BETA)

    delta_rho = np.abs(density - density_s)
    delta_v = velocities[:, :2] - velocities_s
    delta_v = np.sqrt(np.sum(delta_v * delta_v, axis=1))
    delta_u = np.abs(internal_energy_s - internal_energy)

    return coordinates, density, delta_rho, delta_v, delta_u


def load_all(entries: dict, n: int, center: float = 5.) -> dict:
    root = get_root()

    general_info = dict(
        limits={
            "Field": [np.inf, -np.inf],
            "Density": [np.inf, -np.inf],
            "Velocity": [np.inf, -np.inf],
            "Internal energy": [np.inf, -np.inf],
        },
        n=n,
    )
    for basename, this_entries in entries.items():
        data, time = read_particle_data(
            root / f"run/output/{basename}_{n:04}.hdf5")
        coordinates, field, rho, velocity, internal_energy = load(data, center=center)
        this_entries["Coordinates"] = coordinates
        this_entries["Field"] = field
        this_entries["Density"] = rho
        this_entries["Velocity"] = velocity
        this_entries["Internal energy"] = internal_energy
        general_info["limits"]["Field"][0] = min(
            general_info["limits"]["Field"][0], field.min())
        general_info["limits"]["Field"][1] = max(
            general_info["limits"]["Field"][1], field.max())
        general_info["limits"]["Density"][0] = min(
            general_info["limits"]["Density"][0], rho.min())
        general_info["limits"]["Density"][1] = max(
            general_info["limits"]["Density"][1], rho.max())
        general_info["limits"]["Velocity"][0] = min(
            general_info["limits"]["Velocity"][0], velocity.min())
        general_info["limits"]["Velocity"][1] = max(
            general_info["limits"]["Velocity"][1], velocity.max())
        general_info["limits"]["Internal energy"][0] = min(
            general_info["limits"]["Internal energy"][0], internal_energy.min())
        general_info["limits"]["Internal energy"][1] = max(
            general_info["limits"]["Internal energy"][1], internal_energy.max())

    general_info["time"] = time

    return general_info


def plot_single(coordinates, property, ax, norm, title=None, cmap=None, lim=[0, 10], res=300):
    sliced = get_slice(property, coordinates, x_lim=lim, y_lim=lim, res=res)

    im = ax.imshow(sliced, norm=norm, cmap=cmap)
    ax.cax.colorbar(im)
    if title is not None:
        ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])


def plot_comparison(entries, info, savename_base=None, half_drift=None):

    norm_field = Normalize(vmin=info["limits"]["Field"][0],
                         vmax=info["limits"]["Field"][1])    
    norm_rho = Normalize(vmin=info["limits"]["Density"][0],
                         vmax=info["limits"]["Density"][1])
    # norm_rho = LogNorm(vmin=1e-3,
    #                      vmax=info["limits"]["Density"][1])
    norm_v = Normalize(vmin=info["limits"]["Velocity"][0],
                       vmax=info["limits"]["Velocity"][1])
    norm_u = Normalize(vmin=info["limits"]["Internal energy"][0],
                       vmax=info["limits"]["Internal energy"][1])

    if half_drift is None:
        n_plots = len(entries)
    else:
        n_plots = sum((1 for e in entries.values()
                      if e["half_drift"] == half_drift))

    fig = plt.figure(figsize=(3 * n_plots, 12))
    axes = ImageGrid(fig,
                     rect=111,
                     nrows_ncols=(4, n_plots),
                     share_all=True,
                     axes_pad=0.15,
                     cbar_location="right",
                     cbar_mode="edge",
                     cbar_size="7%",
                     cbar_pad=0.15)
    i = 0
    for data in entries.values():
        if half_drift is not None and data["half_drift"] != half_drift:
            continue
        col = axes.axes_column[i]
        plot_single(data["Coordinates"], data["Field"],
                    col[0], norm_field, title=data["title"], cmap="viridis")
        plot_single(data["Coordinates"], data["Density"],
                    col[1], norm_rho, cmap="turbo")
        plot_single(data["Coordinates"], data["Velocity"],
                    col[2], norm_v, cmap="turbo")
        plot_single(data["Coordinates"], data["Internal energy"],
                    col[3], norm_u, cmap="turbo")
        i += 1

    axes.axes_column[0][0].set_ylabel("Density field")
    axes.axes_column[0][1].set_ylabel("Density error")
    axes.axes_column[0][2].set_ylabel("Internal energy error")
    axes.axes_column[0][3].set_ylabel("Velocity error")

    time = round(info["time"], 1)
    if half_drift is None:
        title = f"Yee vortex errors at t={time}"
    elif half_drift:
        title = f"Yee vortex errors at t={time}, Half step schemes"
    else:
        title = f"Yee vortex errors at t={time}, Full step schemes"
    fig.suptitle(title)
    fig.tight_layout()

    if savename_base is not None:
        root = get_root()
        if half_drift is None:
            savename = f"{savename_base}_{round(time):03}.png"
        elif half_drift:
            savename = f"{savename_base}_half_step_{round(time):03}.png"
        else:
            savename = f"{savename_base}_full_step_{round(time):03}.png"
        fig.savefig(root / f"run/{savename}", dpi=300)

    plt.show()


if __name__ == "__main__":
    high_cfl = False
    low_cfl = False
    n = 100
    if high_cfl:
        entries = {
            f"yee_{n}_optimal_high_cfl": dict(title="Optimal", half_drift=False),
            f"yee_{n}_pakmor_high_cfl": dict(title="Pakmor", half_drift=False),
            f"yee_{n}_optimal_half_high_cfl": dict(title="Optimal, half drift", half_drift=True),
            f"yee_{n}_meshless_gradient_half_high_cfl": dict(title="Meshless gradient", half_drift=True),
            # f"yee_{n}_two_volume_half_high_cfl": dict(title="Two Voronoi", half_drift=True),
        }
    else:
        entries = {
            # f"yee_{n}_default": dict(title="Default", half_drift=False),
            # f"yee_{n}_optimal": dict(title="Optimal", half_drift=False),
            f"yee_{n}_pakmor": dict(title="Pakmor", half_drift=False),
            # f"yee_{n}_optimal_half": dict(title="Optimal, half drift", half_drift=True),
            f"yee_{n}_meshless_gradient_half": dict(title="Meshless gradient", half_drift=True),
            # f"yee_{n}_flux_extrapolate_half": dict(title="Flux extrapolate", half_drift=True),
            # f"yee_{n}_two_volume_half": dict(title="Two Voronoi", half_drift=True),
            # f"yee_{n}_two_volume_flux_ext_half": dict(title="Two Voronoi, flux extrapolate", half_drift=True),
        }

    for n in range(1, 6):
        this_entries = dict(entries)
        general_info = load_all(this_entries, n)
        if high_cfl:
            savename_base = f"Yee_{n}_high_cfl"
        elif low_cfl:
            savename_base = f"Yee_{n}_low_cfl"
        else:
            savename_base = f"Yee_{n}"
        plot_comparison(this_entries, general_info,
                        savename_base, half_drift=None)
