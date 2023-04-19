import matplotlib.pyplot as plt
import numpy as np

from utils import read_particle_data, plot_faces, get_root, yee_quantities, get_slice

GAMMA = 1.666666667
T_INF = 1.
BETA = 5.

if __name__ == "__main__":
    root = get_root()
    fname = root / "run/output/yee_25_0017.hdf5"
    data, _ = read_particle_data(fname)
    # faces = plot_faces(fname)
    coordinates = data.loc[:, ["x", "y", "z"]].values
    centroids = data.loc[:, ["c_x", "c_y", "c_z"]].values
    centred_centroids = centroids
    centred_centroids[:, :2] -= 5
    density = data["rho"].values
    internal_energy = data["u"].values
    entropy = data["S"].values
    density_s, _, internal_energy_s = yee_quantities(
        centred_centroids, T_INF, GAMMA, BETA)

    delta_rho = np.abs(density_s - density)
    delta_P = np.abs(internal_energy_s - internal_energy)

    lim = [0, 10]
    slice_rho = get_slice(delta_rho, coordinates, x_lim=lim, y_lim=lim)
    slice_density = get_slice(density, coordinates, x_lim=lim, y_lim=lim)
    slice_u = get_slice(delta_P, coordinates, x_lim=lim, y_lim=lim)
    slice_internal_energy = get_slice(
        internal_energy, coordinates, x_lim=lim, y_lim=lim)
    slice_s = get_slice(entropy, coordinates, x_lim=lim, y_lim=lim)

    fig, ((ax11, ax12, ax13), (ax21, ax22, _)) = plt.subplots(2, 3, figsize=(10, 5))

    im = ax11.imshow(slice_density)
    plt.colorbar(im, ax=ax11)
    im = ax12.imshow(slice_internal_energy)
    plt.colorbar(im, ax=ax12)
    im = ax13.imshow(slice_s)
    plt.colorbar(im, ax=ax13)
    im = ax21.imshow(slice_rho)
    plt.colorbar(im, ax=ax21)
    im = ax22.imshow(slice_u)
    plt.colorbar(im, ax=ax22)

    plt.show()
