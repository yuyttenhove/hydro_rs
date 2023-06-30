import matplotlib.pyplot as plt
from utils import yee_quantities, read_particle_data, get_root
import numpy as np

GAMMA = 1.666666667
T_INF = 1.
BETA = 5.

def l1_norm(fname, center = 5):
    results = dict()
    data, time = read_particle_data(fname)
    results["time"] = time

    velocities = data.loc[:, ["v_x", "v_y"]].values
    centroids = data.loc[:, ["c_x", "c_y"]].values - center
    rho_s, velocity_s, internal_energy_s = yee_quantities(centroids, T_INF, GAMMA, BETA)
    n_part = len(rho_s)
    results["rho"] = np.sum(np.abs(data["rho"].values - rho_s)) / n_part
    delta_v = velocities - velocity_s
    results["v"] = np.sum(np.sqrt(np.sum(delta_v * delta_v, axis=1))) / n_part
    results["u"] = np.sum(np.abs(data["u"].values - internal_energy_s)) / n_part
    
    angular_momentum = velocities[:, 0] * centroids[:, 1] - velocities[:, 1] * centroids[:, 0]
    angular_momentum_s = velocity_s[:, 0] * centroids[:, 1] - velocity_s[:, 1] * centroids[:, 0]
    results["L"] = abs(angular_momentum - angular_momentum_s) / abs(angular_momentum_s)

    return results



if __name__ == "__main__":
    n = 1

    root = get_root()
    resolutions = [50, 100, 200, 400]
    fnames_pakmor = [root / f"run/output/yee_{res}_pakmor_{n:04}.hdf5" for res in resolutions]
    fnames_meshless = [root / f"run/output/yee_{res}_meshless_gradient_half_{n:04}.hdf5" for res in resolutions]

    results_pakmor = [l1_norm(fname) for fname in fnames_pakmor]
    results_meshless = [l1_norm(fname) for fname in fnames_meshless]

    plt.plot(resolutions, [r["rho"] for r in results_pakmor], label="Pakmor")
    plt.plot(resolutions, [r["rho"] for r in results_meshless], label="Meshless gradients")
    plt.plot([50, 300], [1e-3, 1e-3 / 36], "--", c="black", label="N^-2")
    plt.title("L1 convergence of density")
    plt.legend()
    plt.loglog()
    plt.savefig("L1_rho.png", dpi=300)
    plt.show()



