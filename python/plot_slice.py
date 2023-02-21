if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    from utils import get_slice, read_particle_data, get_root

    data = read_particle_data(get_root() / "run" / "output" / "KelvinHelmholtz_boosted_2D_0002.txt")
    coords = np.column_stack([data["x"].values, data["y"].values, data["z"].values])
    values = data["rho"]
    x_lim = [0., 1.]
    y_lim = [0., 1.]
    res = 1500

    interpolated = get_slice(values, coords, x_lim, y_lim, res)
    plt.imshow(interpolated, cmap="viridis")
    plt.show()
    
