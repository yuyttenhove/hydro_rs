if __name__ == "__main__":
    import sys
    from utils import plot_faces
    from pathlib import Path

    try:
        plot_faces(sys.argv[1])
    except IndexError:
        root = Path(__file__).parent.parent
        plot_faces(root / "run/output/sodshock_2D_optimal_0002.hdf5")