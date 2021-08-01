from pathlib import Path

from matplotlib.pyplot import subplots
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import eigsh

import meshio


"""

"Method of snapshots" proper orthogonal decomposition
with mass matrix appropriate to the finite element
discretization of the vorticity, so that the squared norm
is proportional to the enstrophy.

Reference: 

* Schmidt & Colonius (2020) Guide to spectral proper orthogonal decomposition. AIAA Journal 58(3):1023-1033

"""


def main(filename: Path, n: int = 1000, r: int = 6):

    W = load_npz("mass1.npz")

    with meshio.xdmf.TimeSeriesReader(filename) as reader:

        points, cells = reader.read_points_cells()

        qT = np.array([reader.read_data(k)[1]["vorticity"] for k in range(-n, 0)])
    xT = qT - qT.mean(0)
    eiv, Psi = eigsh(xT @ W @ xT.T, k=r)  # Schmidt & Colonius (2020, eq. 17)
    print(eiv)
    Phi = xT.T @ Psi  # Schmidt & Colonius (2020, eq. 15)

    with meshio.xdmf.TimeSeriesWriter(Path(__file__).with_suffix(".xdmf")) as writer:

        writer.write_points_cells(points, cells)

        for lambda_i, phi_i in zip(eiv, Phi.T):
            writer.write_data(lambda_i, point_data={"vorticity": phi_i})


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        type=Path,
        default=Path(__file__).with_name("st08_navier_stokes_cylinder.xdmf"),
    )
    args = parser.parse_args()

    main(Path(args.filename))
