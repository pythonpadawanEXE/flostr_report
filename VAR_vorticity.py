from pathlib import Path
from typing import Tuple

from matplotlib.patches import Ellipse
from matplotlib.pyplot import subplots
import numpy as np
from scipy.sparse import diags, load_npz

import meshio

import modred as mr

# References

# Le Clainche, S. & Vega, J. M. (2017). Higher Order Dynamic Mode
# Decomposition. _SIAM Journal on Applied Dynamical Systems,_ *16*:882--925.
# doi: 10.1137/15m1054924

# Mackey, D. S., Mackey, N. & Tisseur, F. (2015). Polynomial Eigenvalue
# Problems: Theory, Computation, and Structure. In P. Benner,
# M. Bollhöfer, D. Kressner, C. Mehl & T. Stykel (eds.), Numerical Algebra,
# Matrix Theory, Differential-Algebraic Equations and Control Theory:
# Festschrift in Honor of Volker Mehrmann (pp. 319--348). Springer.
# ISBN: 978-3-319-15260-8

# Wikipedia 'General matrix notation of a VAR(p)',
# https://en.wikipedia.org/wiki/General_matrix_notation_of_a_VAR(p)


def L(Q: np.ndarray, mu: np.ndarray, m: int) -> np.ndarray:
    M = diags(mu)
    return np.vstack([Q @ M ** k for k in range(m)])


def varp(
    snapshots: np.ndarray, p: int, trend: str = "c"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return matrix containing horizontally stacked VAR coefficients, eigenvalues, and companion eigenvectors

    The first column is the constant inhomogeneous term, the succeeding
    squares multiply the sumccessive lags from 1 through p.

    The snapshots should also be horizontally stacked columns.
    """
    if trend != "c":
        raise NotImplementedError

    r, m = snapshots.shape

    # The notation Y = B Z is from
    # https://en.wikipedia.org/wiki/General_matrix_notation_of_a_VAR(p)

    Y = snapshots[:, p:]
    Z = np.vstack(
        [np.ones(m - p), *[snapshots[:, r : r + m - p] for r in range(p - 1, -1, -1)]]
    )
    B = np.linalg.lstsq(Z.T, Y.T, rcond=None)[0].T

    uinf = np.linalg.solve(
        np.eye(r) - sum(B[:, i : i + r] for i in range(1, p * r + 1, r)), B[:, 0]
    )

    # first Frobenius companion form (Mackey, Mackey, & Tisseur 2015, Section 3)

    companion = np.vstack([B[:, 1:], np.eye((p - 1) * r, p * r)])
    mu, imbedded_eifs = np.linalg.eig(companion)
    eifs = imbedded_eifs[:r]

    # reconstruction (Le Clainche & Vega 2017 SIADS, Sections 2.1.3, 2.2.3)

    ell = L(eifs, mu, m)
    lstsq = np.linalg.lstsq(ell, (snapshots - uinf[:, None]).flatten("F"), rcond=None)
    a = lstsq[0]

    return B, mu, eifs[:r], uinf, a, ell


def logarithmic_marker_size(
    a: np.ndarray, size: float = 1.0, cutoff: float = 1e-3
) -> np.ndarray:
    """return logarithmic sizes for scatter-plot markers

    such that:

    * max(a) has size `size`

    * `cutoff * max(a)` has size 0

    * size varies linearly inbetween as log(a/max(a))

    """

    a1 = max(a)
    a0 = cutoff * a1
    return size * (1 - np.log(a / a1) / np.log(a0 / a1))


def main(
    filename: Path, mass: Path, start_n: int = 4000, end_n: int = 5000, r: int = 21
):
    W = load_npz(mass)

    with meshio.xdmf.TimeSeriesReader(filename) as reader:

        points, cells = reader.read_points_cells()
        downsampling = 1  # 11
        qT = np.array(
            [
                reader.read_data(k)[1]["vorticity"]
                for k in range(start_n, end_n, downsampling)
            ]
        )

    xT = qT - qT.mean(0)
    x = xT.T

    POD = mr.compute_POD_arrays_direct_method(
        x, inner_product_weights=W.A, mode_indices=range(r)
    )

    p = 2
    r = 8
    downsampling = 16
    v = POD.proj_coeffs[:r, ::downsampling]
    m = v.shape[1]
    B, mu, eifs, uinf, a, ell = varp(v, p)

    print("mu:", mu)

    fig, ax = subplots()
    fig.suptitle("Periodic 2-D flow over a cylinder")
    ax.set_title(rf"Prony roots $\mu$ ($r$={r}, $p$={p}, downsamping={downsampling})")
    ax.set_axis_off()
    ax.axhline()
    ax.axvline()
    ax.plot(
        *np.stack(
            [(np.cos(theta), np.sin(theta)) for theta in np.linspace(0, 2 * np.pi)]
        ).T,
        "k--",
    )
    ax.set_aspect(1.0)

    ax.plot(mu.real, mu.imag, marker="x", linestyle="None")

    for root, width, height in zip(
        mu, *[logarithmic_marker_size(abs(z), 0.2) for z in [a.real, a.imag]]
    ):
        ax.add_artist(Ellipse(xy=(root.real, root.imag), width=width, height=height))

    fig.savefig(Path(__file__).with_name(Path(__file__).stem + "-roots.png"))

    s = np.log(mu)
    print("s:", s)
    print("period:", 2 * np.pi / s.imag)

    print("uinf:", uinf)
    print("a:", a)

    reconstruction = (ell @ a).reshape((r, m), order="F") + uinf[:, None]

    fig, axes = subplots(r // 2, sharex=True)
    colors = []
    for i, u in enumerate(POD.proj_coeffs[:r]):
        (line,) = axes[i // 2].plot(u, label=None if i else "chronos")
        colors.append(line.get_color())
    k = np.arange(m) * downsampling
    for i, (u, color) in enumerate(zip(np.real_if_close(reconstruction), colors)):
        axes[i // 2].plot(
            k,
            u,
            color=color,
            marker="x",
            linestyle="None",
            label=None if i else f"VAR({p})",
        )
    axes[0].hlines(
        uinf,
        xmin=0,
        xmax=POD.proj_coeffs.shape[1],
        colors=colors,
        linestyles="dashed",
        label="constant",
    )
    axes[-1].set_xlabel("sample")
    axes[-1].set_ylabel("vorticity")
    fig.suptitle("Flow over a cylinder")
    axes[0].legend()
    fig.savefig(Path(__file__).with_suffix(".png"))


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        type=Path,
        default=Path(__file__).with_name("st08_navier_stokes_cylinder.xdmf"),
    )
    parser.add_argument(
        "-m", "--mass", type=Path, default=Path(__file__).with_name("mass1.npz")
    )

    args = parser.parse_args()

    main(**vars(args))
