from pathlib import Path
from typing import Tuple

import numpy as np

from matplotlib.pyplot import subplots
import meshio


def main(filename: Path, probes: np.ndarray = np.array([4, 5])):

    with meshio.xdmf.TimeSeriesReader(filename) as reader:

        points, _ = reader.read_points_cells()
        print("Points:", points[probes, :2])

        time = []
        pressure = []

        for k in range(reader.num_steps):
            t, pd, _ = reader.read_data(k)
            time.append(t)
            pressure.append(pd["pressure"][probes])

    pressure = np.array(pressure).T
    np.savetxt(Path(__file__).with_suffix(".csv"), pressure.T, delimiter=",")

    # KLUDGE gdmcbain 2021-03-11: It's a bit tricky deciding what limits to use
    # when plotting the pressure since there are usually wild oscillations at
    # the very start.  The kludge here is to use the values from the second half.
    limits = np.stack(
        [
            pressure[:, pressure.shape[1] // 2 :].min(1),
            pressure[:, pressure.shape[1] // 2 :].max(1),
        ]
    ).T
    
    fig, axs = subplots(2, 2, sharex="col", sharey="row")
    axs[1][0].plot(pressure[0], pressure[1], marker=",", linestyle="None")
    axs[1][0].set_xlabel("fore pressure")
    axs[1][0].set_xlim(limits[0])
    axs[1][0].set_ylabel("aft pressure")
    axs[1][0].set_ylim(limits[1])
    axs[0][0].plot(pressure[0], time, marker=",", linestyle="None")
    axs[0][0].set_ylabel("time")
    axs[1][1].plot(time, pressure[1], marker=",", linestyle="None")
    axs[1][1].set_xlabel("time")
    axs[0][1].axis("off")
    fig.suptitle("Pressure at fore & aft stagnation points")
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
    args = parser.parse_args()

    main(Path(args.filename))
