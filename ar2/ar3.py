from pathlib import Path

from matplotlib.pyplot import subplots
import numpy as np
from scipy.linalg import hankel

from statsmodels.tsa.ar_model import AutoReg

zeta = np.loadtxt(
    Path(__file__).with_name("probe_vorticity.pyRe_72.0.csv"),
    delimiter=",",
    usecols=0,
    skiprows=10000,
    max_rows=10000,
)
downsampling = 2 ** 3
zeta = zeta[::downsampling]

res = AutoReg(zeta, 3, trend="n", old_names=False).fit()
print(res.summary())
unity_arg = np.argmin(abs(res.roots - 1))
nonunity_args = np.setdiff1d(np.arange(res.roots.size), unity_arg)
mu = 1 / res.roots[nonunity_args]
print("Spectrum:", mu)
s = np.log(mu) / downsampling
print("s:", s)

sigma = s.real.mean()
omega = abs(np.diff(s.imag)[0]) / 2
print("Growth rate:", sigma)
print("omega:", omega)

period = 2 * np.pi / omega
print("period:", period)

modes = np.vander(1 / res.roots, zeta.size, True).T
coefficients = np.linalg.lstsq(modes, zeta, rcond=None)[0]
print("coefficients:", coefficients)

step = np.arange(zeta.size) * downsampling
fig, ax = subplots()
ax.set_title('AR(3) with trend="n"')
ax.set_xlabel("relative time-step")
ax.set_ylabel("vorticity")
ax.plot(step, zeta, marker="o", linestyle="None", color="green", label="data")
ax.plot(
    step,
    np.real_if_close(modes @ coefficients),
    color="r",
    label="AR(3) reconstruction",
)

ax.vlines(
    np.arange(6) * period,
    zeta.min(),
    zeta.mean(),
    linestyle="dashed",
    color="blue",
    label="AR(3) period",
)
ax.legend()
fig.savefig(Path(__file__).with_suffix(".png"))
