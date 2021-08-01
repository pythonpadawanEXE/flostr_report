from pathlib import Path

from matplotlib.pyplot import subplots
import numpy as np

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

res = AutoReg(zeta, 2, old_names=False).fit()
print(res.summary())

asymptote = res.params[0] / (1 - res.params[1:].sum())
print("asymptote:", asymptote)

mu = 1 / res.roots
print("Spectrum:", mu)
s = np.log(mu) / downsampling
print("s:", s)

sigma = s.real.mean()
omega = abs(np.diff(s.imag)[0]) / 2
print("Growth rate:", sigma)
print("omega:", omega)

period = 2 * np.pi / omega
print("period:", period, " time-steps")
print("decay time:", -1 / sigma, " time-steps")

modes = np.vander(mu, zeta.size, True).T
coefficients = np.linalg.lstsq(modes, zeta - asymptote, rcond=None)[0]
print("coefficients:", coefficients)

step = np.arange(zeta.size) * downsampling
fig, ax = subplots()
ax.set_title('AR(2) with trend="c"')
ax.set_xlabel("relative time-step")
ax.set_ylabel("vorticity")
ax.plot(step, zeta, marker="o", linestyle="None", color="green", label="data")

kmax = np.argmax(zeta)
amplitude = zeta[kmax] - asymptote
print("amplitude:", amplitude)
ax.plot(
    step,
    asymptote + amplitude * np.exp(sigma * (step - kmax * downsampling)),
    linestyle="dotted",
    color="k",
    label="AR(2) envelope",
)
ax.plot(
    step,
    asymptote + np.real_if_close(modes @ coefficients),
    color="r",
    label="AR(2) reconstruction",
)
ax.vlines(
    np.arange(6) * period,
    zeta.min(),
    asymptote,
    linestyle="dashed",
    color="b",
    label="AR(2) period",
)
ax.legend()
fig.savefig(Path(__file__).with_suffix(".png"))
