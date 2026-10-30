# spectral reconstruction analysis toolkit (SpectRAL)

Python library for linear reconstruction of weighted spectral integrals from Euclidean data for applications in lattice QCD.

For a short overview of the API and application workflows, open the
[HTML documentation](doc/index.html) in a browser.

The distribution is named `spectral-reconstruction`; the import is `spectral`.

## Install and test

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[test,jks]'
.venv/bin/python -m pytest
.venv/bin/python examples/two_pion_scan.py --workers 2
.venv/bin/python examples/jks_predictions.py
.venv/bin/python examples/multi_ensemble_scan.py
```

To run directly from the source tree, add `src` to `PYTHONPATH` from the repository
root: `export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"`.

## Example notebooks

The /notebook folder covers two self-contained, executed
examples: [exact two-pion correlator reconstruction](notebooks/01_two_pion_exact.ipynb)
and [the same reconstruction with Gaussian noise](notebooks/02_two_pion_gaussian_noise.ipynb).
Both plot the R ratio and Euclidean correlator, reconstruct a withheld time with
an HLT lambda scan, and estimate model-based systematic uncertainty. The second
also propagates the full input covariance and compares systematic and statistical
errors, including a Gaussian-toy check.


## Mathematical convention

The data describe `C(t) = integral rho_p(omega) phi_t(omega) d omega`, with
`phi_t(omega) = omega**p B(t, omega)` and integration over `[omega_0, infinity)`.
The forward kernel is exponential, or the sum of forward and backward
exponentials for a finite periodic temporal extent. The reconstructed integral
is `q @ C`, and its approximated kernel is `sum(q_i * phi_i)`.


## Reconstruction

```python
import spectral as sp
from spectral.algorithms import hlt

data = sp.CorrelatorData.from_arrays(
    times=[1, 2, 3], mean=["1.0", "0.6", "0.4"],
    covariance=[[".01", 0, 0], [0, ".008", 0], [0, 0, ".006"]],
)
basis = sp.Basis(
    kernel=sp.kernels.exponential(),
    times=data.times, p=2, omega_0="0.3",
)
solver = hlt.Solver(
    lambda_=["0", "1e-12", "1e-6"],
    regulator="identity",
    constraints=[hlt.constraints.area()],
)
reconstructor = sp.Reconstructor(
    basis=basis, algorithm=solver, numerics=sp.numerics.mpmath(dps=60),
)
prediction = reconstructor.reconstruct(
    data, target=sp.targets.correlator(basis=basis), target_times=[2, 4, 5],
)
result = prediction.at_lambda("1e-6").at_time(4)
print(result.value, result.statistical_error)
print(result.coefficients)
print(result.approximated_kernel(["0.3", "0.5", "1"]))
```

Identity regularization is the default. Covariance regularization requires
`regulator="covariance", covariance_norm=...`; the matrix used is
`data.covariance / covariance_norm`. Statistical errors always use the original
data covariance. Both choices preserve the original parameter convention:
`W = (1-lambda)*A + lambda*regulator`, with `0 <= lambda < 1`.
The range is checked again after conversion to the selected backend. If a lambda
very close to one rounds to one, reconstruction raises an error (or records a
failed point with `on_error="record"`); use mpmath with sufficient precision.

## Targets and scans

```python
targets = sp.targets.grid(
    sp.targets.smooth_high_energy,
    vary={"width": [".05", ".1", ".2"]},
    kernel=basis.at_time(2), cutoff=".7",
)
with sp.execution.processes(workers=4) as execution:
    scan = reconstructor.reconstruct(data, target=targets, execution=execution)

print(scan.axes)                  # width, lambda
print(scan.values.shape)          # (3, 3)
print(scan.coefficients.shape)    # (3, 3, 3 input times)
one = scan.select(width=".1", lambda_="1e-6")
```


Built-ins include Gaussian targets, smooth and sharp high-energy cuts, full HVP,
and standard FULL/SD/ID/LD HVP windows:

```python
hvp = sp.targets.hvp_window(name="ID", a_lat_inv_MeV="3000", p=2)
target = sp.targets.smooth_high_energy(kernel=hvp, cutoff=".3", width=".05")
```

All window-specific mathematics and caches live in
`targets/intermediate_window.py` and `targets/_hvp_math.py`. Arbitrary-precision
window integration is supported. The numerical backend
contains no list of recognized target classes. A custom kernel implements
`scalar(omega, ctx)`, or users can pass a plain Python function as described below.
The existing `CallableKernel(function)` interface accepts `function(omega, ctx)`.
Use `ctx.number`, `ctx.exp`, etc. to retain arbitrary precision. Optional
`basis_integrals(basis, ctx)` supplies optimized overlaps and squared norms.

## Custom target functions

Pass a function of energy and target time directly to reconstruction:

```python
import numpy as np

def mykernel(omega, t):
    return omega**2 * np.exp(-t * omega)

prediction = reconstructor.reconstruct(
    data,
    target=mykernel,
    target_times=[2, 4, 5],
)
```

The function returns the **complete target kernel**, including any `omega**p`
factor; the library does not multiply that factor in automatically. The target
must have finite overlaps and squared norm over the basis integration domain.
Time labels are converted to the active backend's numerical type before calling
the function. A two-argument function requires `target_times`; a function with
signature `mykernel(omega)` is fixed and rejects `target_times`. Use
`functools.partial` to bind additional parameters or fix a time.

For a periodic target, the function can contain both exponentials:

```python
T = 64

def mykernel(omega, t):
    return omega**2 * (
        np.exp(-t * omega) + np.exp(-(T - t) * omega)
    )
```

NumPy and `math` implementations are intended for the NumPy/binary64 backend.
For arbitrary precision, accept an optional **keyword `ctx`** and use its math
operations. This version works with either numerical backend:

```python
def mykernel(omega, t, *, ctx):
    return omega**2 * ctx.exp(-t * omega)
```

Fixed functions can likewise use `def mykernel(omega, *, ctx)`. The existing
`sp.CallableKernel` interface is unchanged. Calling `math` inside a high-precision
calculation does not make the returned values high precision.

Use `from_function` when attaching conventions or quadrature breakpoints:

```python
target = sp.targets.from_function(
    mykernel,
    spectral_power=2,
    # energy_unit="GeV",       # Must agree with the basis if supplied.
    # breakpoints=(".5",),     # Known discontinuities or integration features.
)
fixed_target = target.at_time(2)
```

Function targets also work as `reference_kernel` in systematic estimation and
inside smooth/sharp energy cuts. For example:

```python
targets = sp.targets.grid(
    sp.targets.smooth_high_energy,
    vary={"width": [".05", ".1", ".2"]},
    kernel=mykernel,
    cutoff=".7",
)
scan = reconstructor.reconstruct(data, target=targets, target_times=[2, 4])
```

Serial execution supports locally defined functions and lambdas. For process
execution, define the function at module scope in an importable Python file and
protect the process-launching code with `if __name__ == "__main__":`.
Native archives and checkpoints do not serialize executable Python functions;
use a registered kernel dataclass when native persistence is required. CSV
summaries remain available for results using function targets.

## Physical spectral densities and systematic estimation

```python
density = sp.models.TwoPionModel()  # GeV energies; use matching data/basis units
rho = density(["0.5", "0.8", "1.0"])
physical_kernel = sp.kernels.exponential(energy_unit="GeV").at_time(2)
C2 = density.integrate(physical_kernel, numerics=sp.numerics.mpmath(50))

sharp = sp.targets.sharp_high_energy(kernel=basis.at_time(2), cutoff=".7")
report = sp.systematics.estimate(scan, reference_kernel=sharp, density=density)
```

`sp.SpectralDensity` defines a physical density per unit energy. Implementations
live in `sp.models`: `TwoPionModel` evaluates
`rho_phys(omega) = omega**2 * R(omega**2)/(12*pi**2)`, and
`TabulatedSpectralDensity(energies, values)` interpolates physical density samples.
Neither model takes a spectral power or a basis. Call `density(energies)` or
`density.scalar(omega, ctx)` to evaluate the physical density.

`density.integrate(kernel, ...)` (also `sp.integrate(density=..., kernel=...)`)
integrates that density times the supplied physical kernel. It clips to the
density's domain and includes its quadrature breakpoints; an unbounded domain
is integrated to infinity. `density.integrand(kernel)` returns the corresponding
callable energy-space integrand. Physical integration does not convert powers.

Set `p` on `Basis`. All conversion to `rho_p = rho_phys / omega**p` lives in
`sp.systematics`, and estimation obtains `p` from each result's basis, including
for deformations. For a basis-weighted kernel, use:

```python
integrand = sp.systematics.integrand(density=density, kernel=basis.at_time(2))
values = integrand(["0.5", "0.8", "1.0"])
rho_p = sp.systematics.for_basis(density, basis)
```

To add a model, subclass `sp.SpectralDensity` and implement `scalar(omega, ctx)`;
declare `domain`, `energy_unit`, and optional `breakpoints`. The base class supplies
array evaluation and physical integration. For a callable, use
`sp.SpectralDensity.from_function(function, domain=(...), energy_unit=...)`,
where `function(omega, ctx)` returns the physical density.



## JKS and native storage

`read_correlator(database, observable)` reads every stored timeslice, numbered
`0, 1, ..., N-1` by default. Use `.select_times([1, 2])` afterwards to keep
timeslices 1 and 2, selecting the corresponding mean and covariance while
retaining the original resamples. If the stored vector starts at time 1 or uses
another grid, pass its full time axis with the optional `times=` argument.

```python
data = sp.io.jks.read_correlator(database, observable="C")
data = data.select_times(input_times)
# Reconstruct using this data. Each result retains its original source mapping.
observable = sp.io.jks.to_jks(prediction, source=database)
observable = sp.io.jks.with_systematic(
    observable, tag="reconstruction/ensemble_A",
    shift=-prediction_report.bias_at_maximum,
)
database.add("C_predicted", observable)
database.save("analysis.jks")
```


```python
sp.io.save("scan.spectral", results=scan, systematics=report)
saved = sp.io.load("scan.spectral")
sp.io.write_csv("summary.csv", results=saved.results, systematics=saved.systematics)
```


## Validation

The test suite covers independent two-pion forward transforms, both regulators,
different spectral powers and thresholds, periodic kernels, constraints,
serial/process agreement, systematic envelopes, JKS covariance, and native
round trips. `tests/_reference/` preserves the supplied implementation for
regression comparisons and is excluded from package installation and automatic
test discovery. Comparisons requiring the unbundled legacy application or supplied
two-pion model skip explicitly when those optional reference files are absent.
The application scan, archive and resume checks use synthetic inputs.

## AI statement

The core functionality and software architecture were designed by the author,
while most of the code was generated using AI. The HLT implementation has been
independently checked by human reviewers. Correctness is not guaranteed for every
input or edge case, and users should validate results for their intended
application.

