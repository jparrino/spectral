# Two-pion reconstruction notebooks


| Notebook | Calculation |
| --- | --- |
| Exact correlator | Plot the two-pion R ratio and its Euclidean transform; use times 1, 2, 3, 4 to reconstruct the withheld time 5; scan ten lambdas; inspect kernels, signed biases, and systematic envelopes; compare 50- and 70-digit calculations. |
| Gaussian noise | Repeat with one seeded Gaussian measurement and a known covariance; propagate statistical errors; validate them with 20,000 repeated Gaussian draws; plot systematic and statistical uncertainties versus lambda. |


## Conventions and interpretation

The two-pion toy model uses energies in GeV, times in GeV^-1, and correlators in
GeV^3. With spectral power `p=2`, the density is
`rho_2(omega) = R(omega**2)/(12*pi**2)` and the basis functions are
`phi_t(omega) = omega**2 * exp(-omega*t)`. Direct transforms and systematic
integrals extend from the two-pion threshold to infinity. Plot limits do not
truncate these integrals.

Set `p` only on `Basis`; construct the physical density with
`sp.models.TwoPionModel()`. Its value is `omega**2 * R(omega**2)/(12*pi**2)`.
Forward integrals use `model.integrate(basis.kernel.at_time(t), numerics=...)`
with an ordinary exponential kernel. Systematic estimation takes `density=model`
and resolves the basis power automatically for the nominal density and variations.
For a basis-weighted energy-space integrand, use
`sp.systematics.integrand(density=model, kernel=basis.at_time(t))`.

Both notebooks use identity regularization, no extra constraints, and
`0 <= lambda < 1`, minimizing `(1-lambda)*kernel_mismatch + lambda*(q @ q)`.
The withheld correlator is used only for validation. Its absence from the basis
makes the reconstruction approximate even at lambda zero.

The signed nominal bias is the integral of `(approximated-target)*rho_2`.
The systematic envelope is the largest absolute bias over the nominal model
and four illustrative variations: rho mass shifted by +/-10 MeV, or rho width
shifted by +/-10%, one parameter at a time. These are sensitivity choices,
not measured parameter errors or a confidence region. Pass `deformations=None`
to `sp.systematics.estimate` to keep only the nominal model. Central values are
not bias-corrected.

The noisy example uses 0.5% relative Gaussian errors with correlation 0.95 across
times; set `CORRELATION=0.0` for independent noise. Its covariance describes the
measured mean vector, and the statistical variance is `q @ covariance @ q`.
The 20,000 toys check this variance; they do not reduce the uncertainty of the
displayed measurement. Every lambda uses the same measured input, so scan points
are correlated. Model variation and measurement noise are separate calculations.

HLT solves and quantitative model integrals use mpmath at 50 decimal digits;
plots and Gaussian random numbers use binary64. The notebooks include assertions
for the bias identity, withholding of the target time, covariance propagation,
precision stability, and Monte Carlo agreement.
