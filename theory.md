# Theory — Path Integral Monte Carlo for the 0+1D Anharmonic Oscillator

This repository implements **Path Integral Monte Carlo (PIMC)** for a one-dimensional quantum particle in an **anharmonic (quartic) potential**. The core idea is to rewrite finite-temperature quantum expectation values as classical-looking averages over **closed paths in imaginary time**, then estimate them with Markov Chain Monte Carlo.

---

## 1. Physical system and target quantities

We consider a single quantum degree of freedom \(x\) with Hamiltonian
\[
H = \frac{p^2}{2m} + V(x),
\qquad
V(x)=\frac{1}{2}m\omega^2 x^2 + \lambda x^4,
\qquad \lambda \ge 0,
\]
in units where \(\hbar=1\) (default; \(m,\omega,\lambda\) are configurable).

### Finite-temperature partition function

At inverse temperature \(\beta = 1/T\), the partition function is
\[
Z(\beta) = \mathrm{Tr}\left(e^{-\beta H}\right).
\]

### Euclidean two-point function

A central observable is the **imaginary-time correlator**
\[
G(\tau) = \langle x(\tau)\,x(0)\rangle_\beta,
\qquad 0\le \tau \le \beta,
\]
where \(x(\tau) = e^{\tau H} x e^{-\tau H}\) is the Heisenberg operator in imaginary time.
In operator form,
\[
G(\tau)
=
\frac{1}{Z}\,\mathrm{Tr}\!\left(e^{-(\beta-\tau)H}\,x\,e^{-\tau H}\,x\right).
\]

This correlator is extremely useful because it encodes the **spectral gaps** of the system.

---

## 2. Why the correlator gives the energy gap

Insert a complete set of energy eigenstates \(H|n\rangle = E_n|n\rangle\) into the trace:
\[
G(\tau)
=
\frac{1}{Z}\sum_{m,n}
e^{-(\beta-\tau)E_m}\,e^{-\tau E_n}\,|\langle m|x|n\rangle|^2.
\]

For sufficiently large \(\beta\), \(Z \approx e^{-\beta E_0}\) and the dominant contributions come from the ground state \(m=0\):
\[
G(\tau)
\approx
\sum_{n} |\langle 0|x|n\rangle|^2
\left(e^{-(E_n-E_0)\tau} + e^{-(E_n-E_0)(\beta-\tau)}\right).
\]

Defining \(\Delta_n = E_n - E_0\), this becomes
\[
G(\tau)\approx \sum_n A_n\left(e^{-\Delta_n\tau}+e^{-\Delta_n(\beta-\tau)}\right).
\]

For an even potential \(V(x)=V(-x)\), the operator \(x\) is odd under parity, so it couples the ground state mainly to odd excited states; often a single term dominates:
\[
G(\tau) \simeq A\left(e^{-\Delta\tau}+e^{-\Delta(\beta-\tau)}\right)
=2A\,e^{-\Delta\beta/2}\cosh\!\big(\Delta(\beta/2-\tau)\big).
\]

Therefore, by measuring \(G(\tau)\) we can extract the **energy gap**
\[
\Delta = E_1 - E_0
\]
either via an effective-mass plateau or via a direct cosh fit.

---

## 3. From quantum mechanics to a path integral

### 3.1 Euclidean path integral for \(Z\)

Start from
\[
Z = \mathrm{Tr}(e^{-\beta H}).
\]
Discretize imaginary time into \(N\) slices with step \(a=\beta/N\):
\[
e^{-\beta H} = \left(e^{-aH}\right)^N.
\]

Split \(H=T+V\), with \(T=p^2/(2m)\). Using a symmetric Trotter–Suzuki factorization,
\[
e^{-aH} = e^{-\frac{a}{2}V}\,e^{-aT}\,e^{-\frac{a}{2}V} + \mathcal{O}(a^3),
\]
so the full product has discretization error \(\mathcal{O}(a^2)\).

Insert \(N\) resolutions of identity in the position basis,
\[
\mathbf{1}=\int_{-\infty}^{\infty} \mathrm dx \, |x\rangle\langle x|,
\]
and evaluate the kinetic matrix element (a Gaussian kernel):
\[
\langle x'|e^{-aT}|x\rangle
=
\sqrt{\frac{m}{2\pi a}}\,
\exp\!\left(-\frac{m}{2a}(x'-x)^2\right).
\]

The result is a path integral over periodic trajectories \(x(\tau)\) with \(x(0)=x(\beta)\):
\[
Z
=
\int \mathcal{D}x(\tau)\,e^{-S_E[x]},
\qquad
S_E[x]=\int_{0}^{\beta}\left[\frac{m}{2}\dot{x}(\tau)^2 + V(x(\tau))\right]\mathrm d\tau.
\]

### 3.2 Lattice (discretized) Euclidean action

Discretize \(\tau_i = i a\), \(i=0,\dots,N-1\), and denote \(x_i = x(\tau_i)\) with periodic boundary \(x_N=x_0\).
Approximate the derivative:
\[
\dot{x}(\tau_i)\approx \frac{x_{i+1}-x_i}{a}.
\]

Then the discretized action becomes
\[
S(\{x_i\})
=
\sum_{i=0}^{N-1}
\left[
\frac{m}{2a}(x_{i+1}-x_i)^2
+
a\,V(x_i)
\right].
\]

This shows an important interpretation:

> The quantum problem in \(0+1\) dimensions maps to a classical “ring polymer” of \(N\) beads \(x_i\), with nearest-neighbor harmonic coupling (kinetic term) and on-site potential \(V(x_i)\).

---

## 4. Path Integral Monte Carlo (PIMC)

### 4.1 Sampling distribution

The discretized path integral defines the probability density
\[
\mathbb{P}(\{x_i\}) \propto e^{-S(\{x_i\})}.
\]
So estimating observables reduces to sampling paths \(\{x_i\}\) distributed according to \(e^{-S}\).

### 4.2 Metropolis updates (local, efficient)

A standard approach is a **local Metropolis** update:
- pick a site \(i\),
- propose \(x_i' = x_i + \delta\) (e.g. \(\delta\sim \mathrm{Uniform}[-w,w]\)),
- accept with probability
\[
p_{\text{acc}}=\min\left(1,e^{-\Delta S}\right),
\qquad
\Delta S = S(\ldots,x_i',\ldots)-S(\ldots,x_i,\ldots).
\]

Crucially, \(\Delta S\) is **local**: changing only \(x_i\) affects only terms involving \(x_{i-1},x_i,x_{i+1}\). Define the local contribution
\[
S_i(x_i)=\frac{m}{2a}\Big[(x_i-x_{i-1})^2+(x_{i+1}-x_i)^2\Big]+aV(x_i),
\]
then
\[
\Delta S = S_i(x_i') - S_i(x_i).
\]

This locality makes the sampler fast and easy to validate.

In practice one performs repeated sweeps over the lattice (often using even/odd “red–black” updates for vectorization), discards an initial burn-in phase, and then stores configurations at regular intervals.

---

## 5. Estimating the correlator from sampled paths

Given a single sampled path \(\{x_i\}\), a natural estimator for \(G(\tau)\) uses translational invariance along the Euclidean time circle:

For \(k=0,\dots,N-1\),
\[
g_k(\{x_i\})
=
\frac{1}{N}\sum_{i=0}^{N-1} x_i\,x_{i+k \;\mathrm{mod}\; N}.
\]
Then the Monte Carlo estimate is
\[
G_k \equiv G(\tau_k)\approx \langle g_k\rangle_{\text{MC}},
\qquad \tau_k = k a.
\]

Computationally, \(g_k\) is the **periodic autocorrelation** of \(x_i\) and can be computed efficiently using FFT:
\[
c_k = \sum_{i=0}^{N-1} x_i x_{i+k}
=
\mathrm{IFFT}\!\left(\mathrm{FFT}(x)\cdot \mathrm{FFT}(x)^\*\right)_k,
\qquad
g_k = \frac{c_k}{N}.
\]

---

## 6. Extracting the gap from \(G(\tau)\)

### 6.1 Effective mass (effective gap)

Assuming single-cosh dominance,
\[
G(\tau)\approx A\cosh(\Delta(\beta/2-\tau)).
\]
On the lattice, one can derive the identity
\[
\frac{G(\tau-a)+G(\tau+a)}{2G(\tau)} = \cosh(\Delta a),
\]
leading to the **effective gap**
\[
\Delta_{\text{eff}}(\tau)
=
\frac{1}{a}\operatorname{arcosh}\!\left(\frac{G(\tau-a)+G(\tau+a)}{2G(\tau)}\right).
\]

A plateau in \(\Delta_{\text{eff}}(\tau)\) (away from \(\tau\approx 0\) and \(\tau\approx \beta\)) is strong evidence that a single exponential dominates and that the extracted \(\Delta\) is reliable.

### 6.2 Direct cosh fit

Alternatively, fit \(G(\tau)\) (in a chosen window) to
\[
G(\tau)=A\left(e^{-\Delta\tau}+e^{-\Delta(\beta-\tau)}\right),
\]
using uncertainties from bootstrap or binning to perform a weighted fit.
This provides \(\Delta\) and an error estimate.

---

## 7. Error bars: autocorrelation, binning, bootstrap

Monte Carlo samples are correlated in Markov time. A minimal, robust strategy for uncertainties is:
1. **Bin** consecutive measurements into blocks (bin size \(b\)).
2. Treat bin means as approximately independent.
3. **Bootstrap** resample the bins with replacement to propagate uncertainties through non-linear operations (effective mass, fits).

This produces stable, credible error bars without needing heavy time-series machinery.

---

## 8. Validation: the harmonic oscillator (\(\lambda=0\))

When \(\lambda=0\), the theory is exactly solvable and provides an excellent sanity check.

For the harmonic oscillator, the exact correlator is
\[
G_{\text{exact}}(\tau)
=
\frac{1}{2m\omega}\,
\frac{\cosh\!\left(\omega\left(\beta/2-\tau\right)\right)}{\sinh(\beta\omega/2)}.
\]

In particular,
\[
\langle x^2\rangle = G_{\text{exact}}(0)
=
\frac{1}{2m\omega}\coth(\beta\omega/2),
\]
and the gap is exactly
\[
\Delta = \omega.
\]

A well-tuned PIMC simulation should reproduce these within statistical errors for sufficiently small \(a\) and sufficiently large \(\beta\).

---

## 9. Systematic errors and practical parameter choices

PIMC results depend on:
- **Time discretization** \(a=\beta/N\): smaller \(a\) reduces Trotter error (\(\mathcal{O}(a^2)\) for symmetric splitting).
- **Finite \(\beta\)**: larger \(\beta\) improves ground-state dominance and clean gap extraction.
- **Finite statistics**: more measurements reduce variance, but autocorrelation must be handled.

A practical workflow is:
1. Choose \(\beta\) large enough to see a clear plateau in \(\Delta_{\text{eff}}\).
2. Increase \(N\) (decrease \(a\)) until results stabilize within errors.
3. Increase statistics until plots (correlator + effective mass) have visually stable error bars.

---

## 10. Summary

- Finite-temperature quantum mechanics can be rewritten as a Euclidean path integral.
- Discretization turns the problem into sampling a periodic chain \(\{x_i\}\) with weight \(e^{-S}\).
- PIMC uses MCMC (Metropolis) to sample paths.
- Translationally averaged correlators yield \(G(\tau)\), whose long-\(\tau\) behavior encodes the energy gap.
- Effective-mass plateaus and cosh fits provide a clean and standard gap extraction.
- Harmonic oscillator analytics provide a strong end-to-end validation benchmark.

This repository packages the full pipeline (sampling → correlator → uncertainties → gap extraction → plots) in a modular and reproducible way, driven entirely from a single notebook.