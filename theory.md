# Theory — Path Integral Monte Carlo (PIMC) for the 0+1D Anharmonic Oscillator

This document provides a self-contained and didactic overview of the theory behind the code in this repository: **imaginary-time path integrals**, their **lattice discretization**, and how **Path Integral Monte Carlo (PIMC)** estimates quantum observables by sampling closed Euclidean trajectories.

> **Idea in one sentence:** rewrite quantum expectation values at inverse temperature $\beta$ as averages over **periodic paths** weighted by $e^{-S_E}$, then sample those paths with MCMC.

* * *

## Contents

- [1. Model and targets](#1-model-and-targets)
- [2. From the trace to the Euclidean path integral](#2-from-the-trace-to-the-euclidean-path-integral)
- [3. Lattice discretization](#3-lattice-discretization)
- [4. PIMC sampling](#4-pimc-sampling)
- [5. Measuring the correlator efficiently](#5-measuring-the-correlator-efficiently)
- [6. Extracting the energy gap](#6-extracting-the-energy-gap)
- [7. Uncertainties and systematics](#7-uncertainties-and-systematics)
- [8. Harmonic oscillator as an exact benchmark](#8-harmonic-oscillator-as-an-exact-benchmark)

* * *

## 1. Model and targets

We study a single quantum degree of freedom $x$ with Hamiltonian

$$
H = \frac{p^2}{2m} + V(x),
\qquad
V(x)=\frac{1}{2}m\omega^2 x^2 + \lambda x^4,
\qquad \lambda \ge 0,
$$

using units $\hbar = 1$ (default). The finite-temperature partition function is

$$
Z(\beta) = \mathrm{Tr}\left(e^{-\beta H}\right),
\qquad \beta = \frac{1}{T}.
$$

A central observable is the Euclidean (imaginary-time) two-point function

$$
G(\tau)=\langle x(\tau)\,x(0)\rangle_\beta,
\qquad 0\le \tau \le \beta,
$$

where $x(\tau)=e^{\tau H}x e^{-\tau H}$ and

$$
G(\tau)
=
\frac{1}{Z}\,\mathrm{Tr}\!\left(e^{-(\beta-\tau)H}\,x\,e^{-\tau H}\,x\right).
$$

### Why this correlator matters

The long-$\tau$ decay of $G(\tau)$ encodes the **energy gaps** of the quantum spectrum. In particular, it allows us to estimate

$$
\Delta = E_1 - E_0,
$$

the gap between the first excited state and the ground state.

* * *

## 2. From the trace to the Euclidean path integral

Start from the trace definition

$$
Z = \mathrm{Tr}(e^{-\beta H}).
$$

Split the imaginary-time interval into $N$ slices of size $a=\beta/N$:

$$
e^{-\beta H} = \left(e^{-aH}\right)^N.
$$

Write $H = T + V$ with $T=p^2/(2m)$, and use a symmetric Trotter–Suzuki factorization

$$
e^{-aH} = e^{-\frac{a}{2}V}\,e^{-aT}\,e^{-\frac{a}{2}V} + \mathcal{O}(a^3),
$$

which leads to an overall discretization error $\mathcal{O}(a^2)$ for fixed $\beta$.

Insert $N$ resolutions of identity in the position basis,

$$
\mathbf{1}=\int_{-\infty}^{\infty} dx\,|x\rangle\langle x|.
$$

The kinetic matrix element is a Gaussian kernel:

$$
\langle x'|e^{-aT}|x\rangle
=
\sqrt{\frac{m}{2\pi a}}\,
\exp\!\left(-\frac{m}{2a}(x'-x)^2\right).
$$

Putting everything together yields a Euclidean path integral over **periodic trajectories** $x(\tau)$ with $x(0)=x(\beta)$:

$$
Z
=
\int \mathcal{D}x(\tau)\,e^{-S_E[x]},
$$

where the Euclidean action is

$$
S_E[x]=\int_{0}^{\beta}\left[\frac{m}{2}\dot{x}(\tau)^2 + V(x(\tau))\right]\,d\tau.
$$

> **Interpretation:** quantum statistics at temperature $T$ becomes a classical statistical mechanics problem in imaginary time, with weight $e^{-S_E}$.

* * *

## 3. Lattice discretization

Discretize $\tau_i = i a$ for $i=0,\dots,N-1$ and define $x_i=x(\tau_i)$, with periodic boundary conditions $x_N=x_0$.

Approximate the derivative by a finite difference:

$$
\dot{x}(\tau_i) \approx \frac{x_{i+1}-x_i}{a}.
$$

Then the discretized (lattice) action becomes

$$
S(\{x_i\})
=
\sum_{i=0}^{N-1}
\left[
\frac{m}{2a}(x_{i+1}-x_i)^2
+
a\,V(x_i)
\right].
$$

This is often described as a **ring polymer** of $N$ beads:
- the kinetic term couples nearest neighbors $(i,i+1)$,
- the potential term acts on each bead individually.

* * *

## 4. PIMC sampling

The lattice path integral defines a probability distribution over paths:

$$
\mathbb{P}(\{x_i\}) \propto e^{-S(\{x_i\})}.
$$

Therefore, any observable $\mathcal{O}$ can be estimated as a Monte Carlo average:

$$
\langle \mathcal{O} \rangle \approx \frac{1}{N_{\text{cfg}}}\sum_{c=1}^{N_{\text{cfg}}}\mathcal{O}(\{x_i\}_c),
$$

where $\{x_i\}_c$ are configurations sampled from $\mathbb{P}$.

### Local Metropolis updates (core idea)

A standard MCMC step proposes a local change at some site $i$:

$$
x_i' = x_i + \delta,\qquad \delta \sim \mathrm{Uniform}[-w,w].
$$

Accept with probability

$$
p_{\mathrm{acc}}=\min\left(1,e^{-\Delta S}\right),
\qquad
\Delta S = S(\ldots,x_i',\ldots)-S(\ldots,x_i,\ldots).
$$

**Key optimization:** $\Delta S$ is local. Only terms involving $x_{i-1},x_i,x_{i+1}$ change. Define

$$
S_i(x_i)=\frac{m}{2a}\Big[(x_i-x_{i-1})^2+(x_{i+1}-x_i)^2\Big]+aV(x_i),
$$

then

$$
\Delta S = S_i(x_i') - S_i(x_i).
$$

This locality makes the algorithm fast and easy to validate.

* * *

## 5. Measuring the correlator efficiently

Given a sampled path $\{x_i\}$, a translation-invariant estimator for the correlator is

$$
g_k(\{x_i\})
=
\frac{1}{N}\sum_{i=0}^{N-1} x_i\,x_{i+k \;\mathrm{mod}\; N},
\qquad k=0,\dots,N-1,
$$

and

$$
G_k \equiv G(\tau_k)\approx \langle g_k\rangle_{\mathrm{MC}},
\qquad \tau_k = k a.
$$

Computationally, $g_k$ is a periodic autocorrelation and can be computed in $O(N\log N)$ with FFT:

$$
c_k = \sum_{i=0}^{N-1} x_i x_{i+k}
=
\mathrm{IFFT}\!\left(\mathrm{FFT}(x)\cdot \mathrm{FFT}(x)^\*\right)_k,
\qquad
g_k = \frac{c_k}{N}.
$$

> FFT-based correlators are both faster and numerically cleaner than naive $O(N^2)$ summation.

* * *

## 6. Extracting the energy gap

### Spectral form and single-gap regime

Insert energy eigenstates into $G(\tau)$:

$$
G(\tau)
=
\frac{1}{Z}\sum_{m,n}
e^{-(\beta-\tau)E_m}\,e^{-\tau E_n}\,|\langle m|x|n\rangle|^2.
$$

For sufficiently large $\beta$, the ground state dominates and we get

$$
G(\tau)\approx \sum_n A_n\left(e^{-\Delta_n\tau}+e^{-\Delta_n(\beta-\tau)}\right),
\qquad \Delta_n = E_n - E_0.
$$

If a single term dominates,

$$
G(\tau) \simeq A\left(e^{-\Delta\tau}+e^{-\Delta(\beta-\tau)}\right)
= 2A\,e^{-\Delta\beta/2}\cosh\!\big(\Delta(\beta/2-\tau)\big).
$$

### Effective mass (effective gap)

Assuming the single-cosh form, one can derive the discrete identity

$$
\frac{G(\tau-a)+G(\tau+a)}{2G(\tau)} = \cosh(\Delta a),
$$

leading to the effective gap estimator

$$
\Delta_{\mathrm{eff}}(\tau)
=
\frac{1}{a}\operatorname{arcosh}\!\left(\frac{G(\tau-a)+G(\tau+a)}{2G(\tau)}\right).
$$

A **plateau** in $\Delta_{\mathrm{eff}}(\tau)$ (away from $\tau\approx 0$ and $\tau\approx \beta$) indicates a stable single-gap region.

### Direct cosh fit

Alternatively, fit $G(\tau)$ on a chosen window to

$$
G(\tau)=A\left(e^{-\Delta\tau}+e^{-\Delta(\beta-\tau)}\right),
$$

using uncertainties from binning/bootstrapping to obtain $\Delta \pm \sigma$.

* * *

## 7. Uncertainties and systematics

### Statistical uncertainties (correlated MCMC samples)

Markov chain samples are correlated in Monte Carlo time. A robust approach is:
1. **Bin** consecutive measurements (bin size $b$).
2. Treat bin means as approximately independent.
3. **Bootstrap** resample bins to propagate errors through non-linear steps (effective mass, fits).

### Systematic effects

PIMC results depend mainly on:
- **Time step** $a=\beta/N$: smaller $a$ reduces Trotter error (here $\mathcal{O}(a^2)$).
- **Inverse temperature** $\beta$: larger $\beta$ improves ground-state dominance and cleaner gap extraction.
- **Finite statistics**: more measurements reduce variance, but autocorrelation must be handled.

A practical workflow:
- choose $\beta$ large enough to see a visible effective-mass plateau,
- increase $N$ until results stabilize within errors,
- then increase statistics for smoother plots and tighter error bars.

* * *

## 8. Harmonic oscillator as an exact benchmark ($\lambda = 0$)

When $\lambda=0$, the correlator is known analytically:

$$
G_{\mathrm{exact}}(\tau)
=
\frac{1}{2m\omega}\,
\frac{\cosh\!\left(\omega\left(\beta/2-\tau\right)\right)}{\sinh(\beta\omega/2)}.
$$

In particular,

$$
\langle x^2\rangle = G_{\mathrm{exact}}(0)
=
\frac{1}{2m\omega}\coth(\beta\omega/2),
$$

and the energy gap is exactly

$$
\Delta = \omega.
$$

This repository uses the harmonic case as a clean end-to-end validation of:
- sampling,
- correlator estimation,
- gap extraction.

* * *

## Summary

- Finite-temperature quantum mechanics can be rewritten as a Euclidean path integral.
- Discretization produces a classical-looking ring-polymer action.
- PIMC samples paths with weight $e^{-S}$ using MCMC.
- The Euclidean correlator $G(\tau)$ encodes spectral gaps; effective-mass plateaus and cosh fits extract $\Delta$.
- Binning + bootstrap provide reliable error bars.
- The harmonic oscillator provides an exact benchmark for correctness.

If you want a quick hands-on entry point, run the notebook in `notebooks/` and inspect the correlator and effective-mass plots generated for $\lambda=0$ and $\lambda>0$.
