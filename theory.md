# Theory — Path Integral Monte Carlo (PIMC) for the 0+1D Anharmonic Oscillator

This document provides a compact but rigorous overview of the theory behind the codebase:
from the quantum Hamiltonian to the Euclidean path integral, and finally to **Path Integral Monte Carlo**
sampling and **gap extraction** from Euclidean correlators.

> **TL;DR**  
> We rewrite \(Z=\mathrm{Tr}(e^{-\beta H})\) as an integral over *closed paths* \(x(\tau)\) in imaginary time.  
> Discretization turns the quantum problem into sampling a classical ring of variables \(\{x_i\}\) with weight \(e^{-S}\).  
> From the sampled paths we estimate \(G(\tau)=\langle x(\tau)x(0)\rangle\) and extract the gap \(\Delta=E_1-E_0\).

* * *

## Table of contents

- [1. Model and goal](#1-model-and-goal)
- [2. Spectral meaning of the correlator](#2-spectral-meaning-of-the-correlator)
- [3. From \(Z\) to a Euclidean path integral](#3-from-z-to-a-euclidean-path-integral)
- [4. Lattice discretization](#4-lattice-discretization)
- [5. Path Integral Monte Carlo](#5-path-integral-monte-carlo)
- [6. Estimating \(G(\tau)\) efficiently (FFT)](#6-estimating-gtau-efficiently-fft)
- [7. Gap extraction](#7-gap-extraction)
- [8. Uncertainties and autocorrelation](#8-uncertainties-and-autocorrelation)
- [9. Harmonic oscillator benchmark (\(\lambda=0\))](#9-harmonic-oscillator-benchmark-lambda0)
- [10. Practical guidelines and systematic errors](#10-practical-guidelines-and-systematic-errors)

* * *

## 1. Model and goal

We consider a single quantum degree of freedom \(x\) with Hamiltonian
\[
H = \frac{p^2}{2m} + V(x),
\qquad
V(x)=\frac{1}{2}m\omega^2 x^2 + \lambda x^4,
\qquad \lambda \ge 0,
\]
in units where \(\hbar=1\). The default choice \(m=\omega=1\) yields an immediate reference value for the harmonic gap.

### What we want to compute

- The finite-temperature partition function:
  \[
  Z(\beta) = \mathrm{Tr}\left(e^{-\beta H}\right), \qquad \beta = 1/T.
  \]
- The Euclidean (imaginary-time) two-point function:
  \[
  G(\tau)=\langle x(\tau)\,x(0)\rangle_\beta,
  \qquad
  0\le \tau\le \beta,
  \]
  where \(x(\tau)=e^{\tau H}x e^{-\tau H}\).

This correlator is the key observable because its long-\(\tau\) behavior is controlled by the lowest energy gaps.

* * *

## 2. Spectral meaning of the correlator

Starting from the operator definition
\[
G(\tau)
=
\frac{1}{Z}\,\mathrm{Tr}\!\left(e^{-(\beta-\tau)H}\,x\,e^{-\tau H}\,x\right),
\]
insert a complete basis of energy eigenstates \(H|n\rangle = E_n|n\rangle\):
\[
G(\tau)
=
\frac{1}{Z}\sum_{m,n}
e^{-(\beta-\tau)E_m}\,e^{-\tau E_n}\,|\langle m|x|n\rangle|^2.
\]

For sufficiently large \(\beta\), the trace is dominated by the ground state and one obtains
\[
G(\tau)\approx \sum_n A_n\left(e^{-\Delta_n\tau}+e^{-\Delta_n(\beta-\tau)}\right),
\qquad
\Delta_n=E_n-E_0.
\]

If a single term dominates (very common in practice),
\[
G(\tau)\simeq A\left(e^{-\Delta\tau}+e^{-\Delta(\beta-\tau)}\right)
=2A\,e^{-\Delta\beta/2}\cosh\!\big(\Delta(\beta/2-\tau)\big).
\]

> **Interpretation**  
> Measuring \(G(\tau)\) gives direct access to the lowest energy gap \(\Delta=E_1-E_0\)
> via a **cosh decay** in imaginary time.

* * *

## 3. From \(Z\) to a Euclidean path integral

We start from
\[
Z = \mathrm{Tr}(e^{-\beta H}).
\]
Split imaginary time into \(N\) slices with step \(a=\beta/N\):
\[
e^{-\beta H}=\left(e^{-aH}\right)^N.
\]

### Trotter factorization

Write \(H=T+V\) with \(T=p^2/(2m)\). A symmetric (second-order) splitting is
\[
e^{-aH}\approx e^{-\frac{a}{2}V}\,e^{-aT}\,e^{-\frac{a}{2}V}+\mathcal O(a^3),
\]
so the total discretization error scales as \(\mathcal O(a^2)\) at fixed \(\beta\).

### Insert position resolutions

Insert \(N\) times the identity in the position basis:
\[
\mathbf{1}=\int_{-\infty}^{\infty} \mathrm dx \, |x\rangle\langle x|.
\]

The kinetic matrix element is Gaussian:
\[
\langle x'|e^{-aT}|x\rangle
=
\sqrt{\frac{m}{2\pi a}}\,
\exp\!\left(-\frac{m}{2a}(x'-x)^2\right).
\]

Putting everything together yields a **Euclidean path integral**
\[
Z=\int \mathcal D x(\tau)\,e^{-S_E[x]},
\qquad
S_E[x]=\int_0^\beta\left[\frac{m}{2}\dot x(\tau)^2 + V(x(\tau))\right]\mathrm d\tau,
\]
with periodic boundary conditions \(x(0)=x(\beta)\) (closed paths).

* * *

## 4. Lattice discretization

Discretize \(\tau_i=i a\) for \(i=0,\dots,N-1\) and set \(x_i=x(\tau_i)\) with periodicity \(x_N=x_0\).

Approximate the derivative:
\[
\dot x(\tau_i)\approx \frac{x_{i+1}-x_i}{a}.
\]

Then the discretized (lattice) Euclidean action is
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

> **Ring-polymer picture**  
> The quantum particle becomes a classical “ring” of \(N\) beads \(\{x_i\}\),
> coupled by nearest-neighbor springs (kinetic term) and subject to on-site potential \(V(x_i)\).

* * *

## 5. Path Integral Monte Carlo

### Sampling distribution

The discretized path integral defines a probability density over paths:
\[
\mathbb P(\{x_i\}) \propto e^{-S(\{x_i\})}.
\]
Therefore, any observable \(\mathcal O\) becomes a classical expectation value:
\[
\langle \mathcal O\rangle \approx \frac{1}{N_\text{samples}}\sum_{s=1}^{N_\text{samples}} \mathcal O(\{x_i\}^{(s)}).
\]

### Local Metropolis update (and why it is efficient)

A typical update proposes a local move \(x_i\to x_i'\) and accepts with probability
\[
p_\text{acc}=\min(1,e^{-\Delta S}).
\]

The crucial point is **locality**: changing only \(x_i\) affects only the terms involving
\(x_{i-1},x_i,x_{i+1}\). Define the local contribution
\[
S_i(x_i)=\frac{m}{2a}\Big[(x_i-x_{i-1})^2+(x_{i+1}-x_i)^2\Big]+aV(x_i),
\]
then
\[
\Delta S = S_i(x_i')-S_i(x_i).
\]

This makes MCMC steps cheap, debuggable, and unit-testable.

* * *

## 6. Estimating \(G(\tau)\) efficiently (FFT)

Given a sampled path \(x_0,\dots,x_{N-1}\), use translation invariance along the Euclidean circle:
\[
g_k
=
\frac{1}{N}\sum_{i=0}^{N-1} x_i\,x_{i+k\;\mathrm{mod}\;N},
\qquad k=0,\dots,N-1.
\]
Averaging \(g_k\) over sampled paths yields \(G_k\approx G(\tau_k)\) with \(\tau_k=k a\).

Computationally, \(g_k\) is a periodic autocorrelation and can be computed in \(O(N\log N)\) via FFT:
\[
c_k
=
\sum_{i=0}^{N-1} x_i x_{i+k}
=
\mathrm{IFFT}\!\left(\mathrm{FFT}(x)\cdot \mathrm{FFT}(x)^\*\right)_k,
\qquad
g_k=\frac{c_k}{N}.
\]

* * *

## 7. Gap extraction

### 7.1 Effective mass / effective gap

Assuming single-cosh dominance,
\[
G(\tau)\approx A\cosh(\Delta(\beta/2-\tau)).
\]
On the lattice one can derive
\[
\frac{G(\tau-a)+G(\tau+a)}{2G(\tau)}=\cosh(\Delta a),
\]
hence the **effective gap**
\[
\Delta_\text{eff}(\tau)
=
\frac{1}{a}\operatorname{arcosh}\!\left(\frac{G(\tau-a)+G(\tau+a)}{2G(\tau)}\right).
\]

A plateau in \(\Delta_\text{eff}(\tau)\) indicates a clean single-gap regime.

### 7.2 Direct cosh fit

Alternatively, fit the correlator in a suitable window to
\[
G(\tau)=A\left(e^{-\Delta\tau}+e^{-\Delta(\beta-\tau)}\right),
\]
and read off \(\Delta\) (with error bars estimated via bootstrap; see below).

* * *

## 8. Uncertainties and autocorrelation

Markov chains are correlated in Monte Carlo time. A robust and lightweight strategy is:

1. **Binning**: average consecutive measurements into blocks of size \(b\).
2. Treat binned means as approximately independent.
3. **Bootstrap** the bins to propagate uncertainties through non-linear steps (effective mass, fits).

This produces stable error bars without overcomplicating the pipeline.

* * *

## 9. Harmonic oscillator benchmark (\(\lambda=0\))

When \(\lambda=0\), everything is analytically solvable. The exact correlator is
\[
G_\text{exact}(\tau)
=
\frac{1}{2m\omega}\,
\frac{\cosh\!\left(\omega(\beta/2-\tau)\right)}{\sinh(\beta\omega/2)}.
\]

In particular,
\[
\langle x^2\rangle = G_\text{exact}(0)=\frac{1}{2m\omega}\coth(\beta\omega/2),
\qquad
\Delta=\omega.
\]

This benchmark provides an end-to-end validation of:
- the path sampling,
- the correlator estimator,
- the effective-mass extraction / fit.

* * *

## 10. Practical guidelines and systematic errors

PIMC accuracy is controlled by three main knobs:

- **Imaginary-time step** \(a=\beta/N\): smaller \(a\) reduces Trotter error (\(\mathcal O(a^2)\)).
- **Inverse temperature** \(\beta\): larger \(\beta\) improves ground-state dominance and stabilizes the gap plateau.
- **Statistics**: more (effectively independent) measurements reduce variance.

A practical workflow is:
1. Choose \(\beta\) large enough to see a clear effective-mass plateau.
2. Increase \(N\) until results stabilize within error bars (discretization control).
3. Increase statistics / bin size until uncertainties are visually stable.

> **In one sentence**  
> PIMC turns a quantum oscillator into a classical ring-polymer sampling problem, and the lowest energy gap is read from the exponential/cosh decay of Euclidean correlators.

* * *
