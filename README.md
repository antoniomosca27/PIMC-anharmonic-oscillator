# PIMC-anharmonic-oscillator

[![CI](https://github.com/antoniomosca27/PIMC-anharmonic-oscillator/actions/workflows/ci.yml/badge.svg)](https://github.com/antoniomosca27/PIMC-anharmonic-oscillator/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository provides a Python implementation of Path Integral Monte Carlo (PIMC) for the 0+1D quartic $\phi^4$ anharmonic oscillator. It includes a typed library API, a single end-to-end pipeline notebook, deterministic tests, and CI quality gates.

* * *

## Theory

A theoretical treatment of the problem and a detailed explanation of the PIMC method used in this project are provided in `theory.pdf`.

* * *

## Scope

- Simulate Euclidean-time lattice paths for the quartic anharmonic oscillator.
- Estimate correlators, effective masses, and energy gaps from Monte Carlo samples.
- Persist run artifacts under reproducible `logs/` and `reports/` run directories.
- Maintain code quality via `ruff` and `pytest` in GitHub Actions.

* * *

## Installation

```bash
git clone https://github.com/antoniomosca27/PIMC-anharmonic-oscillator.git
cd PIMC-anharmonic-oscillator
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

To run `notebooks/PIMC_anharmonic_oscillator_pipeline.ipynb`, install
`requirements.txt` first (it includes `jupyter` and `ipykernel`).

For development tools:

```bash
python -m pip install -e .[dev]
```

* * *

## Quickstart

Run the user-facing pipeline notebook:

```bash
jupyter notebook notebooks/PIMC_anharmonic_oscillator_pipeline.ipynb
```

The notebook is parameter-driven: edit only the **Parameters** cell, then run all cells.
By default it runs a fast but stable HMC scan with `lambda = [0.0, 0.25, 0.5, 1.0]`.
The `lambda=0.0` point is the built-in harmonic sanity check:
- sampled correlator points should overlap the analytic harmonic curve within uncertainties,
- the extracted harmonic gap should be close to `OMEGA`,
- the effective-mass plot should show a visible plateau near `OMEGA`.
- invalid effective-mass points are masked as `NaN` by default
  (`EFFECTIVE_MASS_INVALID_AS_NAN = True`) to avoid misleading near-zero artifacts.

Minimal library usage:

```python
from pathlib import Path

from pimc_oscillator import (
    LatticeParams,
    PotentialParams,
    RunConfig,
    SamplerParams,
    run_experiment,
)

config = RunConfig(
    potential=PotentialParams(m=1.0, omega=1.0, lam=0.25),
    lattice=LatticeParams(beta=4.5, n_slices=128),
    sampler=SamplerParams(
        method="hmc",
        n_therm=1_800,
        n_sweeps=5_200,
        measure_every=4,
        proposal_width=1.0,
        tune=True,
        tune_interval=25,
        target_accept=0.75,
        hmc_step_size=0.043,
        hmc_n_leapfrog=18,
        hmc_mass=1.0,
        seed=123,
    ),
    run_name="readme-example",
)

result = run_experiment(config, out_root=Path("."))
print(result["gap_bootstrap"].delta_mean)
print(result["paths"]["reports_dir"])
```

* * *

## Repository Layout

```text
PIMC-anharmonic-oscillator/
├── .github/workflows/ci.yml
├── notebooks/
│   └── PIMC_anharmonic_oscillator_pipeline.ipynb
├── scripts/
│   ├── check_notebooks_clean.py
│   ├── clean_artifacts.sh
│   └── strip_notebook_outputs.py
├── src/
│   ├── __init__.py
│   ├── action.py
│   ├── analysis.py
│   ├── config.py
│   ├── experiment.py
│   ├── io.py
│   ├── observables.py
│   ├── plotting.py
│   ├── potential.py
│   ├── sampler.py
│   └── utils.py
├── tests/
│   ├── conftest.py
│   ├── test_action.py
│   ├── test_analysis.py
│   ├── test_observables.py
│   ├── test_sampler.py
│   └── test_smoke.py
├── data/.gitkeep
├── logs/.gitkeep
├── reports/.gitkeep
├── pyproject.toml
├── requirements.txt
├── theory.pdf
└── README.md
```

* * *

## Runtime Outputs

- `data/`: optional numeric artifacts and cached intermediates.
- `logs/`: per-run configs and sampled chain files.
- `reports/`: per-run analysis JSON and generated figures.

Directory anchors (`.gitkeep`) are tracked; generated artifacts are ignored.

* * *

## Reproducibility

- Deterministic RNG control through `SamplerParams.seed`.
- Immutable run configuration via `RunConfig` dataclasses.
- CI runs lint and tests on Python 3.10, 3.11, and 3.12.

* * *

## Developer

Run local checks:

```bash
python scripts/check_notebooks_clean.py notebooks/PIMC_anharmonic_oscillator_pipeline.ipynb
ruff check .
pytest
```

Strip notebook outputs before committing:

```bash
python scripts/strip_notebook_outputs.py notebooks/PIMC_anharmonic_oscillator_pipeline.ipynb
```

Clean local artifacts:

```bash
bash scripts/clean_artifacts.sh
```

* * *

## Authors

- Antonio Mosca

* * *

## License

MIT
