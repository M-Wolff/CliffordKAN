# Clifford Kolmogorov-Arnold Networks

Authors: Matthias Wolff<sup>1</sup>, Francesco Alesiani<sup>2</sup>, Christof Duhme<sup>1</sup>, Xiaoyi Jiang<sup>1</sup>  
<sup>1</sup>: University of Münster, Department of Computer Science  
<sup>2</sup>: NEC Laboratories Europe GmbH

Paper: [Clifford Kolmogorov-Arnold Networks](https://arxiv.org/abs/2602.05977)

This repository contains the code for Clifford Kolmogorov-Arnold Networks (ClKAN). It is still work in progress while the code and documentation are being cleaned up.

## Abstract

We introduce Clifford Kolmogorov-Arnold Network (ClKAN), a flexible and efficient architecture for function approximation in arbitrary Clifford Algebra spaces. We propose the use of Randomized Quasi-Monte Carlo grid generation as a solution to the exponential scaling associated with higher-dimensional algebras. Our ClKAN also introduces new batch normalization strategies to deal with variable domain input. ClKAN finds application in scientific discovery and engineering, and is validated in synthetic and physics-inspired tasks.

## Table of Contents

- [Quick Start](#quick-start)
- [Demo Scripts](#demo-scripts)
- [Experiment Results](#experiment-results)
- [Experiment CLI](#experiment-cli)
- [Metric and Signature Notes](#metric-and-signature-notes)
- [Repository Structure](#repository-structure)
- [Related Repositories](#related-repositories)

## Quick Start

Install the full experiment dependencies from the lock-style requirements file:

```bash
pip install -r requirements.txt
```

The package metadata in `pyproject.toml` is intentionally lighter than the experiment environment and does not list every imported runtime dependency, notably `cvkan` and the editable `torch_ga` dependency.

## Demo Scripts

Two CPU demos are available under [`examples/`](examples/). They train directly with PyTorch on cached paper datasets and do not append to `clkan/experiments/results.json`.

Train a ClKAN on complex squaring using `Cl(0,1)`:

```bash
python examples/demo_complex_clkan.py
```

This loads [`clkan/experiments/generated_datasets/ff_square.pt`](clkan/experiments/generated_datasets/ff_square.pt) and uses paper-comparable defaults: `5000` cached train samples, `5000` test samples, a fold-0 style `4000/1000` train/validation split, `epochs=5000`, `batch_size=500`, `AdamW(lr=0.1)`, `ReduceLROnPlateau(factor=0.9, patience=20, threshold=0.001)`, `rho=1`, `full_grid`, `cliffordspace`, and `batchnorm_node-wise`.

Train a ClKAN on the cached high-dimensional Clifford square dataset with metric `[1, 1]`:

```bash
python examples/demo_clifford_cached.py
```

This loads [`clkan/experiments/generated_datasets/clifford_square_[1, 1]_80000.pt`](clkan/experiments/generated_datasets/clifford_square_%5B1,%201%5D_80000.pt) and uses paper-comparable defaults: `80000` cached train samples, `80000` test samples, a fold-0 style `64000/16000` train/validation split, `epochs=5000`, `batch_size=2000`, `AdamW(lr=0.1)`, `ReduceLROnPlateau(factor=0.9, patience=20, threshold=0.001)`, `rho=1`, `random_grid`, `cliffordspace`, and `batchnorm_node-wise`.

Both demos include comments on the `torch_ga` metric convention and define a small `metric2signature` inverse for the available `torch_ga.utils.signature2metric` helper. The repository currently has no cached quaternion/`Cl(3,0)` dataset; Hamilton quaternions are noted below for metric/signature reference only.

### Demo Usage

The default demo commands use paper-comparable settings and may run for a long time because they default to `5000` epochs:

```bash
python examples/demo_complex_clkan.py
python examples/demo_clifford_cached.py
```

For a quick smoke test, reduce epochs. The cached Clifford dataset is large, so use a large batch for the one-epoch smoke test if you only want to verify that the script runs:

```bash
python examples/demo_complex_clkan.py --epochs 5
python examples/demo_clifford_cached.py --epochs 1 --batch-size 80000
```

Useful flags:

- `--epochs`: training epochs; default `5000` to match the experiment scripts.
- `--batch-size`: default `500` for `ff_square` and `2000` for the cached Clifford dataset.
- `--lr`: default `0.1`, matching `clkan/train/train_loop.py`.
- `--num-grids`: default `8` for `ff_square` and `2` for the cached Clifford random-grid example.
- `--hidden`: `0` uses the shallow `[1, 1]` architecture; `2` uses `[1, 2, 1]`, matching the second square architecture used in the experiment code.
- `--dataset`: path to a compatible cached `.pt` dataset if you want to swap the default cache file.

Each demo prints the dataset, metric/signature, architecture, grid/RBF/norm settings, batch size, epoch count, LR, and periodic train/validation/test MSE. The scripts intentionally do not call `run_crossval.py`, so they do not append to `clkan/experiments/results.json`.

## Experiment Results

The main paper/experiment results are available in [`clkan/experiments/results.json`](clkan/experiments/results.json). The file is a JSON list of result dictionaries with model metadata, dataset metadata, losses per fold, aggregate means/stds, grid/RBF settings, normalization settings, and metric information.

Experiment runs through `run_crossval.py` append to this tracked JSON file under a file lock. If you only want a smoke test or demo, use the scripts in `examples/` instead of the experiment runner to avoid modifying the results file.

An older archived result log with duplicate entries is kept at [`clkan/experiments/results_withDuplicates.json`](clkan/experiments/results_withDuplicates.json).

## Experiment CLI

The experiment entrypoint is [`clkan/experiments/start_experiments.py`](clkan/experiments/start_experiments.py). Prefer module execution from the repository root:

```bash
python -m clkan.experiments.start_experiments --model cliffkan --task funcfit --dataset square --clifford_grid random_grid --clifford_rbf cliffordspace --norm batchnorm_node-wise --num_grids 2
```

Supported practical model values are `cvkan` and `cliffkan`. The CLI rejects `pykan`, `fastkan`, and `model=all` in the current code.

Important Clifford options:

- `--clifford_grid`: `full_grid` or `random_grid` are usable; the CLI also accepts the misspelled `independant_grid`, but `CliffordKANLayer` currently raises `NotImplementedError` for it.
- `--clifford_rbf`: `naive` or `cliffordspace`.
- `--norm`: values from `Norms`, for example `batchnorm_comp-wise`, `batchnorm_dim-wise`, `batchnorm_node-wise`, or `nonorm`.
- `--metric`: used by `--task highdims`, specified with brackets, for example `--metric [1,1]`.

Most experiment code defaults to CUDA. CPU-only runs may need small code changes or the demo scripts above.

## Metric and Signature Notes

`torch_ga` constructs `CliffordAlgebra(metric=...)` from a diagonal metric list where each entry is the square of one basis vector:

```python
from torch_ga.clifford.algebra import CliffordAlgebra
from torch_ga.utils import signature2metric

metric = signature2metric(3, 0, 0)  # [1, 1, 1]
algebra = CliffordAlgebra(metric=metric)
```

`torch_ga.utils.signature2metric(p, q=0, r=0)` converts a signature `(p, q, r)` to a metric with `p` positive, `q` negative, and `r` degenerate basis directions. There is no repo-provided inverse helper; for diagonal metrics, the inverse is simply:

```python
def metric2signature(metric):
    return (
        sum(v > 0 for v in metric),
        sum(v < 0 for v in metric),
        sum(v == 0 for v in metric),
    )
```

Useful examples:

- Complex numbers: `Cl(0,1)` has signature `(0, 1, 0)` and metric `[-1]`; elements are represented by `[scalar, e0]`.
- Hamilton quaternions: use the even subalgebra of `Cl(3,0)`, with signature `(3, 0, 0)` and metric `[1, 1, 1]`; the quaternion coordinates live in the even blades `{1, e01, e02, e12}` up to basis/sign convention.

## Repository Structure

- [`clkan/models/CliffordKAN.py`](clkan/models/CliffordKAN.py): core ClKAN layer and network implementation.
- [`clkan/train/train_loop.py`](clkan/train/train_loop.py): shared training loop used by experiment runners.
- [`clkan/experiments/start_experiments.py`](clkan/experiments/start_experiments.py): CLI dispatcher for function fitting, physics, knot, and high-dimensional experiments.
- [`clkan/experiments/run_crossval.py`](clkan/experiments/run_crossval.py): k-fold training and result logging to `results.json`.
- [`clkan/experiments/generated_datasets/`](clkan/experiments/generated_datasets/): cached generated `.pt` datasets reused by experiment scripts.
- [`clkan/utils/dataloading/`](clkan/utils/dataloading/): CSV and generated-dataset helpers.
- [`clkan/utils/plotting/`](clkan/utils/plotting/): scripts for plotting flattened experiment results and figures.
- [`figures/`](figures/): generated figures used by the paper/repository.
