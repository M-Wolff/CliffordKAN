"""Reproduce a small ClKAN complex-square experiment on cached paper data.

Run from the repository root:
    python examples/demo_complex_clkan.py

Defaults mirror the `ff_square` ClKAN setup in `clkan/experiments/fit_formulas.py`:
cached 5000/5000 split, fold-0 style 4000/1000 train/val split, 5000 epochs,
batch size 500, AdamW lr 0.1, ReduceLROnPlateau scheduling, rho=1.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_ga.clifford.algebra import CliffordAlgebra
from torch_ga.utils import signature2metric

from clkan.models.CliffordKAN import CliffordKAN
from clkan.utils.norm_functions import Norms

DATASET_PATH = Path("clkan/experiments/generated_datasets/ff_square.pt")


def metric2signature(metric):
    """Inverse of torch_ga.utils.signature2metric for diagonal metrics."""
    return (
        sum(v > 0 for v in metric),
        sum(v < 0 for v in metric),
        sum(v == 0 for v in metric),
    )


def evaluate(model, algebra, x, y, batch_size):
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            pred = model(x[start : start + batch_size])
            loss = torch.mean(algebra.norm(pred - y[start : start + batch_size]) ** 2)
            losses.append(loss * len(pred))
    model.train()
    return torch.stack(losses).sum() / len(x)


def complex_dataset_to_clifford(data):
    result = {}
    for split in ("train", "val", "test"):
        x = data[f"{split}_input"]
        y = data[f"{split}_label"]
        if x.numel() == 0:
            result[f"{split}_input"] = torch.empty(0, 1, 2)
            result[f"{split}_label"] = torch.empty(0, 1, 2)
            continue
        result[f"{split}_input"] = torch.stack((x.real, x.imag), dim=-1).float()
        result[f"{split}_label"] = torch.stack((y.real, y.imag), dim=-1).float()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num-grids", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=0, choices=[0, 2])
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)

    # Complex numbers are represented as Cl(0,1): e0^2 = -1.
    # torch_ga's metric is the diagonal list of basis-vector squares.
    signature = (0, 1, 0)
    metric = signature2metric(*signature)  # [-1]
    assert metric == [-1]
    assert metric2signature(metric) == signature

    algebra = CliffordAlgebra(metric=metric, device=args.device)

    # Load the cached paper dataset for f(z)=z^2; do not generate new data.
    data = complex_dataset_to_clifford(torch.load(args.dataset, map_location=device))
    val_size = len(data["train_input"]) // 5
    val_x = data["train_input"][:val_size]
    val_y = data["train_label"][:val_size]
    train_x = data["train_input"][val_size:]
    train_y = data["train_label"][val_size:]
    test_x = data["test_input"]
    test_y = data["test_label"]
    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True
    )

    model = CliffordKAN(
        algebra=algebra,
        layers_hidden=[1, 1] if args.hidden == 0 else [1, args.hidden, 1],
        num_grids=args.num_grids,
        grid_mins=-2,
        grid_maxs=2,
        use_norm=Norms.BatchNormNodewise,
        extra_args={"clifford_grid": "full_grid", "clifford_rbf": "cliffordspace"},
    )
    model.to(device)

    print(
        "dataset=ff_square metric=[-1] signature=(0,1,0) "
        f"layers={model.layers_hidden} num_grids={args.num_grids} "
        "grid=full_grid rbf=cliffordspace norm=batchnorm_node-wise "
        f"batch_size={args.batch_size} epochs={args.epochs} lr={args.lr} "
        f"device={device}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=20, threshold=0.001
    )
    for epoch in range(args.epochs):
        train_mse = torch.tensor(float("nan"))
        for batch_x, batch_y in loader:
            pred = model(batch_x)
            train_mse = torch.mean(algebra.norm(pred - batch_y) ** 2)
            train_mse.backward()
            optimizer.step()
            optimizer.zero_grad()
        val_mse = evaluate(model, algebra, val_x, val_y, args.batch_size)
        scheduler.step(val_mse)
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            test_mse = evaluate(model, algebra, test_x, test_y, args.batch_size)
            print(
                f"epoch={epoch:04d} train_mse={train_mse.item():.6f} "
                f"val_mse={val_mse.item():.6f} test_mse={test_mse.item():.6f} "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )


if __name__ == "__main__":
    main()
