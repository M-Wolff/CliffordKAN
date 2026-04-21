"""
This script was used for debugging and transferring the weights of a CVKAN to a CliffordKAN network to see if they behave the same way
with the same weights.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from cvkan.models.CVKAN import CVKAN
from icecream import ic
from torch_ga.clifford.algebra import CliffordAlgebra

from clkan.experiments.fit_formulas import convert_complex_dataset_to_clifford
from clkan.experiments.run_crossval import run_crossval
from clkan.models.CliffordKAN import CliffordKAN
from clkan.utils.dataloading.create_complex_dataset import create_complex_dataset
from clkan.utils.dataloading.crossval_splitter import split_crossval
from clkan.utils.dataloading.csv_dataloader import CSVDataset
from clkan.utils.loss_functions import MAE, MSE
from clkan.utils.norm_functions import Norms


# TODO also care about batchNorm weights...
def seed_all(seed: int, deterministic=False):
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # CUDA / CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Required for some CUDA ops to be deterministic (according to ChatGPT)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # PyTorch >= 1.8
        torch.use_deterministic_algorithms(True)


@torch.no_grad()
def map_weights_cvkan_to_cliffkan(cvkan: CVKAN, cliffkan: CliffordKAN):
    for layer_cv, layer_cliff in zip(cvkan.layers, cliffkan.layers):
        # layer_cv.realweights [I,O,G,G]
        # layer_cv.complexweights [I,O,G,G]
        i, o, g = layer_cv.realweights.shape[:-1]
        print(layer_cv)
        print(layer_cliff)
        # print(layer_cv.norm.weight)
        d = 2
        assert layer_cliff.weights.shape == (i, o, g, g, d), (
            "Networks don't seem to have matching dimensionality..."
        )
        assert type(layer_cv.norm) == type(layer_cliff.norm), (
            f"Networks need to use the same normalization scheme {type(layer_cv.norm)} != {type(layer_cliff.norm)}"
        )
        cliffkan_weights = torch.stack(
            [layer_cv.realweights, layer_cv.complexweights], dim=-1
        )
        # layer_cliff.weights [I,O,G,G,D]
        layer_cliff.weights.copy_(cliffkan_weights)
        # layer_cv.silu_bias [I,O] dtype complex
        # layer_cv.silu_weight [I,O] dtype complex
        cliffkan_silu_weights = torch.stack(
            [layer_cv.silu_weight.real, layer_cv.silu_weight.imag], dim=-1
        )
        cliffkan_silu_bias = torch.stack(
            [layer_cv.silu_bias.real, layer_cv.silu_bias.imag], dim=-1
        )
        # layer_cliff.silu_bias [I,O,D]
        # layer_cliff.silu_weight [I,O,D]
        layer_cliff.silu_weight.copy_(cliffkan_silu_weights)
        layer_cliff.silu_bias.copy_(cliffkan_silu_bias)


def train_and_compare(cliffkan: CliffordKAN, cvkan: CVKAN):
    k = 5
    _DATASET_SAVEDIR = Path("clkan/experiments/generated_datasets/")
    loss_fns = {"MSE": MSE(ga=cliffkan.algebra), "MAE": MAE(ga=cliffkan.algebra)}
    loss_fn_backprop = loss_fns["MSE"]
    sqsq = lambda x: ((x[:, [0]]) ** 2 + x[:, [1]] ** 2) ** 2
    dataset_sqsq_c = create_complex_dataset(
        sqsq,
        ranges=[-2, 2],
        n_var=2,
        train_num=5000,
        test_num=0,
        filepath_save=_DATASET_SAVEDIR / "ff_squaresquare.pt",
    )
    dataset_sqsq_c = CSVDataset(
        dataset_sqsq_c,
        input_vars=["z_1", "z_2"],
        output_vars=["(z_1^2 + z_2^2)^2"],
        categorical_vars=[],
    )
    dataset_sqsq_cliff = convert_complex_dataset_to_clifford(dataset_sqsq_c)
    datasets_cliff = split_crossval(
        dataset_sqsq_cliff, k=k
    )  # returns list of datasets (with different crossval splits each)
    datasets_c = split_crossval(
        dataset_sqsq_c, k=k
    )  # returns list of datasets (with different crossval splits each)
    # run training on current fold
    results_cliff = run_crossval(
        cliffkan,
        dataset_sqsq_cliff,
        dataset_name=" dummy_cliff",
        loss_fn_backprop=loss_fn_backprop,
        loss_fns=loss_fns,
        batch_size=500,
        logging_interval=100,
        add_softmax_lastlayer=False,
        epochs=1000,
        convert_model_output_to_real=False,
    )
    loss_fns = {"MSE": MSE(), "MAE": MAE()}
    loss_fn_backprop = loss_fns["MSE"]
    results_c = run_crossval(
        cvkan,
        dataset_sqsq_c,
        dataset_name="dummy_c",
        loss_fn_backprop=loss_fn_backprop,
        loss_fns=loss_fns,
        batch_size=500,
        logging_interval=100,
        add_softmax_lastlayer=False,
        epochs=1000,
        convert_model_output_to_real=False,
    )
    ic(results_cliff["train_losses"])
    ic(results_cliff["test_losses"])
    ic(results_c["train_losses"])
    ic(results_c["test_losses"])


if __name__ == "__main__":
    cvk = CVKAN(layers_hidden=[2, 4, 2, 1], num_grids=8, use_norm=Norms.NoNorm)
    algebra = CliffordAlgebra(metric=[-1])
    extra_args = {"clifford_rbf": "naive", "clifford_grid": "full_grid"}
    clk = CliffordKAN(
        layers_hidden=[2, 4, 2, 1],
        num_grids=8,
        algebra=algebra,
        extra_args=extra_args,
        use_norm=Norms.BatchNormNodewise,
    )
    # map_weights_cvkan_to_cliffkan(cvkan=cvk, cliffkan=clk)
    train_and_compare(cliffkan=clk, cvkan=cvk)
