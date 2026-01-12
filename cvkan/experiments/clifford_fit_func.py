import torch
from pathlib import Path

from torch.utils import data
from torch_ga.clifford.algebra import CliffordAlgebra
import torch_ga

from cvkan.utils.dataloading.create_complex_dataset import create_complex_dataset
from cvkan.utils.dataloading.csv_dataloader import CSVDataset
from cvkan.experiments.fit_formulas import convert_complex_dataset_to_clifford,convert_complex_dataset_to_real
from cvkan.models.CliffordKAN import CliffordKAN
from cvkan.utils.loss_functions import MSE, MAE
from cvkan.experiments.run_crossval import run_crossval
from cvkan.utils.norm_functions import Norms

_DATASET_SAVEDIR = Path(__file__).parent / "generated_datasets"
_DEVICE = "cuda"

mse_loss = MSE()
mae_loss = MAE()

loss_fns = dict()
loss_fns["mse"] = mse_loss
loss_fns["mae"] = mae_loss


def synthetic_clifford(train_num=5000):
    # daraset cliff needs to be [B x vars x cliffdim]
    algebra = CliffordAlgebra(metric=[-1], device=_DEVICE)
    num_dim =  1 << algebra.num_bases
    # random samples in range [-2 +2] for each dimension
    x1 = torch.rand((train_num,1, num_dim)) * 4 - 2
    x2 = torch.rand((train_num,1, num_dim)) * 4 - 2
    x1 = x1.to(_DEVICE)
    x2 = x2.to(_DEVICE)
    # x1 and x2 have shape B x num_vars x num_dim
    x1x2 = torch.stack((x1,x2), dim=-1)  # shape B x 1 x num_dim x num_vars
    x1x2 = x1x2.squeeze(dim=1)  # squeeze 1 dimension away
    x1x2 = x1x2.permute((0,2,1))  # permute so num_vars is in middle
    # x1x2 has shape B x num_vars x num_dim

    #b = batch, v = number of variables in synthetic dataset (input), xyz = clifford dims
    x1_squared = torch.einsum("bvx,bvy,xyz->bvz", x1, x1, algebra.cayley)
    x2_squared = torch.einsum("bvx,bvy,xyz->bvz", x2, x2, algebra.cayley)
    sum_of_squares = x1_squared + x2_squared
    square_of_squares = torch.einsum("bvx,bvy,xyz->bvz", sum_of_squares, sum_of_squares, algebra.cayley)
    dataset_sq_cliff = {"train_input": x1, "train_label": x1_squared, "test_input": torch.zeros((0,1)), "test_label": torch.zeros((0,1))}
    dataset_sq_cliff = CSVDataset(dataset_sq_cliff, input_vars=["x1"], output_vars=["x1^2"], categorical_vars=[], train_test="100:0")

    dataset_sqsq_cliff = {"train_input":x1x2 , "train_label": square_of_squares, "test_input": torch.zeros((0,1)), "test_label": torch.zeros((0,1))}
    dataset_sqsq_cliff = CSVDataset(dataset_sqsq_cliff, input_vars=["x1"], output_vars=["x1^2"], categorical_vars=[], train_test="100:0")

    clifford_extra_args = {"clifford_rbf": "naive", "clifford_grid": "full_grid"}

    for arch in ([2,3,1], ):
            cliffkan = CliffordKAN(layers_hidden=arch, algebra=algebra, num_grids=8, rho=1, use_norm=Norms.BatchNormComponentWise, clifford_extra_args=clifford_extra_args)
            loss_fns["mse"] = MSE(ga=cliffkan.algebra)
            loss_fn_backprop = loss_fns["mse"]
            run_crossval(cliffkan, dataset_sqsq_cliff, dataset_name="clifford_square", loss_fn_backprop=loss_fn_backprop,loss_fns=loss_fns, batch_size=500, add_softmax_lastlayer=False, epochs=1000, convert_model_output_to_real=False)



if __name__ == "__main__":
    synthetic_clifford()
