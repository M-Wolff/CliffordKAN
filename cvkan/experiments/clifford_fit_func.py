import torch
from pathlib import Path

from torch_ga.clifford.algebra import CliffordAlgebra

from cvkan.utils.dataloading.create_complex_dataset import load_dataset, move_dataset_to_device, save_dataset
from cvkan.utils.dataloading.csv_dataloader import CSVDataset
from cvkan.models.CliffordKAN import CliffordKAN
from cvkan.utils.loss_functions import MSE, MAE
from cvkan.experiments.run_crossval import run_crossval
from cvkan.utils.norm_functions import Norms

_DATASET_SAVEDIR = Path(__file__).parent / "generated_datasets"
_DEVICE = torch.device("cuda")

def create_clifford_dataset(name, metric, train_num, test_num): 
    filename = f"clifford_{name}_{metric}_{train_num}.pt"
    filepath = _DATASET_SAVEDIR / filename
    if filepath.exists():
        print(f"Using cached dataset {filepath}")
        dataset_dict = load_dataset(filepath)
        return dataset_dict
    
    algebra = CliffordAlgebra(metric=metric)
    num_dim =  1 << algebra.num_bases
    if name == "square":
        # random samples in range [-2 +2] for each dimension
        x1 = torch.rand((train_num+test_num,1, num_dim)) * 4 - 2
        x1_squared = torch.einsum("bvx,bvy,xyz->bvz", x1, x1, algebra.cayley)
        dataset_dict = {"train_input": x1[0:train_num, ...], "train_label": x1_squared[0:train_num, ...], "val_input": torch.tensor([]), "val_label": torch.tensor([]), "test_input": x1[train_num:,...], "test_label": x1_squared[train_num:,...]}
    elif name == "squaresquare":
        x1 = torch.rand((train_num+test_num,1, num_dim)) * 4 - 2
        x2 = torch.rand((train_num+test_num,1, num_dim)) * 4 - 2
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
        dataset_dict = {"train_input": x1x2[0:train_num,...], "train_label": square_of_squares[0:train_num,...], "val_input": torch.tensor([]), "val_label": torch.tensor([]), "test_input": x1x2[train_num:,...], "test_label": square_of_squares[train_num:,...]}
    elif name == "mult":
        x1 = torch.rand((train_num+test_num,1, num_dim)) * 4 - 2
        x2 = torch.rand((train_num+test_num,1, num_dim)) * 4 - 2
        x1x2 = torch.stack((x1,x2), dim=-1)  # shape B x 1 x num_dim x num_vars
        x1x2 = x1x2.squeeze(dim=1)  # squeeze 1 dimension away
        x1x2 = x1x2.permute((0,2,1))  # permute so num_vars is in middle
        x1_mult_x2 = torch.einsum("bvx,bvy,xyz->bvz", x1, x2, algebra.cayley)
        dataset_dict = {"train_input": x1x2[0:train_num,...], "train_label": x1_mult_x2[0:train_num,...], "val_input": torch.tensor([]), "val_label": torch.tensor([]), "test_input": x1x2[train_num:,...], "test_label": x1_mult_x2[train_num:,...]}
    else:
        raise NotImplementedError(f"Clifford Dataset for name {name} is not defined!")
    if not filepath.exists():
        save_dataset(dataset_dict=dataset_dict, filepath=filepath)
    return dataset_dict


def synthetic_clifford(name, metric, extra_args, train_num=80000, test_num=80000):
    # 80k points because it was 5k for 2d problems (cv), they have a volume of 16 ([-2,2]^2 = 4^2 = 16)
    # now we need [-2,2]^4 = 256, so 5k/16 := xk / 256 --> x=80
    # daraset cliff needs to be [B x vars x cliffdim]
    norm_to_use = Norms(extra_args["norm"])
    algebra = CliffordAlgebra(metric=metric)
    num_dim =  1 << algebra.num_bases
    num_grids = extra_args["num_grids"]
    input_vars = {"square": ["x1"], "squaresquare": ["x1", "x2"], "mult": ["x1", "x2"]}
    output_vars = {"square": ["x1^2"], "squaresquare": ["(x1^2+x2^2)^2"], "mult": ["x1 * x2"]}
    dataset_dict = create_clifford_dataset(name=name, metric=metric, train_num=train_num, test_num=test_num)
    dataset_cliff = CSVDataset(dataset_dict, input_vars=input_vars[name], output_vars=output_vars[name], categorical_vars=[])
    archs = {"square": [[1,1],[1,2,1]], "squaresquare": [[2,1,1], [2,4,2,1]], "mult": [[2,2,1], [2,4,2,1]]}
    for arch in archs[name]:
        algebra = CliffordAlgebra(metric=metric, device=_DEVICE)
        cliffkan = CliffordKAN(layers_hidden=arch, algebra=algebra, num_grids=num_grids, rho=1, use_norm=norm_to_use, extra_args=extra_args)
        mse_loss = MSE(cliffkan.algebra)
        mae_loss = MAE(cliffkan.algebra)
        loss_fns = dict()
        loss_fns["mse"] = mse_loss
        loss_fns["mae"] = mae_loss
        loss_fn_backprop = loss_fns["mse"]
        run_crossval(cliffkan, dataset_cliff, dataset_name=f"clifford_{name}_{metric}_{train_num}", loss_fn_backprop=loss_fn_backprop,loss_fns=loss_fns, batch_size=2000, add_softmax_lastlayer=False, epochs=5000, convert_model_output_to_real=False)

