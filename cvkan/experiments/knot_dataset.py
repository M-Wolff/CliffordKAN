"""
File: knot_dataset.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Experiments on the Knot Dataset (download dataset.csv from google storage
             gs://maths_conjectures/knot_theory/knot_theory_invariants.csv
             and also see their Jupyter Notebook:
             https://github.com/google-deepmind/mathematics_conjectures/blob/main/knot_theory.ipynb
             )
"""

from pathlib import Path
import pandas as pd
import torch
import torchmetrics
from icecream import ic

from ..experiments.run_crossval import run_crossval
from ..models.CVKAN import Norms
from ..models.FastKAN import FastKAN
from ..models.wrapper.CVKANWrapper import CVKANWrapper
from ..models.wrapper.PyKANWrapper import PyKANWrapper
from ..models.CliffordKAN import CliffordKAN
from ..utils.dataloading.csv_dataloader import CSVDataset
from ..experiments.fit_formulas import convert_complex_dataset_to_clifford
from ..utils.loss_functions import MAE,MSE

def load_knot_dataset_real(input_filename=Path("/home/m_wolf37/Datasets/knot_theory_invariants.csv"), train_test_split="70:30"):
    """
    Load Knot Dataset from .csv file
    :param input_filename: Path to dataset .csv file
    :param train_test_split: String specifying the share of train and test split as 'trainpercent:testpercent'
    :param complex_dataset: Whether return should be a complex-valued dataset (True) or real-valued (False)
    :return: Complex-valued or real-valued dataset
    """
    full_df = pd.read_csv(input_filename)
    full_df.drop(full_df.columns[0], axis=1, inplace=True)  # drop first column (id)
    # extract column names for input and output
    input_vars = list(full_df.columns[0:len(full_df.columns) - 1])
    output_var = [full_df.columns[len(full_df.columns) - 1]]
    # load dataset (real valued)
    dataset = CSVDataset(full_df, input_vars=input_vars, output_vars=output_var, categorical_vars=output_var, train_test=train_test_split)
    # normalize dataset (to our fixed grid range of [-2, 2])
    dataset.normalize()

    return dataset
def convert_knot_real_to_complex(dataset: CSVDataset):
    # otherwise: use this dataset to construct a complex-valued dataset
    num_train, num_test = dataset.data["train_input"].shape[0], dataset.data["test_input"].shape[0]
    num_complex_input_vars = len(dataset.input_varnames) - 2  # -2 because there are already 2 complex numbers in the dataset (split in Re and Im)
    dataset_complex = dict()
    dataset_complex["train_input"] = torch.zeros((num_train, num_complex_input_vars), dtype=torch.complex64)
    dataset_complex["test_input"] = torch.zeros((num_test, num_complex_input_vars), dtype=torch.complex64)

    # copy input features into complex dataset setting imaginary part to zero (if it doesnt exist)
    for idx_orig, idx_complex in [(0,0), (1,1), (4,3), (5,4), (6,5), (7,6), (10,8), (11,9), (12,10), (13,11), (14,12), (15,13), (16,14)]:
        print(idx_orig, idx_complex, dataset.data["train_input"].shape)
        dataset_complex["train_input"][:,idx_complex] = torch.complex(dataset.data["train_input"][:,idx_orig], torch.zeros_like(dataset.data["train_input"][:,idx_orig]))
        dataset_complex["test_input"][:,idx_complex] = torch.complex(dataset.data["test_input"][:,idx_orig], torch.zeros_like(dataset.data["test_input"][:,idx_orig]))

    # make complex number for "short geodesic"
    dataset_complex["train_input"][:, 2] = torch.complex(dataset.data["train_input"][:, 2], dataset.data["train_input"][:, 3])
    dataset_complex["test_input"][:, 2] = torch.complex(dataset.data["test_input"][:, 2], dataset.data["test_input"][:, 3])
    # make complex number for "meridinal translation" (in CSV imag and real are swapped...)
    dataset_complex["train_input"][:, 7] = torch.complex(dataset.data["train_input"][:, 9], dataset.data["train_input"][:, 8])
    dataset_complex["test_input"][:, 7] = torch.complex(dataset.data["test_input"][:, 9], dataset.data["test_input"][:, 8])

    # copy labels
    dataset_complex["train_label"] = dataset.data["train_label"]
    dataset_complex["test_label"] = dataset.data["test_label"]

    # change input vars
    input_vars = dataset.input_varnames
    input_vars[2] = "short_geodesic_complex"
    input_vars[8] = "meridinal_translation_complex"
    del input_vars[9]
    del input_vars[3]
    dataset_complex = CSVDataset(dataset_complex, input_vars=input_vars, output_vars=["c" + str(i) for i in range(14)], categorical_vars=[])

    dataset_complex.num_classes = dataset.num_classes
    # varnames for complex-valued dataset (abbreviated). Note that this list contains imag and real parts as
    # one single complex-valued variable
    input_vars_short_complex = ["adjoint_torsion_deg", "torsion_deg", "short_geodesic", "inject_rad", "chern_simons", "cusp_vol", "longit_translat", "merid_translat_c", "volume", "sym_0", "sym_d3", "sym_d4", "sym_d6", "sym_d8", "sym_Z/2+Z/2"]
    dataset_complex.input_varnames = input_vars_short_complex
    return dataset_complex
def convert_knot_complex_to_clifford(dataset: CSVDataset):
    clifford_dict = dict()
    clifford_dict["train_input"] = torch.zeros(size=dataset.data["train_input"].shape + (2,))
    clifford_dict["train_input"][...,0] = dataset.data["train_input"].real
    clifford_dict["train_input"][...,1] = dataset.data["train_input"].imag

    clifford_dict["test_input"] = torch.zeros(size=dataset.data["test_input"].shape + (2,))
    clifford_dict["test_input"][...,0] = dataset.data["test_input"].real
    clifford_dict["test_input"][...,1] = dataset.data["test_input"].imag

    clifford_dict["train_label"] = dataset.data["train_label"]
    clifford_dict["test_label"] = dataset.data["test_label"]
    dataset_clifford = CSVDataset(clifford_dict, input_vars=dataset.input_varnames, output_vars=["c" + str(i) for i in range(14)], categorical_vars=[])
    return dataset_clifford

def run_experiments_knot(run_model="all", clifford_extra_args=None):
    """Run the experiments on the Knot Dataset"""
    run_models = [False] * 4
    if run_model == "all":
        run_models = [True] * 4
    elif run_model == "pykan":
        run_models[0] = True
    elif run_model == "fastkan":
        run_models[1] = True
    elif run_model == "cvkan":
        run_models[2] = True
    elif run_model == "cliffkan":
        run_models[3] = True
    _DEVICE = torch.device("cuda")
    # load knot complex and real-valued with 100% train split
    # use 100% train split because run_crossval expects it this way and splits it into 5 non-overlapping folds
    knot_dataset_real = load_knot_dataset_real(train_test_split="100:0")
    knot_dataset_complex = convert_knot_real_to_complex(knot_dataset_real)
    knot_dataset_cliff = convert_knot_complex_to_clifford(knot_dataset_complex)
    in_features_complex = len(knot_dataset_complex.input_varnames)
    num_classes = len(knot_dataset_complex.output_varnames)

    crossentropy_loss = torch.nn.CrossEntropyLoss()

    loss_fns = dict()
    loss_fns["cross_entropy"] = crossentropy_loss
    loss_fns["accuracy"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(_DEVICE)
    for arch in [(in_features_complex, 1, num_classes), (in_features_complex, 2, num_classes)]:
            ################################# FastKAN #################################
            if run_models[1]:
                fastkan = FastKAN(layers_hidden=list(arch), num_grids=64, use_batchnorm=True, grid_mins=-2, grid_maxs=2)
                run_crossval(fastkan, knot_dataset_real, dataset_name="knot_r", loss_fn_backprop=crossentropy_loss, loss_fns=loss_fns, device=_DEVICE,
                             batch_size=10000, logging_interval=50, add_softmax_lastlayer=True, epochs=200, convert_model_output_to_real=False)
            ################################# CVKAN #################################
            if run_models[2]:
                cvkan = CVKANWrapper(layers_hidden=arch, num_grids=8, rho=1, use_norm=Norms.BatchNorm, grid_mins=-2, grid_maxs=2, csilu_type="complex_weight")
                run_crossval(cvkan, knot_dataset_complex, dataset_name="knot_c", loss_fn_backprop=crossentropy_loss,
                             loss_fns=loss_fns, device=_DEVICE,
                             batch_size=10000, logging_interval=50, add_softmax_lastlayer=True, epochs=200, convert_model_output_to_real=True)
            ################################# Clifford-KAN #################################
            if run_models[3]:
                # TODO metric should not be hardcoded here for later experiments
                cliffkan = CliffordKAN(layers_hidden=list(arch), metric=[-1], num_grids=8, rho=1, use_norm=Norms.BatchNormComponentWise, clifford_extra_args=clifford_extra_args)
                run_crossval(cliffkan, knot_dataset_cliff, dataset_name="knot_cliff", loss_fn_backprop=crossentropy_loss,loss_fns=loss_fns, batch_size=10000, logging_interval=50,add_softmax_lastlayer=True, epochs=1000, convert_model_output_to_real=True)


def train_knot_feature_subset():
    """Training on the Knot Dataset on the most important 3 or 7 features only
    as well as on the inverse set of every feature except the most important 3 or 7"""
    # use 100% train split because run_crossval expects it this way and splits it into 5 non-overlapping folds
    knot_dataset = load_knot_dataset(train_test_split="100:0", complex_dataset=True)
    # indices for features to train on
    indices_big = [1,2,3,5,6,7,8]
    indices_small = [2,6,7]
    indices_big_inverse = [0,4,9,10,11,12,13,14]
    indices_small_inverse = [0,1,3,4,5,8,9,10,11,12,13,14]

    num_classes = 14
    _DEVICE = torch.device("cuda")
    crossentropy_loss = torch.nn.CrossEntropyLoss()

    loss_fns = dict()
    loss_fns["cross_entropy"] = crossentropy_loss
    loss_fns["accuracy"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(_DEVICE)
    # train only on a specific subset of input-features
    for indices in [indices_small, indices_big, indices_small_inverse, indices_big_inverse]:
        cvkan = CVKANWrapper(layers_hidden=[len(indices), 1, 14], num_grids=8, rho=1, use_norm=Norms.BatchNorm, grid_mins=-2, grid_maxs=2, zsilu_type="complex_weight")
        train_input = knot_dataset.data["train_input"][:,indices]
        # reduce dataset to the specified subset of features
        knot_dataset_complex_reduced = dict()
        knot_dataset_complex_reduced["train_input"] = train_input
        knot_dataset_complex_reduced["train_label"] = knot_dataset.data["train_label"]
        knot_dataset_complex_reduced["test_label"] = knot_dataset.data["test_label"]
        knot_dataset_complex_reduced["test_input"] = knot_dataset.data["test_label"]
        knot_dataset_complex_reduced = CSVDataset(knot_dataset_complex_reduced, input_vars=[knot_dataset.input_varnames[i] for i in indices], output_vars=knot_dataset.output_varnames, categorical_vars=[])

        run_crossval(cvkan, knot_dataset_complex_reduced, dataset_name="knot_c_"+str(indices), loss_fn_backprop=crossentropy_loss,
                     loss_fns=loss_fns, device=_DEVICE,
                     batch_size=10000, logging_interval=50, add_softmax_lastlayer=True, epochs=200,
                     convert_model_output_to_real=True)

if __name__ == "__main__":
    #run_experiments()
    train_knot_feature_subset()
