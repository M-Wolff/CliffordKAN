"""
File: fit_formulas.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Experiments for Function Fitting (physically meaningful circuit & holography
             as well as arbitrary simple formulae)
"""

from pathlib import Path

import torch
from cvkan.models.wrapper.CVKANWrapper import CVKANWrapper
from torch_ga.clifford.algebra import CliffordAlgebra

from clkan.experiments.run_crossval import run_crossval
from clkan.models.CliffordKAN import CliffordKAN
from clkan.utils.dataloading.create_complex_dataset import create_complex_dataset
from clkan.utils.dataloading.csv_dataloader import CSVDataset
from clkan.utils.loss_functions import MAE, MSE
from clkan.utils.norm_functions import Norms

mse_loss = MSE()
mae_loss = MAE()

loss_fns = dict()
loss_fns["mse"] = mse_loss
loss_fns["mae"] = mae_loss

_DATASET_SAVEDIR = Path(__file__).parent / "generated_datasets"
_DEVICE = "cuda"


def convert_complex_dataset_to_clifford(dataset: CSVDataset):
    """Converts a complex dataset to a dataset in clifford space"""
    if dataset.categorical_vars:
        raise NotImplementedError("complex dataset can't contain categorical vars!")
    num_vars_complex = len(dataset.input_varnames)
    num_outputs_complex = len(dataset.output_varnames)
    num_samples_train, num_samples_val, num_samples_test = (
        dataset.get_train_val_test_size()
    )
    num_dims = 2  # complex-valued has 2 dimensions

    train_input = torch.zeros(
        (num_samples_train, num_vars_complex, num_dims), dtype=torch.float32
    )
    val_input = torch.zeros(
        (num_samples_val, num_vars_complex, num_dims), dtype=torch.float32
    )
    test_input = torch.zeros(
        (num_samples_test, num_vars_complex, num_dims), dtype=torch.float32
    )
    train_label = torch.zeros(
        (num_samples_train, num_outputs_complex, num_dims), dtype=torch.float32
    )
    val_label = torch.zeros(
        (num_samples_val, num_outputs_complex, num_dims), dtype=torch.float32
    )
    test_label = torch.zeros(
        (num_samples_test, num_outputs_complex, num_dims), dtype=torch.float32
    )

    for in_feature in range(num_vars_complex):
        train_input[:, in_feature, 0] = dataset.data["train_input"][:, in_feature].real
        train_input[:, in_feature, 1] = dataset.data["train_input"][:, in_feature].imag

        if num_samples_test > 0:
            test_input[:, in_feature, 0] = dataset.data["test_input"][
                :, in_feature
            ].real
            test_input[:, in_feature, 1] = dataset.data["test_input"][
                :, in_feature
            ].imag
        if num_samples_val > 0:
            val_input[:, in_feature, 0] = dataset.data["val_input"][:, in_feature].real
            val_input[:, in_feature, 1] = dataset.data["val_input"][:, in_feature].imag

    for out_feature in range(num_outputs_complex):
        train_label[:, out_feature, 0] = dataset.data["train_label"][
            :, out_feature
        ].real
        train_label[:, out_feature, 1] = dataset.data["train_label"][
            :, out_feature
        ].imag

        if num_samples_test > 0:
            test_label[:, out_feature, 0] = dataset.data["test_label"][
                :, out_feature
            ].real
            test_label[:, out_feature, 1] = dataset.data["test_label"][
                :, out_feature
            ].imag
        if num_samples_val > 0:
            val_label[:, out_feature, 0] = dataset.data["val_label"][
                :, out_feature
            ].real
            val_label[:, out_feature, 1] = dataset.data["val_label"][
                :, out_feature
            ].imag

    # build a dictionary out of the now clifford-valued datapoints
    realdata_dict = dict()
    realdata_dict["train_input"] = train_input
    realdata_dict["train_label"] = train_label
    realdata_dict["val_input"] = val_input
    realdata_dict["val_label"] = val_label
    realdata_dict["test_input"] = test_input
    realdata_dict["test_label"] = test_label

    # create a CVDataset object from this dict
    dataset_clifford = CSVDataset(
        realdata_dict,
        input_vars=dataset.input_varnames,
        output_vars=dataset.output_varnames,
        categorical_vars=[],
    )
    return dataset_clifford


def convert_complex_dataset_to_real(dataset: CSVDataset):
    """Converts a complex dataset to a real-valued dataset by doubling input and output dimension (one real number for
    real and imaginary part each)"""
    if dataset.categorical_vars:
        raise NotImplementedError("complex dataset can't contain categorical vars!")
    num_vars_complex = len(dataset.input_varnames)
    num_vars_real = 2 * num_vars_complex
    num_outputs_complex = len(dataset.output_varnames)
    num_outputs_real = 2 * num_outputs_complex
    num_samples_train, num_samples_val, num_samples_test = (
        dataset.get_train_val_test_size()
    )

    train_input = torch.zeros((num_samples_train, num_vars_real), dtype=torch.float32)
    val_input = torch.zeros((num_samples_val, num_vars_real), dtype=torch.float32)
    test_input = torch.zeros((num_samples_test, num_vars_real), dtype=torch.float32)
    train_label = torch.zeros(
        (num_samples_train, num_outputs_real), dtype=torch.float32
    )
    val_label = torch.zeros((num_samples_val, num_outputs_real), dtype=torch.float32)
    test_label = torch.zeros((num_samples_test, num_outputs_real), dtype=torch.float32)

    # make sure varnames are also doubled for later plotting
    input_varnames_real = []
    output_varnames_real = []

    for in_feature in range(num_vars_complex):
        # fmt: off
        # formatter would auto-indent this block and make it unreadable.
        train_input[:, 2 * in_feature] = dataset.data["train_input"][:, in_feature].real
        train_input[:, 2 * in_feature + 1] = dataset.data["train_input"][
            :, in_feature
        ].imag

        if num_samples_test > 0:
            test_input[:, 2 * in_feature] = dataset.data["test_input"][:, in_feature].real
            test_input[:, 2 * in_feature + 1] = dataset.data["test_input"][:, in_feature].imag
        if num_samples_val > 0:
            val_input[:, 2 * in_feature] = dataset.data["val_input"][:, in_feature].real
            val_input[:, 2 * in_feature + 1] = dataset.data["val_input"][:, in_feature].imag
        # fmt: on

        input_varnames_real.append(dataset.input_varnames[in_feature] + ".real")
        input_varnames_real.append(dataset.input_varnames[in_feature] + ".imag")
    for out_feature in range(num_outputs_complex):
        # fmt: off
        # formatter would auto-indent this block and make it unreadable.
        train_label[:, 2 * out_feature] = dataset.data["train_label"][:, out_feature].real
        train_label[:, 2 * out_feature + 1] = dataset.data["train_label"][:, out_feature].imag

        if num_samples_test > 0:
            test_label[:, 2 * out_feature] = dataset.data["test_label"][:, out_feature].real
            test_label[:, 2 * out_feature + 1] = dataset.data["test_label"][:, out_feature].imag
        if num_samples_val > 0:
            val_label[:, 2 * out_feature] = dataset.data["val_label"][:, out_feature].real
            val_label[:, 2 * out_feature + 1] = dataset.data["val_label"][:, out_feature].imag
        # fmt: on

        output_varnames_real.append(dataset.output_varnames[out_feature] + ".real")
        output_varnames_real.append(dataset.output_varnames[out_feature] + ".imag")
    # build a dictionary out of the now real-valued datapoints
    realdata_dict = dict()
    realdata_dict["train_input"] = train_input
    realdata_dict["train_label"] = train_label
    realdata_dict["val_input"] = val_input
    realdata_dict["val_label"] = val_label
    realdata_dict["test_input"] = test_input
    realdata_dict["test_label"] = test_label

    # create a CVDataset object from this dict
    dataset_real = CSVDataset(
        realdata_dict,
        input_vars=input_varnames_real,
        output_vars=output_varnames_real,
        categorical_vars=[],
    )
    return dataset_real


def run_experiments_physics(run_dataset, run_model, extra_args):
    """Main method to run experiments on physically meaningul formula fitting (circuit & holography)"""
    _num_samples = 100000
    _dataset_name_suffix = "_100k"  # differentiate runs on 100k and 5k samples
    loss_fn_backprop = loss_fns["mse"]
    norm_to_use = Norms(extra_args["norm"])
    num_grids = extra_args["num_grids"]

    # holography formula
    holography = lambda x: torch.abs(x[:, [0]] + x[:, [1]]) ** 2 * x[:, [2]]
    # generate a fixed holdout test split; run_crossval will only split the training portion into k folds
    dataset_holography_c = create_complex_dataset(
        holography,
        ranges=[-2, 2],
        n_var=3,
        train_num=_num_samples,
        test_num=_num_samples,
        filepath_save=_DATASET_SAVEDIR / "ph_holo_c.pt",
    )
    dataset_holography_c = CSVDataset(
        dataset_holography_c,
        input_vars=["Er1", "E0", "Er2"],
        output_vars=["holography"],
        categorical_vars=[],
    )
    # create real-valued holography dataset from complex one
    dataset_holography_cliff = convert_complex_dataset_to_clifford(dataset_holography_c)

    # check which model and dataset to run
    run_models = [False] * 4
    if run_model == "all":
        run_models = [True] * 3
        raise NotImplementedError(
            "Please run experiments as single independent calls. Some global dictionaries are mutated for some models, which could cause chaos."
        )
    elif run_model == "pykan":
        raise NotImplementedError("PyKAN is not supported!")
        run_models[0] = True
    elif run_model == "fastkan":
        raise NotImplementedError("FastKAN is not supported!")
        run_models[1] = True
    elif run_model == "cvkan":
        run_models[2] = True
    elif run_model == "cliffkan":
        run_models[3] = True

    run_datasets = [False] * 2
    if run_dataset == "all":
        run_datasets = [True] * 2
    elif run_dataset == "holography":
        run_datasets[0] = True
    elif run_dataset == "circuit":
        raise NotImplementedError("Circuit dataset is not supported")
        run_datasets[1] = True

    if run_datasets[0]:  # holography
        for arch in [
            [3, 1],
            [3, 1, 1],
            [3, 3, 1],
            [3, 10, 1],
            [3, 10, 3, 1],
            [3, 10, 5, 3, 1],
        ]:
            if run_models[2]:
                cvkan = CVKANWrapper(
                    layers_hidden=arch, num_grids=num_grids, rho=1, use_norm=norm_to_use
                )
                run_crossval(
                    cvkan,
                    dataset_holography_c,
                    dataset_name="ph_holo_c" + _dataset_name_suffix,
                    loss_fn_backprop=loss_fn_backprop,
                    loss_fns=loss_fns,
                    batch_size=10000,
                    add_softmax_lastlayer=False,
                    epochs=5000,
                    convert_model_output_to_real=False,
                )
            if run_models[3]:
                # TODO metric should not be hardcoded here for later experiments
                algebra = CliffordAlgebra(metric=[-1], device=_DEVICE)
                cliffkan = CliffordKAN(
                    layers_hidden=arch,
                    algebra=algebra,
                    num_grids=num_grids,
                    rho=1,
                    use_norm=norm_to_use,
                    extra_args=extra_args,
                )
                loss_fns["mse"] = MSE(ga=cliffkan.algebra)
                loss_fns["mae"] = MAE(ga=cliffkan.algebra)
                loss_fn_backprop = loss_fns["mse"]
                run_crossval(
                    cliffkan,
                    dataset_holography_cliff,
                    dataset_name="ph_holo_cliff" + _dataset_name_suffix,
                    loss_fn_backprop=loss_fn_backprop,
                    loss_fns=loss_fns,
                    batch_size=10000,
                    add_softmax_lastlayer=False,
                    epochs=5000,
                    convert_model_output_to_real=False,
                )


def run_experiments_funcfitting(run_dataset, run_model, extra_args):
    """Main method to run experiments on arbitrary simple formula fitting (z^2, sin(z), z_1*z_2, (z_1^2 + z_2^2)^2)"""
    loss_fn_backprop = loss_fns["mse"]
    norm_to_use = Norms(extra_args["norm"])
    num_grids = extra_args["num_grids"]
    sq = lambda x: x[:, [0]] ** 2
    sqsq = lambda x: ((x[:, [0]]) ** 2 + x[:, [1]] ** 2) ** 2
    mult = lambda x: (x[:, [0]]) * x[:, [1]]
    sinus = lambda x: torch.sin(x[:, [0]])

    run_datasets = [False] * 4
    if run_dataset == "all":
        run_datasets = [True] * 4
    elif run_dataset == "square":
        run_datasets[0] = True
    elif run_dataset == "squaresquare":
        run_datasets[1] = True
    elif run_dataset == "mult":
        run_datasets[2] = True
    elif run_dataset == "sinus":
        run_datasets[3] = True

    run_models = [False] * 4
    if run_model == "all":
        raise NotImplementedError(
            "Please run experiments as single independent calls. Some global dictionaries are mutated for some models, which could cause chaos."
        )
        run_models = [True] * 4
    elif run_model == "pykan":
        raise NotImplementedError("PyKAN is not supported!")
        run_models[0] = True
    elif run_model == "fastkan":
        raise NotImplementedError("FastKAN is not supported!")
        run_models[1] = True
    elif run_model == "cvkan":
        run_models[2] = True
    elif run_model == "cliffkan":
        run_models[3] = True

    # generate a fixed holdout test split; run_crossval will only split the training portion into k folds

    dataset_sq_c = create_complex_dataset(
        sq,
        ranges=[-2, 2],
        n_var=1,
        train_num=5000,
        test_num=5000,
        filepath_save=_DATASET_SAVEDIR / "ff_square.pt",
    )
    dataset_sq_c = CSVDataset(
        dataset_sq_c, input_vars=["z"], output_vars=["z^2"], categorical_vars=[]
    )
    dataset_sq_cliff = convert_complex_dataset_to_clifford(dataset_sq_c)

    dataset_sqsq_c = create_complex_dataset(
        sqsq,
        ranges=[-2, 2],
        n_var=2,
        train_num=5000,
        test_num=5000,
        filepath_save=_DATASET_SAVEDIR / "ff_squaresquare.pt",
    )
    dataset_sqsq_c = CSVDataset(
        dataset_sqsq_c,
        input_vars=["z_1", "z_2"],
        output_vars=["(z_1^2 + z_2^2)^2"],
        categorical_vars=[],
    )
    dataset_sqsq_cliff = convert_complex_dataset_to_clifford(dataset_sqsq_c)

    dataset_mult_c = create_complex_dataset(
        mult,
        ranges=[-2, 2],
        n_var=2,
        train_num=5000,
        test_num=5000,
        filepath_save=_DATASET_SAVEDIR / "ff_mult.pt",
    )
    dataset_mult_c = CSVDataset(
        dataset_mult_c,
        input_vars=["z_1", "z_2"],
        output_vars=["z_1 * z_2"],
        categorical_vars=[],
    )
    dataset_mult_cliff = convert_complex_dataset_to_clifford(dataset_mult_c)

    dataset_sin_c = create_complex_dataset(
        sinus,
        ranges=[-2, 2],
        n_var=1,
        train_num=5000,
        test_num=5000,
        filepath_save=_DATASET_SAVEDIR / "ff_sin.pt",
    )
    dataset_sin_c = CSVDataset(
        dataset_sin_c, input_vars=["z"], output_vars=["sin(z)"], categorical_vars=[]
    )
    dataset_sin_cliff = convert_complex_dataset_to_clifford(dataset_sin_c)

    # Square Dataset = z**2
    if run_datasets[0]:
        if run_models[2]:
            cvkan = CVKANWrapper(
                layers_hidden=[1, 1], num_grids=num_grids, rho=1, use_norm=norm_to_use
            )
            run_crossval(
                cvkan,
                dataset_sq_c,
                dataset_name="ff_square",
                loss_fn_backprop=loss_fn_backprop,
                loss_fns=loss_fns,
                batch_size=500,
                add_softmax_lastlayer=False,
                epochs=5000,
                convert_model_output_to_real=False,
            )

            cvkan = CVKANWrapper(
                layers_hidden=[1, 2, 1],
                num_grids=num_grids,
                rho=1,
                use_norm=norm_to_use,
            )
            run_crossval(
                cvkan,
                dataset_sq_c,
                dataset_name="ff_square",
                loss_fn_backprop=loss_fn_backprop,
                loss_fns=loss_fns,
                batch_size=500,
                add_softmax_lastlayer=False,
                epochs=5000,
                convert_model_output_to_real=False,
            )
        if run_models[3]:
            # TODO metric should not be hardcoded here for later experiments
            algebra = CliffordAlgebra(metric=[-1], device=_DEVICE)
            cliffkan = CliffordKAN(
                layers_hidden=[1, 1],
                algebra=algebra,
                num_grids=num_grids,
                rho=1,
                use_norm=norm_to_use,
                extra_args=extra_args,
            )
            loss_fns["mse"] = MSE(ga=cliffkan.algebra)
            loss_fns["mae"] = MAE(ga=cliffkan.algebra)
            loss_fn_backprop = loss_fns["mse"]
            run_crossval(
                cliffkan,
                dataset_sq_cliff,
                dataset_name="ff_square",
                loss_fn_backprop=loss_fn_backprop,
                loss_fns=loss_fns,
                batch_size=500,
                add_softmax_lastlayer=False,
                epochs=5000,
                convert_model_output_to_real=False,
            )

            cliffkan = CliffordKAN(
                layers_hidden=[1, 2, 1],
                algebra=algebra,
                num_grids=num_grids,
                rho=1,
                use_norm=norm_to_use,
                extra_args=extra_args,
            )
            loss_fns["mse"] = MSE(ga=cliffkan.algebra)
            loss_fns["mae"] = MAE(ga=cliffkan.algebra)
            loss_fn_backprop = loss_fns["mse"]
            run_crossval(
                cliffkan,
                dataset_sq_cliff,
                dataset_name="ff_square",
                loss_fn_backprop=loss_fn_backprop,
                loss_fns=loss_fns,
                batch_size=500,
                add_softmax_lastlayer=False,
                epochs=5000,
                convert_model_output_to_real=False,
            )
    # Square Square Dataset = (z_1**2 + z_2**2)**2
    if run_datasets[1]:
        if run_models[2]:
            cvkan = CVKANWrapper(
                layers_hidden=[2, 1, 1],
                num_grids=num_grids,
                rho=1,
                use_norm=norm_to_use,
            )
            run_crossval(
                cvkan,
                dataset_sqsq_c,
                dataset_name="ff_squaresquare",
                loss_fn_backprop=loss_fn_backprop,
                loss_fns=loss_fns,
                batch_size=500,
                add_softmax_lastlayer=False,
                epochs=5000,
                convert_model_output_to_real=False,
            )

            cvkan = CVKANWrapper(
                layers_hidden=[2, 4, 2, 1],
                num_grids=num_grids,
                rho=1,
                use_norm=norm_to_use,
            )
            run_crossval(
                cvkan,
                dataset_sqsq_c,
                dataset_name="ff_squaresquare",
                loss_fn_backprop=loss_fn_backprop,
                loss_fns=loss_fns,
                batch_size=500,
                add_softmax_lastlayer=False,
                epochs=5000,
                convert_model_output_to_real=False,
            )
        if run_models[3]:
            # TODO metric should not be hardcoded here for later experiments
            algebra = CliffordAlgebra(metric=[-1], device=_DEVICE)
            cliffkan = CliffordKAN(
                layers_hidden=[2, 1, 1],
                algebra=algebra,
                num_grids=num_grids,
                rho=1,
                use_norm=norm_to_use,
                extra_args=extra_args,
            )
            loss_fns["mse"] = MSE(ga=cliffkan.algebra)
            loss_fns["mae"] = MAE(ga=cliffkan.algebra)
            loss_fn_backprop = loss_fns["mse"]
            run_crossval(
                cliffkan,
                dataset_sqsq_cliff,
                dataset_name="ff_squaresquare",
                loss_fn_backprop=loss_fn_backprop,
                loss_fns=loss_fns,
                batch_size=500,
                add_softmax_lastlayer=False,
                epochs=5000,
                convert_model_output_to_real=False,
            )

            cliffkan = CliffordKAN(
                layers_hidden=[2, 4, 2, 1],
                algebra=algebra,
                num_grids=num_grids,
                rho=1,
                use_norm=norm_to_use,
                extra_args=extra_args,
            )
            loss_fns["mse"] = MSE(ga=cliffkan.algebra)
            loss_fns["mae"] = MAE(ga=cliffkan.algebra)
            loss_fn_backprop = loss_fns["mse"]
            run_crossval(
                cliffkan,
                dataset_sqsq_cliff,
                dataset_name="ff_squaresquare",
                loss_fn_backprop=loss_fn_backprop,
                loss_fns=loss_fns,
                batch_size=500,
                add_softmax_lastlayer=False,
                epochs=5000,
                convert_model_output_to_real=False,
            )

    # Mult Dataset = z_1 * z_2
    if run_datasets[2]:
        if run_models[2]:
            cvkan = CVKANWrapper(
                layers_hidden=[2, 2, 1],
                num_grids=num_grids,
                rho=1,
                use_norm=norm_to_use,
            )
            run_crossval(
                cvkan,
                dataset_mult_c,
                dataset_name="ff_mult",
                loss_fn_backprop=loss_fn_backprop,
                loss_fns=loss_fns,
                batch_size=500,
                add_softmax_lastlayer=False,
                epochs=5000,
                convert_model_output_to_real=False,
            )

            cvkan = CVKANWrapper(
                layers_hidden=[2, 4, 2, 1],
                num_grids=num_grids,
                rho=1,
                use_norm=norm_to_use,
            )
            run_crossval(
                cvkan,
                dataset_mult_c,
                dataset_name="ff_mult",
                loss_fn_backprop=loss_fn_backprop,
                loss_fns=loss_fns,
                batch_size=500,
                add_softmax_lastlayer=False,
                epochs=5000,
                convert_model_output_to_real=False,
            )
        if run_models[3]:
            # TODO metric should not be hardcoded here for later experiments
            algebra = CliffordAlgebra(metric=[-1], device=_DEVICE)
            cliffkan = CliffordKAN(
                layers_hidden=[2, 2, 1],
                algebra=algebra,
                num_grids=num_grids,
                rho=1,
                use_norm=norm_to_use,
                extra_args=extra_args,
            )
            loss_fns["mse"] = MSE(ga=cliffkan.algebra)
            loss_fns["mae"] = MAE(ga=cliffkan.algebra)
            loss_fn_backprop = loss_fns["mse"]
            run_crossval(
                cliffkan,
                dataset_mult_cliff,
                dataset_name="ff_mult",
                loss_fn_backprop=loss_fn_backprop,
                loss_fns=loss_fns,
                batch_size=500,
                add_softmax_lastlayer=False,
                epochs=5000,
                convert_model_output_to_real=False,
            )

            cliffkan = CliffordKAN(
                layers_hidden=[2, 4, 2, 1],
                algebra=algebra,
                num_grids=num_grids,
                rho=1,
                use_norm=norm_to_use,
                extra_args=extra_args,
            )
            loss_fns["mse"] = MSE(ga=cliffkan.algebra)
            loss_fns["mae"] = MAE(ga=cliffkan.algebra)
            loss_fn_backprop = loss_fns["mse"]
            run_crossval(
                cliffkan,
                dataset_mult_cliff,
                dataset_name="ff_mult",
                loss_fn_backprop=loss_fn_backprop,
                loss_fns=loss_fns,
                batch_size=500,
                add_softmax_lastlayer=False,
                epochs=5000,
                convert_model_output_to_real=False,
            )

    # sin Dataset = sin(z)
    if run_datasets[3]:
        if run_models[2]:
            cvkan = CVKANWrapper(
                layers_hidden=[1, 1], num_grids=num_grids, rho=1, use_norm=norm_to_use
            )
            run_crossval(
                cvkan,
                dataset_sin_c,
                dataset_name="ff_sin",
                loss_fn_backprop=loss_fn_backprop,
                loss_fns=loss_fns,
                batch_size=500,
                add_softmax_lastlayer=False,
                epochs=5000,
                convert_model_output_to_real=False,
            )

            cvkan = CVKANWrapper(
                layers_hidden=[1, 2, 1],
                num_grids=num_grids,
                rho=1,
                use_norm=norm_to_use,
            )
            run_crossval(
                cvkan,
                dataset_sin_c,
                dataset_name="ff_sin",
                loss_fn_backprop=loss_fn_backprop,
                loss_fns=loss_fns,
                batch_size=500,
                add_softmax_lastlayer=False,
                epochs=5000,
                convert_model_output_to_real=False,
            )
        if run_models[3]:
            # TODO metric should not be hardcoded here for later experiments
            algebra = CliffordAlgebra(metric=[-1], device=_DEVICE)
            cliffkan = CliffordKAN(
                layers_hidden=[1, 1],
                algebra=algebra,
                num_grids=num_grids,
                rho=1,
                use_norm=norm_to_use,
                extra_args=extra_args,
            )
            loss_fns["mse"] = MSE(ga=cliffkan.algebra)
            loss_fns["mae"] = MAE(ga=cliffkan.algebra)
            loss_fn_backprop = loss_fns["mse"]
            run_crossval(
                cliffkan,
                dataset_sin_cliff,
                dataset_name="ff_sin",
                loss_fn_backprop=loss_fn_backprop,
                loss_fns=loss_fns,
                batch_size=500,
                add_softmax_lastlayer=False,
                epochs=5000,
                convert_model_output_to_real=False,
            )

            cliffkan = CliffordKAN(
                layers_hidden=[1, 2, 1],
                algebra=algebra,
                num_grids=num_grids,
                rho=1,
                use_norm=norm_to_use,
                extra_args=extra_args,
            )
            loss_fns["mse"] = MSE(ga=cliffkan.algebra)
            loss_fns["mae"] = MAE(ga=cliffkan.algebra)
            loss_fn_backprop = loss_fns["mse"]
            run_crossval(
                cliffkan,
                dataset_sin_cliff,
                dataset_name="ff_sin",
                loss_fn_backprop=loss_fn_backprop,
                loss_fns=loss_fns,
                batch_size=500,
                add_softmax_lastlayer=False,
                epochs=5000,
                convert_model_output_to_real=False,
            )
