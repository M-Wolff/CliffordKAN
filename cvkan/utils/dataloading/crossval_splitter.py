"""
File: crossval_splitter.py
Author: Matthias Wolff, Florian Eilers, Xiaoyi Jiang
Description: Convert dataset with 100% train split to k datasets with non-overlapping test-splits of size (1/k)
using k-fold cross-validation
"""
import torch
from cvkan.utils.dataloading.csv_dataloader import CSVDataset

def split_crossval(dataset: CSVDataset, k=5):
    """

    :param dataset: CSVDataset dataset with train data, possibly test data (will not change) and **empty** val split
    :param k: number of folds to do
    :return: list of k datasets, each element representing one fold.
    """
    num_train, num_val, num_test = dataset.get_train_val_test_size()
    assert num_val == 0, "Val Split needs to be empty for crossval splitter"
    crossval_datasets = []
    # loop over all possible split points (0,1,2,...,k-1)
    for split in range(k):
        # calculate val start and end indices
        val_start_index = split * (num_train//k)
        val_end_index = (split + 1) * (num_train//k)
        # create new dataset dictionary
        data = dict()
        # copy test set from original dataset for current fold
        data["test_input"] = dataset.data["test_input"]
        data["test_label"] = dataset.data["test_label"]
        # copy val set from original train dataset for current fold based on previously calculated range of indices
        data["val_input"] = dataset.data["train_input"][val_start_index:val_end_index]
        data["val_label"] = dataset.data["train_label"][val_start_index:val_end_index]
        # if val split is surrounded by train data
        if val_start_index > 0 and val_end_index < num_train:
            # copy data before and after the test split into the train split
            data["train_input"] = torch.cat((dataset.data["train_input"][0:val_start_index], dataset.data["train_input"][val_end_index:num_train, :]))
            data["train_label"] = torch.cat((dataset.data["train_label"][0:val_start_index], dataset.data["train_label"][val_end_index:num_train, :]))
        # if val split is the last split
        elif val_start_index > 0 and val_end_index == num_train:
            # copy everything before val split to train split
            data["train_input"] = dataset.data["train_input"][0:val_start_index, :]
            data["train_label"] = dataset.data["train_label"][0:val_start_index, :]
        # if val split is the first split
        elif val_start_index == 0 and val_end_index != num_train:
            # copy everything after test split to train split
            data["train_input"] = dataset.data["train_input"][val_end_index:, :]
            data["train_label"] = dataset.data["train_label"][val_end_index:, :]
        # create a CSVDataset object out of the constructed dictionary, copying varnames
        ds = CSVDataset(data, input_vars=dataset.input_varnames, output_vars=dataset.output_varnames, categorical_vars=dataset.categorical_vars)
        # set attribute num_classes the same as in the original datasets, if exists
        if hasattr(dataset, "num_classes"):
            ds.num_classes = dataset.num_classes
        # append current fold's dataset to the list of datasets
        crossval_datasets.append(ds)
    # return list of k datasets
    return crossval_datasets
