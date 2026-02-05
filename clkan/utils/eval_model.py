"""
File: eval_model.py
Description: Methods to evaluate a given model on arbitrary loss functions as well as for plotting a confusion matrix
for Predictions / GT.
"""
import matplotlib
import numpy as np
import sklearn
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from clkan.models.CliffordKAN import CliffordKAN

def batched_forward(model, data, batch_size):
    predictions = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch_input = data[i:i+batch_size]
            pred = model(batch_input)
            predictions.append(pred)
    if len(predictions) > 0:
        return torch.cat(predictions, dim=0)
    else:
        return None




def eval_model(model, loss_fns, *, data_dict, add_softmax_lastlayer, batch_size, splits_to_eval, device="cuda"):
    """
    Evaluates a given model on given train and test data as well as ground-truth labels using a dictionary of loss functions
    :param model: model to evaluate
    :param loss_fns: dictionary of loss functions to evaluate the model on
    :param train_data: train input samples (no batching)
    :param test_data: test input samples (no batching)
    :param train_label: train ground truth labels (no batching)
    :param test_label: test ground truth labels (no batching)
    :param add_softmax_lastlayer: whether softmax should be applied after the last layer before evaluation of loss functions
    (CE loss named 'cross_entropy' will ignore this value and always receive raw logits)
    :param batch_size: Batch-Size used for forward pass
    :param splits_to_val: List which can contain multiple of the following: {'train', 'val', 'test'}
    :return: Tuple of two dictionaries (train_losses, test_losses) that each have the names of the loss functions
    evaluated as keys and the resulting value of each loss function as value.
    """
    # dictionaries to store train and test losses for all loss functions to evaluate
    losses = dict()
    model.eval()
    for split in splits_to_eval:
        losses[split] = dict()
        # create predictions for current split
        data = data_dict[f"{split}_input"]
        data = data.to(device)
        predictions = batched_forward(model, data, batch_size=batch_size)
        label = data_dict[f"{split}_label"]
        label = label.to(device)
        # maybe split is empty, then also make predictions empty with matching size
        if predictions is None:
            predictions = torch.zeros((0,) + label.shape[1:])
            predictions = predictions.to(device)
        # iterate over all loss functions
        for loss_fn_name, loss_fn in loss_fns.items():
            # Create clones of everything because some operations might change them in-place
            current_predictions = predictions.clone()
            current_label = label.clone()
            # if loss function is CE, accuracy, F1 or precision and model output is complex
            if loss_fn_name in ["cross_entropy", "accuracy", "f1", "precision"]:
                if torch.is_complex(current_predictions):
                    # only use real parts
                    current_predictions = predictions.real
                elif isinstance(model, CliffordKAN):
                    current_predictions = current_predictions[...,0]
            # if softmax should be applied (and loss function is not CE; CE requires raw logits)
            if add_softmax_lastlayer and loss_fn_name != "cross_entropy":
                # apply softmax
                current_predictions = torch.nn.functional.softmax(current_predictions, dim=1)
            # some loss functions require argmax and not softmax values
            if loss_fn_name in ["accuracy", "f1", "precision"]:
                current_label = torch.argmax(current_label, dim=1)
                current_predictions = torch.argmax(current_predictions, dim=1)
            # insert train and test losses into the dictionaries with key = name of loss function
            losses[split][loss_fn_name] = loss_fn(current_predictions, current_label)
    return losses

def plot_confusion_matrix(pred, gt, labelmapping=None):
    """
    Plot a confusion matrix based on Predictions and GT values
    :param pred: Predictions
    :param gt: Labels
    :param labelmapping: List of class names in order of class ids for axis descriptions
    """
    # set plot parameters & create figure
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 11}
    matplotlib.rc('font', **font)
    plt.figure(figsize=(16, 16))
    # potentially complex predictions are converted to real part only
    if pred.dtype == torch.complex64:
        pred = pred.real
    # if predictions are softmax values, convert to class ids (argmax)
    if len(pred.shape) > 1:
        pred = pred.argmax(axis=1)
    # if labels are One-Hot encoded values, convert to class ids (argmax)
    if len(gt.shape) > 1:
        gt = gt.argmax(axis=1)
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    # build confusion matrix
    cm = confusion_matrix(y_true=gt, y_pred=pred, normalize="true",labels=[i for i in range(len(labelmapping))] if labelmapping is not None else None)
    # round to 2 decimal places
    cm = np.round(cm, decimals=2)
    # display confusion matrix
    cmd = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=labelmapping)
    cmd.plot()
    plt.xlabel('Predicted label', fontsize=20)
    plt.ylabel('True label', fontsize=20)
    #plt.savefig('images/cvkan_knot_confusionmatrix.svg', transparent=True, bbox_inches='tight', pad_inches=0.5)
    plt.show()
