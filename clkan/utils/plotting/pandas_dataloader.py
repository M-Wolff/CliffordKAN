import json
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List

def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = "."
) -> Dict[str, Any]:
    """ Written by ChatGPT
    Recursively flattens a nested dictionary.
    Example:
        {"a": {"b": 1}} -> {"a.b": 1}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_json_make_pandas(json_path: Path, filter_option=None):
    if filter_option is None:
        filter_func = lambda x: True
    elif filter_option == "highdims":
        filter_func = lambda x: x["dataset_name"].startswith("clifford_")
    elif filter_option == "funcfit":
        filter_func = lambda x: x["dataset_name"].startswith("ff_")
    else:
        raise NotImplementedError()
    with open(json_path, "r") as f:
        data = json.load(f)
    print(len(data))
    results_dict = dict()
    flattened_keys = ["dataset_name", "model_name", "num_grids", "layers", "use_norm", "num_trainable_params", "extra_args.clifford_rbf", "extra_args.clifford_grid", "test_losses.mean.mse", "test_losses.std.mse", "test_losses.mean.mae", "test_losses.std.mae","metric"]

    for k in flattened_keys:
        results_dict[k] = []
    
    for experiment in data:
        experiment_dict_flat = flatten_dict(experiment) 
        if not filter_func(experiment_dict_flat):
            continue
        for k in flattened_keys:
            if k not in experiment_dict_flat:
                print(f"Skipping key {k} for model {experiment_dict_flat['model_name']}")
                val_to_append = None
            elif k == "use_norm":
                if experiment_dict_flat[k][0] != "nonorm":
                    val_to_append = experiment_dict_flat[k][0].split("_")[1]
                else:
                    val_to_append = experiment_dict_flat[k][0]
            elif k == "layers":
                val_to_append = ",".join(map(str, experiment_dict_flat[k]))
            elif k == "metric":
                replacement_dict = {"[1, 1]": "Cl(2,0,0)", "[1, 0]": "Cl(1,0,1)", "[-1, -1]": "Cl(0,2,0)", "[1, -1]": "Cl(1,1,0)", "[-1]": "CV", "None": None}
                val_to_append = replacement_dict[str(experiment_dict_flat[k])]
            elif k == "dataset_name" and "[" in experiment_dict_flat[k]:
                val_to_append = "_".join(experiment_dict_flat[k].split("_")[1:2])
            else:
                val_to_append = experiment_dict_flat[k]
            results_dict[k].append(val_to_append)
    results_df = pd.DataFrame(results_dict)
    return results_df
