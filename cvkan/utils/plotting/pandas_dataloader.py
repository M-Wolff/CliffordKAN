import json
from pathlib import Path
from numpy import exp
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


def load_json_make_pandas(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)
    print(len(data))
    results_dict = dict()
    flattened_keys = ["dataset_name", "model_name", "num_grids", "layers", "use_norm", "num_trainable_params", "extra_args.clifford_rbf", "extra_args.clifford_grid", "test_losses.mean.mse", "test_losses.std.mse", "test_losses.mean.mae", "test_losses.std.mae"]

    for k in flattened_keys:
        results_dict[k] = []
    
    for experiment in data:
        experiment_dict_flat = flatten_dict(experiment) 
        for k in flattened_keys:
            if k not in experiment_dict_flat:
                print(f"Skipping key {k} for model {experiment_dict_flat['model_name']}")
                val_to_append = None
            elif k == "use_norm":
                val_to_append = experiment_dict_flat[k][0]
            elif k == "layers":
                val_to_append = ",".join(map(str, experiment_dict_flat[k]))
            else:
                val_to_append = experiment_dict_flat[k]
            results_dict[k].append(val_to_append)
    results_df = pd.DataFrame(results_dict)
    return results_df



if __name__ == "__main__":
    res_df = load_json_make_pandas(Path("/home/m_wolf37/Sciebo/Doktorand/Workspace/Alesiani-KANs/CliffordKAN/results_backup.json"))
    print(res_df)
