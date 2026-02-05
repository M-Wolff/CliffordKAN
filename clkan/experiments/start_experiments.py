import argparse
from pathlib import Path
from clkan.experiments.clifford_fit_func import synthetic_clifford
from clkan.experiments.fit_formulas import run_experiments_funcfitting,run_experiments_physics
from clkan.experiments.knot_dataset import run_experiments_knot
def comma_separated_ints(value):
    value = value[1:-1]  # clip away [ ... ] brackets
    return [int(x) for x in value.split(",")]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='?', default="all", type=str, help="Dataset to run on. Can be {holography,circuit,square,squaresquare,mult,sinus,knot}")
    parser.add_argument('--model', nargs='?', default="all", type=str, help="Model to run experiment on. Can be {pykan,fastkan,cvkan,cliffkan,all}")
    parser.add_argument("--task", type=str, default="funcfit", help="Task to run on. Can be one of {funcfit, physics,knot,highdims}") #, required=True)
    parser.add_argument("--clifford_grid", type=str, default=None, help="One of {full_grid, independant_grid, random_grid} for CliffordKAN")
    parser.add_argument("--clifford_rbf", type=str, default=None, help="Type of RBF calculation to use for CliffordKAN. One of {naive,cliffordspace}")
    parser.add_argument("--norm", type=str, default="nonorm", help="Type of Norm to use. Possible values (for cvkan and cliffkan): {batchnorm_comp-wise, batchnorm_node-wise, batchnorm_dim-wise}")
    parser.add_argument("--metric", type=comma_separated_ints, default=None, help="Type of Metric to use. Specify as [a,b,...] with brackets")
    parser.add_argument("--num_grids", type=int, default=8, help="Number of grid points to use")

    args = parser.parse_args()
    print("Running Function Fitting Eperiments for Dataset ", args.dataset, " and Model ", args.model, " and Task ", args.task)
    # check if clifford_grid and clifford_rbf exist, if model is cliffkan or all, otherwise throw error
    if args.model in ["cliffkan", "all"]:
        assert args.clifford_grid in ["full_grid","independant_grid", "random_grid"], "For Model 'cliffkan' parameter --clifford_grid must be set"
        assert args.clifford_rbf in ["naive","cliffordspace"], "For Model 'cliffkan' parameter --clifford_rbf must be set"
    extra_args = {"clifford_grid": args.clifford_grid, "clifford_rbf": args.clifford_rbf}
    extra_args["norm"] = args.norm
    extra_args["num_grids"] = args.num_grids
    if args.task == "funcfit":
        run_experiments_funcfitting(run_dataset=args.dataset, run_model=args.model, extra_args=extra_args)
    elif args.task == "physics":
        run_experiments_physics(run_dataset=args.dataset, run_model=args.model, extra_args=extra_args)
    elif args.task == "knot":
        run_experiments_knot(run_model=args.model, extra_args=extra_args)
    elif args.task == "highdims":
        assert args.metric is not None
        synthetic_clifford(name=args.dataset, metric=args.metric, extra_args=extra_args)
