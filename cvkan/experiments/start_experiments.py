import argparse
from pathlib import Path
from cvkan.experiments.fit_formulas import run_experiments_funcfitting,run_experiments_physics
from cvkan.experiments.knot_dataset import run_experiments_knot
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='?', default="all", type=str, help="Dataset to run on. Can be {holography,circuit,square,squaresquare,mult,sinus,knot}")
    parser.add_argument('--model', nargs='?', default="all", type=str, help="Model to run experiment on. Can be {pykan,fastkan,cvkan,cliffkan,all}")
    parser.add_argument("--task", type=str, default="funcfit", help="Task to run on. Can be one of {funcfit, physics,knot}") #, required=True)
    parser.add_argument("--clifford_grid", type=str, default=None, help="One of {full_grid, independant_grid} for CliffordKAN")
    parser.add_argument("--clifford_rbf", type=str, default=None, help="Type of RBF calculation to use for CliffordKAN. One of {naive,cliffordspace}")

    args = parser.parse_args()
    print("Running Function Fitting Eperiments for Dataset ", args.dataset, " and Model ", args.model, " and Task ", args.task)
    # check if clifford_grid and clifford_rbf exist, if model is cliffkan or all, otherwise throw error
    if args.model in ["cliffkan", "all"]:
        assert args.clifford_grid in ["full_grid","independant_grid"], "For Model 'cliffkan' parameter --clifford_grid must be set"
        assert args.clifford_rbf in ["naive","cliffordspace"], "For Model 'cliffkan' parameter --clifford_rbf must be set"
    clifford_extra_args = {"clifford_grid": args.clifford_grid, "clifford_rbf": args.clifford_rbf}
    if args.task == "funcfit":
        run_experiments_funcfitting(run_dataset=args.dataset, run_model=args.model, clifford_extra_args=clifford_extra_args)
    elif args.task == "physics":
        run_experiments_physics(run_dataset=args.dataset, run_model=args.model, clifford_extra_args=clifford_extra_args)
    elif args.task == "knot":
        run_experiments_knot(run_model=args.model, clifford_extra_args=clifford_extra_args)
