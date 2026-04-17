{
#CliffordKAN
parallel --dry-run python3 cvkan/experiments/start_experiments.py --model cliffkan --task funcfit --dataset {1} --clifford_grid full_grid --clifford_rbf {2} --norm {3} --num_grids 8 ::: {square,squaresquare,mult,sinus} ::: {naive,cliffordspace} ::: {batchnorm_comp-wise,batchnorm_dim-wise,batchnorm_node-wise,nonorm}
parallel --dry-run python3 cvkan/experiments/start_experiments.py --model cliffkan --task funcfit --dataset {1} --clifford_grid independant_grid --clifford_rbf {2} --norm {3} --num_grids 8 ::: {square,squaresquare,mult,sinus} ::: {naive,cliffordspace} ::: {batchnorm_comp-wise,batchnorm_dim-wise,batchnorm_node-wise,nonorm}
parallel --dry-run python3 cvkan/experiments/start_experiments.py --model cliffkan --task funcfit --dataset {1} --clifford_grid random_grid --clifford_rbf {2} --norm {3} --num_grids {4} ::: {square,squaresquare,mult,sinus} ::: {naive,cliffordspace} ::: {batchnorm_comp-wise,batchnorm_dim-wise,batchnorm_node-wise,nonorm} ::: {2,3,4,5,6,7,8}
# CVKAN
parallel --dry-run python3 cvkan/experiments/start_experiments.py --model cvkan --task funcfit --dataset {1} --norm {2} ::: {square,squaresquare,mult,sinus} ::: {batchnorm_comp-wise,batchnorm_dim-wise,batchnorm_node-wise,nonorm}
} > all_commands.txt





{
##################################### Knot #####################################
#CliffordKAN Knot full
echo python3 cvkan/experiments/start_experiments.py --model cliffkan --task knot --dataset knot --clifford_grid full_grid --clifford_rbf cliffordspace --norm batchnorm_node-wise --num_grids 8
#CliffordKAN Knot independant_grid
echo python3 cvkan/experiments/start_experiments.py --model cliffkan --task knot --dataset knot --clifford_grid independant_grid --clifford_rbf cliffordspace --norm batchnorm_node-wise --num_grids 8
#CliffordKAN Knot random grid
parallel --dry-run python3 cvkan/experiments/start_experiments.py --model cliffkan --task knot --dataset knot --clifford_grid random_grid --clifford_rbf cliffordspace --norm batchnorm_node-wise --num_grids {1} ::: {2,3,4,5,6,7,8}
# CVKAN Knot
echo python3 cvkan/experiments/start_experiments.py --model cvkan --task knot --dataset knot --norm batchnorm_node-wise --num_grids 8
################################## Holography ##################################
#CliffordKAN Holo full
echo python3 cvkan/experiments/start_experiments.py --model cliffkan --task physics --dataset holography --clifford_grid full_grid --clifford_rbf cliffordspace --norm batchnorm_node-wise --num_grids 8
#CliffordKAN Holo independant_grid
echo python3 cvkan/experiments/start_experiments.py --model cliffkan --task physics --dataset holography --clifford_grid independant_grid --clifford_rbf cliffordspace --norm batchnorm_node-wise --num_grids 8
#CliffordKAN Holo random grid
parallel --dry-run python3 cvkan/experiments/start_experiments.py --model cliffkan --task physics --dataset holography --clifford_grid random_grid --clifford_rbf cliffordspace --norm batchnorm_node-wise --num_grids {1} ::: {2,3,4,5,6,7,8}
# CVKAN Holo
echo python3 cvkan/experiments/start_experiments.py --model cvkan --task physics --dataset holography --norm batchnorm_node-wise --num_grids 8
################################### Highdims ###################################
#CliffordKAN Highdims full
parallel --dry-run python3 cvkan/experiments/start_experiments.py --model cliffkan --task highdims --dataset {1} --clifford_grid full_grid --clifford_rbf cliffordspace --norm batchnorm_node-wise --num_grids 8 --metric {2} ::: {square,squaresquare,mult} ::: {"[1,1]","[-1,-1]","[1,-1]","[1,0]"}
#CliffordKAN Highdims independant_grid
parallel --dry-run python3 cvkan/experiments/start_experiments.py --model cliffkan --task highdims --dataset {1} --clifford_grid independant_grid --clifford_rbf cliffordspace --norm batchnorm_node-wise --num_grids 8 --metric {2} ::: {square,squaresquare,mult} ::: {"[1,1]","[-1,-1]","[1,-1]","[1,0]"}
#CliffordKAN Highdims random grid
parallel --dry-run python3 cvkan/experiments/start_experiments.py --model cliffkan --task highdims --dataset {1} --clifford_grid random_grid --clifford_rbf cliffordspace --norm batchnorm_node-wise --num_grids {3} --metric {2} ::: {square,squaresquare,mult} ::: {"[1,1]","[-1,-1]","[1,-1]","[1,0]"} ::: {2,3,4,5,6,7,8}
} > all_commands2.txt
