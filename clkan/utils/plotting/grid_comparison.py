import torch
from matplotlib import pyplot as plt
import numpy as np
from cvkan.models.CliffordKAN import create_gridnd_full
from icecream import ic

grid_points_full = 8

sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
random_grid = sobol.draw(grid_points_full**2) * 4 - 2

independent_grid = torch.rand(grid_points_full**2, 2) * 4 - 2
full_grid = create_gridnd_full(grid_min=-2, grid_max=2, num_dim=2, num_grids=grid_points_full)

fig, axs = plt.subplots(ncols=3)

axs[0].scatter(full_grid[:,:,0],full_grid[:,:,1])
axs[0].set_title("full grid")

axs[1].scatter(independent_grid[:,0], independent_grid[:,1])
axs[1].set_title("random grid")

axs[2].scatter(random_grid[:,0], random_grid[:,1])
axs[2].set_title("random grid (Sobol)")

for ax in axs:
    ax.set_aspect("equal")

plt.rcParams.update({'font.size': 16})
plt.tight_layout()
plt.savefig("grid-comparison.svg", bbox_inches="tight")
#plt.show()

