# in 3D: metric=[3,0,0] and PGA [3,0,1] and CGA in 3d: [4,1,0]
# in 2D: [2,0,0],  PGA [2,0,1], CGA [3,1,0]
"""
File: CliffordKAN.py
Author: Matthias Wolff, Francesco Alesiani, Xiaoyi Jiang
Description: Clifford-KAN Model definition
"""
from typing import List
import torch
from torch_ga.clifford import CliffordAlgebra
from ..utils.norm_functions import ComponentwiseBatchNorm1d, NodewiseBatchNorm1d, DimensionwiseBatchNorm1d, Norms

def create_gridnd_full(grid_min, grid_max, num_grids, num_dim):
    axes = [
        torch.linspace(grid_min, grid_max, num_grids)
        for _ in range(num_dim)
    ]

    # meshgrid creates num_dim tensors of shape [num_grids, ..., num_grids]
    grids = torch.meshgrid(*axes, indexing="ij")

    # stack coordinates in last dimension
    grid = torch.stack(grids, dim=-1)
    return grid

def create_gridnd_independent(num_dim, grid_min, grid_max, num_grids):
    # create grid points n-D array
    # for complex-valued we have num_dim=2 and thus [2 x num_grids] as grid shape. In CVKAN we had [num_grids x num_grids] but this probably doesn't scale well for clifford kan and higher dims? So now we treat each component independently.
    # create grid (linspace from grid_min to grid_max consisting of num_grids grid points)
    # then add one more dimension (not just a view!) to have a grid per dimension of CA / GA
    grid = torch.linspace(grid_min, grid_max, num_grids).unsqueeze(-1).repeat(1, num_dim)
    return grid

def create_gridnd_random(num_dim, grid_min, grid_max, num_gridpoints_total):
    grid_range = grid_max - grid_min
    sobol = torch.quasirandom.SobolEngine(dimension=num_dim, scramble=True)
    grid = sobol.draw(num_gridpoints_total) * grid_range + grid_min
    #grid = torch.rand((num_gridpoints_total, num_dim)) * grid_range + grid_min
    return grid

def create_grid2d_full(grid_min, grid_max, num_grids):
    grid_real = torch.linspace(grid_min, grid_max, num_grids)
    grid_real = grid_real.unsqueeze(1).expand(num_grids, num_grids)
    grid_imag = torch.linspace(grid_min, grid_max, num_grids)
    grid_imag = grid_imag.unsqueeze(0).expand(num_grids, num_grids)
    grid = torch.stack((grid_real, grid_imag), dim=2)
    return grid
    """ CVKAN approach     
        # create grid points 2D array
        real = torch.linspace(grid_min, grid_max, num_grids)
        real = real.unsqueeze(1).expand(num_grids, num_grids)

        imag = torch.linspace(grid_min, grid_max, num_grids)
        imag = imag.unsqueeze(0).expand(num_grids, num_grids)
        # make it complex-valued from real and imaginary parts
        grid = torch.complex(real, imag)
    """
    

class CliffordKANLayer(torch.nn.Module):
    def __init__(self, algebra: CliffordAlgebra, input_dim: int, output_dim: int, num_grids: int = 8, grid_min = -2, grid_max = 2,
                 rho=1, use_norm=Norms.BatchNormComponentWise, silu_type="componentwise", extra_args=None):

        """
        :param algebra: the Geometric Algebra to use
        :param input_dim: input dimension size of Layer (Layer Width)
        :param output_dim: output dimension size of Layer (next Layer's Width)
        :param num_grids: number of grid points ***per dimension***
        :param grid_min: left limit of grid
        :param grid_max: right limit of grid
        :param rho: rho for use in RBF (default rho=1)
        :param use_norm: which Normalization scheme to use
        :param silu_type: the kind of SiLU to use
        :param extra_args: additional args dictionary containing 'metric', 'clifford_rbf' and 'clifford_grid' with respective values
        """
        super().__init__()
        assert extra_args is not None
        self.extra_args = extra_args
        self.algebra = algebra
        self.cayley = self.algebra.cayley  # we need a copy to move cayley to gpu/cpu accordingly in to(...) method
        self.num_grids_per_dim = num_grids  # number of gridpoints for each dimension
        # num_dim is 2^num_bases
        self.num_dim = 1 << self.algebra.num_bases
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.silu_type = silu_type
        self.use_norm = use_norm
        # initialize Norm instance corresponding to self.use_norm
        if self.use_norm == Norms.BatchNormComponentWise:
            # component-wise BatchNorm
            self.norm = ComponentwiseBatchNorm1d(num_dimensions=self.num_dim, input_length=self.output_dim)
        elif self.use_norm == Norms.BatchNormDimensionwise:
            # dimension-wise BatchNorm
            self.norm = DimensionwiseBatchNorm1d(num_dimensions=self.num_dim, input_length=self.output_dim)
        elif self.use_norm == Norms.BatchNormNodewise:
            # node-wise BatchNorm
            self.norm = NodewiseBatchNorm1d(num_dimensions=self.num_dim, input_length=self.output_dim)
        elif self.use_norm == Norms.NoNorm:
            self.norm = lambda x: x  # no-op
        else:
            # everything else is not yet implemented
            print(self.use_norm)
            raise NotImplementedError()

        # grid is a non-trainable Parameter
        if self.extra_args["clifford_grid"] == "full_grid":
            self.num_grids = num_grids
            grid = create_gridnd_full(grid_min, grid_max, num_grids, self.num_dim)
        elif self.extra_args["clifford_grid"] == "independant_grid":
            # TODO: fix independent grid
            raise NotImplementedError("Independent grid was not implemented correctly. Fix is WIP")
            self.num_grids = num_grids ** self.num_dim
            grid = create_gridnd_independent(self.num_dim,grid_min, grid_max, self.num_grids)
        elif self.extra_args["clifford_grid"] == "random_grid":
            self.num_grids = num_grids ** self.num_dim
            grid = create_gridnd_random(num_dim=self.num_dim, grid_min=grid_min, grid_max=grid_max, num_gridpoints_total=self.num_grids)
        else:
            raise NotImplementedError(f"Grid-Type {self.extra_args['clifford_grid']} is not supported!")

        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        
        self.rho = rho
        # weights for each RBF centered around the grid points
        self.random_init_weights()
        # initialize CSiLU weight to use based on selected csilu_type
        if self.silu_type == "componentwise":
            self.silu_weight = torch.ones(size=(self.input_dim, self.output_dim, self.num_dim), dtype=torch.float32)
            self.silu_weight = torch.nn.Parameter(self.silu_weight, requires_grad=True)
            self.silu = torch.nn.SiLU()
        else:
            raise NotImplementedError()
        # add bias to each SiLU activation
        self.silu_bias = torch.nn.Parameter(torch.zeros(size=(input_dim, output_dim, self.num_dim), dtype=torch.float32), requires_grad=True)
    def random_init_weights(self):
        if hasattr(self, "weights"):
            orig_device = self.weights.device
        else:
            orig_device = None
        # weights for each RBF centered around the grid points
        if self.extra_args["clifford_grid"] == "full_grid":
            self.weights = torch.nn.Parameter(torch.randn(size=((self.input_dim, self.output_dim,) + self.num_dim*(self.num_grids,)+ (self.num_dim,))), requires_grad=True)
            #self.weights = torch.nn.Parameter(torch.zeros(size=((input_dim, output_dim,) + self.num_dim*(self.num_grids,)+ (self.num_dim,))), requires_grad=True)
            # self.weights now is [I,O, (num_grids)**num_dims, num_dims)
        elif self.extra_args["clifford_grid"] in ["independant_grid", "random_grid"]:
            self.weights = torch.nn.Parameter(torch.randn(size=(self.input_dim, self.output_dim, self.num_grids,self.num_dim)), requires_grad=True)  # for clifford kan independant dims
            # TODO think about this and maybe fix?
            # self.weight [I, O, Num_Dim, Num_Grids, Num_Dim)

        if self.extra_args["clifford_grid"] == "random_grid":  # maybe also resample grid
            grid = create_gridnd_random(num_dim=self.num_dim, grid_min=self.grid_min, grid_max=self.grid_max, num_gridpoints_total=self.num_grids)
            self.grid = torch.nn.Parameter(grid, requires_grad=False)
        if orig_device is not None:
            self.to(orig_device)

    def forward(self, x):
        # x has shape BATCH x Input-Dim x self.num_dim
        assert len(x.shape) == 3 and x.shape[1] == self.input_dim and x.shape[2] == self.num_dim, f"Wrong Input Dimension! Got {x.shape} for Layer with Dimensions[{self.input_dim}, {self.output_dim}] and num_dims = {self.num_dim}"
        # grid is [num_grids x num_dim]
        # apply RBF on x (centered around each grid point)
        if self.extra_args["clifford_grid"] == "full_grid":
            #x = x.unsqueeze(-1).unsqueeze(-1).expand(x.shape + (self.num_grids, self.num_grids))
            #x = x.unsqueeze(-1).unsqueeze(-1).expand(x.shape + self.num_dim*(self.num_grids,))
            #new_shape = (x.shape[0:-1] + self.num_dim * (self.num_grids,) + (self.num_dim, ))
            #for _ in range(self.num_dim):
            #    x = x.unsqueeze(-2)
            #x = x.expand(*new_shape)
            x_view = x.view(*x.shape[:-1], *([1] * self.num_dim), x.shape[-1])
            #x = torch.permute(x, dims=(0,1,3,4,2))  # switch num_dim and num_grids dimensions (last 2 dimensions)
        elif self.extra_args["clifford_grid"] in ["independant_grid", "random_grid"]:
            #x = x.unsqueeze(-1).expand(x.shape + (self.num_grids,))
            #x = torch.permute(x, dims=(0,1,3,2))  # switch num_dim and num_grids dimensions (last 2 dimensions)
            x_view = x.view(*x.shape[:-1], 1, x.shape[-1])
        else:
            raise NotImplementedError()
        # i and o are input and output indices within layer layer_idx and layer_ix+1
        # d is dimension and g is grid
        if self.extra_args["clifford_rbf"]=="naive":
            result = torch.exp(-(self.algebra.norm(x_view - self.grid)) ** 2 / self.rho)
            result = result.squeeze(dim=-1)
            if self.extra_args["clifford_grid"] == "full_grid":
                result = torch.einsum("bi...,io...x->biox", result, self.weights)  # clifford dependant dimensions
                #slice_dims = [slice(None)] * 2
                #slice_dims += [0] * (x.ndim - 3)
                #slice_dims += [slice(None)]
                #x = x[tuple(slice_dims)]
            elif self.extra_args["clifford_grid"] in ["independant_grid", "random_grid"]:
                result = torch.einsum("big,iogx->biox", result, self.weights)
                #x = x[:,:,0,:]

        elif self.extra_args["clifford_rbf"]=="cliffordspace":
            result = torch.exp(-(self.algebra.norm(x_view - self.grid)) ** 2 / self.rho)
            #result = torch.stack([result, torch.zeros_like(result)], dim=-1)  # after norm and exp last dimension is always 1, fill up to make it clifford-valued again
            result = result.unsqueeze(dim=-1)  # algebra.embed(...) requires last dim to already exist
            result = self.algebra.embed(result, tensor_index=torch.tensor([0]))
            if self.extra_args["clifford_grid"] == "full_grid":
                # multiply by (x-self.grid) in clifford space # TODO check if this is correct
                result = torch.einsum("bi...x,bi...y,xyz->bi...z", result, (x_view-self.grid), self.cayley)  # clifford dependant dimensions
                result = torch.einsum("bi...x,io...y,xyz->bioz", result, self.weights, self.cayley)  # clifford dependant dimensions
                #x = x[:,:,0,0,:]
                #slice_dims = [slice(None)] * 2
                #slice_dims += [0] * (x.ndim - 3)
                #slice_dims += [slice(None)]
                #x = x[tuple(slice_dims)]
            elif self.extra_args["clifford_grid"] in ["independant_grid", "random_grid"]:
                # multiply by (x-self.grid) in clifford space # TODO check if this is correct
                result = torch.einsum("bigx,bigy,xyz->bigz", result, (x_view-self.grid), self.cayley)  # clifford independant dimensions
                # Intention: result = torch.einsum("bidg,iodg->bod", result, self.weights)
                result = torch.einsum("bigx,iogy,xyz->bioz", result, self.weights, self.cayley)  # clifford independant dimensions
                #x = x[:,:,0,:]
        assert result.shape[2] == self.output_dim, f"Wrong Output Dimension! Got {result.shape} for Layer with Dimensions[{self.input_dim}, {self.output_dim}]"
        # SiLU
        silu_value = torch.einsum("iox,biy,xyz->bioz", self.silu_weight, self.silu(x), self.cayley)
        # Intention: silu_value = torch.einsum("iod,bigd->biod",self.silu_weight, self.silu(x))
        #silu_value = torch.einsum("biox,ioy,xyz->boz",silu_value, self.silu_bias, self.cayley)
        silu_value += self.silu_bias
        #Intention: silu_value = torch.einsum("biod,iod->bod", silu_value, self.silu_bias)
        result = result + silu_value
        # add all incoming edges together
        result = torch.einsum("biox->box", result)
        # potentially apply Normalization
        result = self.norm(result)
        return result

    def to(self, device):
        super().to(device)
        # move grid to the right device
        self.grid = self.grid.to(device)
        self.algebra = self.algebra.to(device)
        if self.use_norm is not Norms.NoNorm:  # and Norm as well
            self.norm = self.norm.to(device)
        self.silu = self.silu.to(device)
        self.cayley = self.cayley.to(device)
class CliffordKAN(torch.nn.Module):
    def __init__(self,
                 algebra,
                 layers_hidden: List[int],
                 num_grids: int = 8,
                 rho=1,
                 use_norm=Norms.BatchNormComponentWise,
                 grid_mins = -2,
                 grid_maxs = 2,
                 silu_type = "componentwise",
                 extra_args=None):
        """
        :param metric: the Geometric Algebra metric (list[float])
        :param layers_hidden: List with Layer Sizes (i.e. [1,5,3,1] for a 1x5x3x1 CVKAN)
        :param num_grids: Number of Grid Points **per dimension**
        :param rho: rho for RBF (default rho=1)
        :param use_norm: which Normalization scheme to use. Normalization is applied AFTER every layer except the last
        :param grid_mins: left limit of grid
        :param grid_maxs: right limit of grid
        :param silu_type: type of SiLU to use (i.e. componentwise, ...)
        :param extra_args: controls grid and RBF variant
        """
        super().__init__()
        # convert grid limits to list if not already is list (limits for each layer independently)
        if not type(grid_mins) == list:
            grid_mins = [grid_mins] * len(layers_hidden)
        if not type(grid_maxs) == list:
            grid_maxs = [grid_maxs] * len(layers_hidden)
        assert extra_args is not None
        self.extra_args = extra_args
        self.algebra = algebra
        # coordinate dimension
        self.num_dim = 1 << self.algebra.num_bases
        self.layers_hidden = layers_hidden
        self.num_grids = num_grids
        self.rho = rho
        self.use_norm = use_norm
        self.silu_type = silu_type
        # convert silu_type to list (each layer could get it's own SiLU type)
        if type(self.use_norm) != list:
            self.use_norm = [self.use_norm] * (len(layers_hidden) - 1)
        else:
            assert len(self.use_norm) == len(self.layers_hidden)
        # lambdas to calculate if Layer i should have Normalization applied after it
        is_last_layer = lambda i: i >= len(self.layers_hidden) - 2
        norm_to_use = lambda i: self.use_norm[i] if not is_last_layer(i) else Norms.NoNorm
        # Array with Normalization schemes to use after every layer
        self.use_norm = [norm_to_use(i) for i in range(len(layers_hidden)-1)]
        # stack Layers into a ModuleList
        self.layers = torch.nn.ModuleList([CliffordKANLayer(algebra=self.algebra, input_dim=layers_hidden[i], output_dim=layers_hidden[i+1],
                                                      num_grids=self.num_grids, grid_min=grid_mins[i], grid_max=grid_maxs[i],
                                                      rho=self.rho, use_norm=self.use_norm[i],
                                                      silu_type=self.silu_type, extra_args=extra_args) for i in range(len(layers_hidden) - 1)])
    def random_init_weights(self):
        for l in self.layers:
            l.random_init_weights()
    def forward(self, x):
        # make sure x is batched
        if len(x.shape) == 2:  # x might be [input-dim x num_dims] (i.e. a single clifford number)
            x = x.unsqueeze(1)
        # make sure first Layer's grid limits aren't overstepped. This would indicate a problem with dataset
        # normalization (which must be done before entering the data into the model!)
        assert (self.layers[0].grid_min <= torch.amin(x)).all() and (torch.amax(x) <= self.layers[0].grid_max).all(), "Input data does not fall completely within the grid range of the first layer. Please normalize the data!"
        # feed data through the layers
        for layer in self.layers:
            #print("forward call with input", x, "and size", x.size())
            x = layer(x)
        return x
    def to(self, device):
        for layer in self.layers:
            layer.to(device)
