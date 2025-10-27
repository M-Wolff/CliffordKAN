"""
File: CliffordKAN.py
Authors: Matthias Wolff
Description: CliffordKAN Model definition
"""
from typing import List
import torch
import torch_ga


def make_grid_centers(num_centers_per_dimension: int, ga:torch_ga.GeometricAlgebra, grid_min=-2, grid_max=2):
    """
    Create a grid of RBF centers with grid_shape[i] = number of centers for i-th dimension from GA
    num_centers_per_dimension: number of centers per dimension
    return: Tensor with shape (num_centers ^ ga.dim, ga.dim) 
    """
    assert type(num_centers_per_dimension) is int, "num_centers_per_dimension has to be int"
    ranges = [torch.linspace(grid_min, grid_max, steps=num_centers_per_dimension) for _ in range(ga.dim)]
    mesh = torch.meshgrid(*ranges, indexing="ij")  # list of meshgrids
    coords = torch.stack([m.reshape(-1) for m in mesh], dim=-1)  # (num_centers ^ ga.dim, ga.dim)
    assert coords.shape[0]==num_centers_per_dimension ** ga.dim and coords.shape[1]==ga.dim
    return coords

# TODO: come up with Normalizations for Geometric Algebra; have a look at complex-Normalizations for inspiration
#        probably easiest to start like Eq. (13) from CVKAN paper? --> component-wise Batch-Norm?
#class Norms:
#    """Enum for Normalization Types"""
#    LayerNorm = "layernorm"
#    BatchNorm = "batchnorm"  # BN_{\mathbb{C}}
#    BatchNormNaiv = "batchnormnaiv"  # BN_{\mathbb{R}^2}
#    BatchNormVar = "batchnormvar"  # BN_{\mathbb{V}} using variance
#    NoNorm = None


class CliffordKANLayer(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, ga: torch_ga.GeometricAlgebra, num_grids: int = 5, grid_min = -2, grid_max = 2, rho=1):
        """
        :param input_dim: input dimension size of Layer (Layer Width)
        :param output_dim: output dimension size of Layer (next Layer's Width)
        :param ga: GeometricAlgebra to use
        :param num_grids: number of grid points ***per dimension***
        :param grid_min: left limit of grid
        :param grid_max: right limit of grid
        :param rho: rho for use in RBF (default rho=1)
        """
        # TODO: :param use_norm: which Normalization scheme to use
        # TODO: experiment with something similar to param csilu_type: the kind of CSiLU to use ('complex_weight' or 'real_weights') but for Clifford-Algebra
        #       previous experiments with CVKAN have shown that some residual function like SiLU etc. is very necessary for training.
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_grids = num_grids
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.ga = ga
        #self.csilu_type = csilu_type
        # TODO implement batchnorm for Clifford space
        #self.use_norm = use_norm
        # initialize Norm instance corresponding to self.use_norm
        """
        if self.use_norm == Norms.LayerNorm:
            self.norm = Complex_LayerNorm()
        elif self.use_norm == Norms.BatchNorm:
            self.norm = Complex_BatchNorm(num_channel=output_dim)
        elif self.use_norm == Norms.BatchNormNaiv:
            self.norm = Complex_BatchNorm_naiv(num_channel=output_dim)
        elif self.use_norm == Norms.BatchNormVar:
            self.norm = Complex_BatchNorm_var(num_channel=output_dim)
        elif self.use_norm == Norms.NoNorm:
            self.norm = None
        else:
            raise NotImplementedError()
        """
        # create grid centers array
        self.grid_centers = make_grid_centers(num_centers_per_dimension=num_grids, ga=self.ga)
        # grid is a non-trainable Parameter
        self.grid_centers = torch.nn.Parameter(self.grid_centers, requires_grad=False)
        self.rho = rho
        # weights for each RBF centered around the grid points. Thus each grid-center gets one weight
        self.weights = torch.nn.Parameter(torch.randn(size=(input_dim, output_dim, *self.grid_centers.shape)), requires_grad=True)
        # self.grid_centers has dim: [centers_per_dim ^ ga.dim, ga.dim] where each row lists one combination of coordinates for all dims
        # self.weights has dim: [input_dim, output_dim, ...] where ... is dim of self.grid_centers
        # so every connection can have different weights but centers for RBF are shared
        # TODO: until now the weights and grid_centers are real-valued; how to convert a [8*8,2] = [64, 2] tensor to an element of GA with dim [64]?
        # TODO: how to convert [2,3,64,2] to element of GA with dim [2,3,64] (input x output x num_centers_per_dimension ^ ga.dim)
        #       similar to converting real and imaginary parts into one tensor with dtype=torch.complex64
        #       should I convert? or just use torch_ga.multiply(...)? In which dimension does that take place? --> would be easiest to just use torch.einsum()
        print(self.weights.shape, self.grid_centers.shape)
        # TODO Csilu for Clifford
        """
        # initialize CSiLU weight to use based on selected csilu_type
        if self.csilu_type == "complex_weight":
            self.silu_weight = torch.nn.Parameter(torch.ones(size=(self.input_dim, self.output_dim), dtype=torch.complex64), requires_grad=True)
        elif self.csilu_type == "real_weights":
            self.silu_weight = torch.nn.Parameter(torch.ones(size=(self.input_dim, self.output_dim, 2), dtype=torch.float32), requires_grad=True)
        else:
            raise NotImplementedError()
        # add complex-valued bias to CSiLU
        self.silu_bias = torch.nn.Parameter(torch.zeros(size=(input_dim, output_dim), dtype=torch.complex64), requires_grad=True)
        """
        self.silu_bias=0


    def forward(self, x):
        """
        Expect input x to already be inside GA
        """
        # TODO assert x in self.ga ?
        # x has shape BATCH x Input-Dim and contains elements from GA
        assert len(x.shape) == 2 and x.shape[1] == self.input_dim, f"Wrong Input Dimension! Got {x.shape} for Layer with Dimensions[{self.input_dim}, {self.output_dim}]"
        # apply RBF on x (centered around each grid point)
        print(x.shape)
        print(x.dtype)
        print(self.grid_centers.shape)
        result = torch.exp(-(torch.abs(x - self.grid_centers)) ** 2 / self.rho)
        # i and o are input and output indices in current layer
        # b is batch dimension
        # c is "grid_center dimension" (different c's represent the different grid centers for RBFs)
        # TODO: does torch.einsum work on dtype GA?
        result = torch.einsum("bi,ioc->bo", result, self.weights)
        assert result.shape[1] == self.output_dim, f"Wrong Output Dimension! Got {result.shape} for Layer with Dimensions[{self.input_dim}, {self.output_dim}]"
        # TODO: SiLU in GA
        """
        # kind of CSiLU used
        if self.csilu_type == "complex_weight":
            silu_value = torch.einsum("io,bi->bio",self.silu_weight, complex_silu_complexweight(x))
        elif self.csilu_type == "real_weights":
            silu_value_raw = complex_silu_realweights(x)
            silu_value = torch.complex(torch.einsum("io,bi->bio",self.silu_weight[:,:,0], silu_value_raw[0]),
                                       torch.einsum("io,bi->bio",self.silu_weight[:,:,1], silu_value_raw[1]))
        # Add complex CSiLU bias
        silu_value += self.silu_bias
        silu_value = torch.einsum("bio->bo", silu_value)
        result = result + silu_value
        """
        # TODO Norms in GA
        """
        # potentially apply Normalization
        if self.use_norm is not None:
            result_complex = self.norm(result_complex)
        """
        return result

    def to(self, device):
        super().to(device)
        # move grid to the right device
        self.grid_centers = self.grid_centers.to(device)
        # TODO Norms in GA
        """
        if self.use_norm is not None:  # and Norm as well
            self.norm = self.norm.to(device)
        """
class CliffordKAN(torch.nn.Module):
    def __init__(self,
                 ga: torch_ga.GeometricAlgebra,
                 layers_hidden: List[int],
                 num_grids: int = 8,
                 rho=1,
                 #use_norm=Norms.BatchNorm,
                 grid_mins = -2,
                 grid_maxs = 2,
                 csilu_type = "complex_weight"):
        """
        :param ga: GeometricAlgebra object to use
        :param layers_hidden: List with Layer Sizes (i.e. [1,5,3,1] for a 1x5x3x1 CliffordKAN)
        :param num_grids: Number of Grid Points ***per dimension*** 
        :param rho: rho for RBF (default rho=1)
        :param use_norm: which Normalization scheme to use. Normalization is applied AFTER every layer except the last
        :param grid_mins: left limit of grid
        :param grid_maxs: right limit of grid
        :param csilu_type: type of CSiLU to use ('complex_weight' or 'real_weights')
        """
        super().__init__()
        # convert grid limits to list if not already is list (limits for each layer independently)
        if not type(grid_mins) == list:
            grid_mins = [grid_mins] * len(layers_hidden)
        if not type(grid_maxs) == list:
            grid_maxs = [grid_maxs] * len(layers_hidden)
        self.layers_hidden = layers_hidden  # width of hidden layers (including input and output layer)
        self.num_grids = num_grids
        self.rho = rho
        #self.use_norm = use_norm
        self.csilu_type = csilu_type
        # convert csilu_type to list (each layer could get it's own CSiLU type)
        #if type(self.use_norm) != list:
        #    self.use_norm = [self.use_norm] * (len(layers_hidden) - 1)
        #else:
        #    assert len(self.use_norm) == len(self.layers_hidden)
        # lambdas to calculate if Layer i should have Normalization applied after it
        is_last_layer = lambda i: i >= len(self.layers_hidden) - 2
        norm_to_use = lambda i: self.use_norm[i] if not is_last_layer(i) else None#Norms.NoNorm
        # Array with Normalization schemes to use after every layer
        self.use_norm = [norm_to_use(i) for i in range(len(layers_hidden)-1)]
        # stack Layers into a ModuleList
        # TODO pass SiLU type and norm to CliffordKANLayer
        self.layers = torch.nn.ModuleList([CliffordKANLayer(input_dim=layers_hidden[i], output_dim=layers_hidden[i+1], ga=ga,
                                                      num_grids=num_grids, grid_min=grid_mins[i], grid_max=grid_maxs[i],
                                                      rho=self.rho) for i in range(len(layers_hidden) - 1)])
    def forward(self, x):
        # make sure x is batched
        if len(x.shape) == 1:  # TODO adapt this one to GA
            x = x.unsqueeze(1)
        # find mins over all samples within batch
        # x has shape Batch x Input-Dim and we want each component of x to be inside of our grid
        
        mins_per_channel = torch.minimum(torch.amin(x.real, dim=0), torch.amin(x.imag, dim=0))
        maxs_per_channel = torch.maximum(torch.amax(x.real, dim=0), torch.amax(x.imag, dim=0))
        # make sure first Layer's grid limits aren't overstepped. This would indicate a problem with dataset
        # normalization (which must be done before entering the data into the model!)
        assert (mins_per_channel >= self.layers[0].grid_min).all() and (maxs_per_channel <= self.layers[0].grid_max).all(), "Input data does not fall completely within the grid range of the first layer. Please normalize the data!"
        # feed data through the layers
        for layer in self.layers:
            x = layer(x)
        return x
    def to(self, device):
        for layer in self.layers:
            layer.to(device)
if __name__ == "__main__":
    ga = torch_ga.GeometricAlgebra([1,-1])
    cliffKAN = CliffordKAN(ga=ga, layers_hidden=[1,1])
    x = ga.from_tensor_with_kind(torch.tensor([[1,-1.5], [2,-1]]), kind="vector")
    y = ga.from_tensor_with_kind(torch.tensor([[2.0,0],[-1,1]]), kind="vector")
    x_mv = ga(x)
    y_mv = ga(y)
    print("x")
    print(x)
    print(x)
    ga.print(x[0])
    ga.print(x[1])
    print("y")
    print(y)
    ga.print(y[0])
    ga.print(y[1])
    print("x*y")
    print(x*y)  # TODO: this does element-wise multiplication, right? Is this what we want for weights * rbf_values?
    ga.print((x*y)[0])
    ga.print((x*y)[1])
    # TODO: this doesnt seem to do the right thing? (1 - 1.5i) * (2 + 0i) should be (2 - 3i) right? acutally is just 2 + 0i
    exit(0)
    # TODO: what is a cayley?
    #print(torch_ga.mv_multiply(x,y))
    #print(torch_ga.mv_multiply_element_wise(x,y))
    ga.print(x[0])
    ga.print(x[1])
    print(ga.num_bases)
    exit(0)
    ga.print(cliffKAN(x))

    #cliffkan = CliffordKAN(geometric_algebra=ga, )
