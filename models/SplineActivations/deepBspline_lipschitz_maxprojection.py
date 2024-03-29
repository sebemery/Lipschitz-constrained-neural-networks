""" See deepBspline_base.py.

Constraining the Lipschitz constant of each spline activation function to be 1
by dividing the coefficients by the maximum absolute slope (which is the Lipschitz constant of the spline).
"""

import torch
from torch import nn
from torch.nn import functional as F
from models.SplineActivations.deepBspline_base import DeepBSplineBase
from qpth.qp import QPFunction
import cvxpy as cp
import numpy as np
# from cvxpylayers.torch import CvxpyLayer


class DeepBSplineLipschitzMaxProjection(DeepBSplineBase):
    """ See deepBspline_base.py
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # tensor with locations of spline coefficients
        grid_tensor = self.grid_tensor # size: (num_activations, size)
        coefficients = torch.zeros_like(grid_tensor) # spline coefficients

        # The coefficients are initialized with the value of the activation
        # at each knot (c[k] = f[k], since B1 splines are interpolators).
        if self.init == 'even_odd':
            # initalize half of the activations with an even function (abs) and
            # and the other half with an odd function (soft threshold).
            half = self.num_activations // 2
            coefficients[0:half, :] = (grid_tensor[0:half, :]).abs()
            coefficients[half::, :] = F.softshrink(grid_tensor[half::, :], lambd=0.5)

        elif self.init == 'relu':
            coefficients = F.relu(grid_tensor)

        elif self.init == 'leaky_relu':
            coefficients = F.leaky_relu(grid_tensor, negative_slope=0.01)

        elif self.init == 'softplus':
            coefficients = F.softplus(grid_tensor, beta=3, threshold=10)

        elif self.init == 'random':
            coefficients.normal_()

        elif self.init == 'identity':
            coefficients = grid_tensor.clone()

        elif self.init != 'zero':
            raise ValueError('init should be even_odd/relu/leaky_relu/softplus/'
                            'random/identity/zero]')

        # Need to vectorize coefficients to perform specific operations
        self.coefficients_vect = nn.Parameter(coefficients.contiguous().view(-1)) # size: (num_activations*size)

        # Create the finite-difference matrix for max projection
        self.init_D()


    @property
    def coefficients_vect_(self):
        return self.coefficients_vect


    def init_D(self):
        """
        Setting up the finite-difference matrix D
        """

        self.D = torch.zeros([self.size-1, self.size], device=self.device)
        size = self.grid.item()
        for i in range(self.size-1):
            self.D[i, i] = -1.0/size
            self.D[i, i+1] = 1.0/size

        # Transpose of D
        self.DT = torch.transpose(self.D, 0, 1)


    def do_lipschitz_projection(self):
        """
        Perform the Lipschitz projection step by dividing the coefficients by the maximum absolute slope
        """
        with torch.no_grad():
            spline_coeffs = self.coefficients.data
            spline_slopes = torch.matmul(spline_coeffs, self.DT)
            div_vals = torch.max(torch.abs(spline_slopes), dim=1, keepdim=True)
            max_abs_slopes = div_vals[0]
            max_abs_slopes[max_abs_slopes < 1.0] = 1.0
            new_spline_coeffs = torch.div(spline_coeffs, max_abs_slopes)
            self.coefficients_vect_.data = new_spline_coeffs.view(-1)


    @staticmethod
    def parameter_names(**kwargs):
        yield 'coefficients_vect'


    def forward(self, input):
        """
        Args:
            input : 2D/4D tensor
        """
        if self.shared_channels is True:
            b, c, h, w = input.shape
            new_dim = int(np.sqrt(c * h * w))
            input_reshaped = torch.reshape(input, (b, 1, new_dim, new_dim))
            output_reshaped = super().forward(input_reshaped)
            output = torch.reshape(output_reshaped, (b, c, h, w))
            return output

        output = super().forward(input)

        return output


    def extra_repr(self):
        """ repr for print(model)
        """
        s = ('mode={mode}, num_activations={num_activations}, '
            'init={init}, size={size}, grid={grid[0]}.')

        return s.format(**self.__dict__)
