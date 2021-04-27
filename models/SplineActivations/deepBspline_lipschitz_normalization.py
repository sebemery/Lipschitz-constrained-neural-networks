""" See deepspline_base.py
This code implements linear splines activation functions with a B-spline parametrization.
The Lipschitz constant is constrained by normalizing the coefficients before the forward pass (similar to spectral normalization)
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
import numpy as np
from scipy.linalg import toeplitz
from abc import abstractproperty
from models.SplineActivations.deepspline_base import DeepSplineBase
from models.SplineActivations.deepBspline_base import DeepBSpline_Func



class DeepBSplineLipschitzNormalization(DeepSplineBase):
    """ See deepspline_base.py
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.init_zero_knot_indexes()
        self.init_derivative_filters()

        # Initializing the spline activation functions

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

        # Create the finite-difference matrix
        self.init_D()

        # Flag to keep track of sparsification process
        self.sparsification_flag = False


    def init_zero_knot_indexes(self):
        """ Initialize indexes of zero knots of each activation.
        """
        # self.zero_knot_indexes[i] gives index of knot 0 for filter/neuron_i.
        # size: (num_activations,)
        activation_arange = torch.arange(0, self.num_activations).to(**self.device_type)
        self.zero_knot_indexes = (activation_arange*self.size + (self.size//2))


    def init_derivative_filters(self):
        """ Initialize D1, D2 filters.
        """
        # Derivative filters
        self.D1_filter = Tensor([-1,1]).view(1,1,2).to(**self.device_type).div(self.grid)
        self.D2_filter = Tensor([1,-2,1]).view(1,1,3).to(**self.device_type).div(self.grid)


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


    @property
    def coefficients_vect_(self):
        """ B-spline vectorized coefficients of activations """
        return self.coefficients_vect


    @property
    def coefficients(self):
        """ B-spline coefficients.
        """
        return self.coefficients_vect_.view(self.num_activations, self.size)


    @property
    def coefficients_grad(self):
        """ B-spline coefficients gradients.
        """
        return self.coefficients_vect_.grad.view(self.num_activations, self.size)


    @property
    def normalized_coefficients(self):
        """ Normalized B-spline coefficients.
        """
        spline_slopes = torch.matmul(self.coefficients, self.DT)
        div_vals = torch.max(torch.abs(spline_slopes), dim=1, keepdim=True)
        max_abs_slopes = div_vals[0]
        max_abs_slopes[max_abs_slopes < 1.0] = 1.0
        new_spline_coeffs = torch.div(self.coefficients, max_abs_slopes)

        return new_spline_coeffs


    @property
    def normalized_coefficients_vect_(self):
        """ Normalized B-spline vectorized coefficients of activations """
        return self.normalized_coefficients.view(-1)


    @property
    def slopes(self):
        """ Get the activation slopes {a_k},
        by doing a valid convolution of the normalized coefficients {c_k}
        with the second-order finite-difference filter [1,-2,1].
        """
        # F.conv1d():
        # out(i, 1, :) = self.D2_filter(1, 1, :) *conv* coefficients(i, 1, :)
        # out.size() = (num_activations, 1, filtered_activation_size)
        # after filtering, we remove the singleton dimension
        #slopes = F.conv1d(self.coefficients.unsqueeze(1), self.D2_filter).squeeze(1)
        slopes = F.conv1d(self.normalized_coefficients.unsqueeze(1), self.D2_filter).squeeze(1)

        return slopes



    def reshape_forward(self, input):
        """ """
        input_size = input.size()
        if self.mode == 'linear':
            if len(input_size) == 2:
                # one activation per conv channel
                x = input.view(*input_size, 1, 1) # transform to 4D size (N, num_units=num_activations, 1, 1)
            elif len(input_size) == 4:
                # one activation per conv output unit
                x = input.view(input_size[0], -1).unsqueeze(-1).unsqueeze(-1)
            else:
                raise ValueError(f'input size is {len(input_size)}D but should be 2D or 4D...')
        else:
            assert len(input_size) == 4, 'input to activation should be 4D (N, C, H, W) if mode="conv".'
            x = input

        return x


    def reshape_back(self, output, input_size):
        """ """
        if self.mode == 'linear':
            output = output.view(*input_size) # transform back to 2D size (N, num_units)

        return output


    @staticmethod
    def parameter_names(**kwargs):
        yield 'coefficients_vect'

    
    def forward(self, input):
        """
        Args:
            input : 2D/4D tensor
        """
        input_size = input.size()
        x = self.reshape_forward(input)
        assert x.size(1) == self.num_activations, 'input.size(1) != num_activations.'

        if self.sparsification_flag is True:
            coefficients = self.sparsified_coefficients
        else:
            coefficients = self.normalized_coefficients

        # Linear extrapolations:
        # f(x_left) = leftmost coeff value + left_slope * (x - leftmost coeff)
        # f(x_right) = second rightmost coeff value + right_slope * (x - second rightmost coeff)
        # where the first components of the sums (leftmost/second rightmost coeff value)
        # are taken into account in DeepBspline_Func() and linearExtrapolations adds the rest.

        leftmost_slope = (coefficients[:,1] - coefficients[:,0]).div(self.grid).view(1,-1,1,1)
        rightmost_slope = (coefficients[:,-1] - coefficients[:,-2]).div(self.grid).view(1,-1,1,1)

        # x.detach(): gradient w/ respect to x is already tracked in DeepBSpline_Func
        leftExtrapolations  = (x.detach() + self.grid*(self.size//2)).clamp(max=0) * leftmost_slope
        rightExtrapolations = (x.detach() - self.grid*(self.size//2-1)).clamp(min=0) * rightmost_slope
        # linearExtrapolations is zero for values inside range
        linearExtrapolations = leftExtrapolations + rightExtrapolations

        output = DeepBSpline_Func.apply(x, coefficients.view(-1), self.grid, self.zero_knot_indexes, self.size) + \
                linearExtrapolations

        output = self.reshape_back(output, input_size)

        return output



    def reset_first_coefficients_grad(self):
        """ """
        first_knots_indexes = torch.cat((self.zero_knot_indexes - self.size//2,
                                    self.zero_knot_indexes - self.size//2 + 1))
        first_knots_indexes = first_knots_indexes.long()

        zeros = torch.zeros_like(first_knots_indexes).float()
        if not self.coefficients_vect_[first_knots_indexes].allclose(zeros):
            raise AssertionError('First coefficients are not zero...')

        self.coefficients_vect_.grad[first_knots_indexes] = zeros


    def iterative_slopes_to_coefficients(self, slopes):
        """ Better conditioned than matrix formulation (see self.P)

        This way, if we set a slope to zero, we can do (b0,b1,a)->c->a'
        and still have the same slope being practically equal to zero.
        This might not be the case if we do: a' = L(self.P(b0,b1,a))
        """
        coefficients = self.normalized_coefficients
        coefficients[:, 2::] = 0. # first two coefficients remain the same

        for i in range(2, self.size):
            coefficients[:, i] = (coefficients[:, i-1] - coefficients[:, i-2]) + \
                                    slopes[:, i-2].mul(self.grid) + coefficients[:, i-1]

        return coefficients


    def apply_threshold(self, threshold):
        """ See DeepSplineBase.apply_threshold method
        """
        with torch.no_grad():
            new_slopes = super().apply_threshold(threshold)
            self.sparsified_coefficients = \
                self.iterative_slopes_to_coefficients(new_slopes).view(-1)
            self.sparsification_flag = True
