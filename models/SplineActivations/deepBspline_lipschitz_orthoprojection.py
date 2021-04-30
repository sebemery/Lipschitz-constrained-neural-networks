""" See deepBspline_base.py.

Constraining the Lipschitz constant of each spline activation function to be 1
via quadratic programming.
"""

import torch
from torch import nn
from torch.nn import functional as F
from models.SplineActivations.deepBspline_base import DeepBSplineBase
from qpth.qp import QPFunction
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np


class DeepBSplineLipschitzOrthoProjection(DeepBSplineBase):
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

        # Setup the matrices for quadratic programming
        self.init_QP()


    @property
    def coefficients_vect_(self):
        return self.coefficients_vect


    def init_QP(self):
        """
        Setting up the matrices Q, G, h for the QP
        """
        if self.QP == "qpth":
            # Using the qpth library for the QP
            self.Q = 2.0*torch.eye(self.size, self.size, device=self.device)
            self.e = torch.Tensor()
            self.e = (self.e).to(device=self.device)

            # Create the finite-difference matrix
            T = self.grid.item()
            D = torch.zeros(self.size-1, self.size, device=self.device)
            for i in range(self.size-1):
                D[i, i] = -1.0/T
                D[i, i+1] = 1.0/T

            self.G = torch.cat([D, -D], dim=0)
            self.h = torch.ones(2*(self.size-1), device=self.device)

        elif self.QP == "cvxpy":
            # Using the cvxpylayers library for the QP
            Q = 2.0*np.eye(self.size)
            p = cp.Parameter(self.size)

            # Create the finite-difference matrix
            D = np.zeros([self.size-1, self.size])
            size = self.grid.item()
            for i in range(self.size-1):
                D[i, i] = -1.0/size
                D[i, i+1] = 1.0/size

            G = np.concatenate((D, -D), axis=0)
            h = np.ones(2*(self.size-1))

            # Create the QP
            x = cp.Variable(self.size)
            objective = cp.Minimize((1/2)*cp.quad_form(x, Q) + p.T @ x)
            constraints = [G @ x <= h]
            problem = cp.Problem(objective, constraints)

            self.qp = CvxpyLayer(problem, parameters=[p], variables=[x])

    def do_lipschitz_projection(self):
        """
        Perform the Lipschitz projection step by solving the QP
        """

        with torch.no_grad():
            if self.QP == "qpth":
                # qpth library
                proj_coefficients = QPFunction(verbose=False)(nn.Parameter(self.Q), -2.0*self.coefficients, nn.Parameter(self.G), nn.Parameter(self.h), nn.Parameter(self.e), nn.Parameter(self.e))
                self.coefficients_vect_.data = proj_coefficients.view(-1)

            elif self.QP == "cvxpy":
                # cvxpylayers library
                """
                # row_wise verification 
                proj_coefficients = torch.empty(self.coefficients.data.shape)
                for i in range(self.coefficients.data.shape[0]):
                    proj_coefficient, = self.qp(-2.0*self.coefficients.data[i, :])
                    proj_coefficients[i, :] = proj_coefficient
                self.coefficients_vect_.data = proj_coefficients.view(-1)
                """
                proj_coefficients, = self.qp(-2.0 * self.coefficients.data)
                self.coefficients_vect_.data = proj_coefficients.view(-1)
    @staticmethod
    def parameter_names(**kwargs):
        yield 'coefficients_vect'

    def forward(self, input):
        """
        Args:
            input : 2D/4D tensor
        """
        output = super().forward(input)

        return output

    def extra_repr(self):
        """ repr for print(model)
        """
        s = ('mode={mode}, num_activations={num_activations}, '
            'init={init}, size={size}, grid={grid[0]}.')

        return s.format(**self.__dict__)
