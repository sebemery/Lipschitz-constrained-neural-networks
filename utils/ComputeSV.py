import torch
from torch.nn.functional import conv2d
from torch.nn.parameter import Parameter


def normalize(tensor, eps=1e-12):
    norm = float(torch.sqrt(torch.sum(tensor * tensor)))
    norm = max(norm, eps)
    # if out is None:
    #     ret = tensor / norm
    # else:
    # ret = torch.div(tensor, norm, out=torch.jit._unwrap_optional(out))
    ans = tensor / norm
    return ans


class SingularValues:
    def __init__(self, model, n_power_iterations=100, dim=0, eps=1e-12):
        self.model = model
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.sigmas = []
        self.names = []

    def compute_sigma(self, weight):
        weight_mat = weight

        if weight_mat.shape[0] == 1:
            C_out = 1
        else:
            C_out = 64

        u = normalize(weight.new_empty(1, C_out, 40, 40).normal_(0, 1), eps=self.eps)

        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                # are the first left and right singular vectors.
                # This power iteration produces approximations of `u` and `v`.
                # v = normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                # u = normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
                # v = normalize(conv2d(u, weight_mat.permute(1,0,2,3), padding=1), dim=0, eps=self.eps)
                # u = normalize(conv2d(v, weight_mat, padding=1), dim=0, eps=self.eps)
                v = normalize(conv2d(u.flip(2,3), weight_mat.permute(1, 0, 2, 3), padding=2),
                              eps=self.eps).flip(2,3)[:,:,1:-1,1:-1]
                u = normalize(conv2d(v, weight_mat, padding=1), eps=self.eps)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone()
                    v = v.clone()

        sigma = torch.sum(u * conv2d(v, weight_mat, padding=1))
        self.sigmas.append(sigma)

    def compute_layer_sv(self):
        for name, param in self.model.dncnn.named_parameters():
            # check convolutional layer
            if "weight" in name:
                self.compute_sigma(param.data)
                self.names.append(name)

