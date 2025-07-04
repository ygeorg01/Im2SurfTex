import math
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, pack, unpack, repeat, reduce
from torch import einsum
from collections import  OrderedDict
import torch.nn.functional as F

def exists(val):
    return val is not None

class InstanceNormalization(nn.Module):
    def __init__(self, dim, batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.norm = nn.InstanceNorm1d(dim, affine=True)

    def forward(self, x):
        # print('x shape: ', x.shape)
        return self.norm(x.permute(1,2,0)).permute(2,0,1)

class LayerNormalization(nn.Module):
    def __init__(self, dim_out, batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.norm = nn.LayerNorm(dim_out)#, elementwise_affine=True)

    def forward(self, x):

        return self.norm(x.unsqueeze(0).squeeze(-2)).squeeze(0).unsqueeze(-2)

class LayerNormalization(nn.Module):
    def __init__(self, dim_out, batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.norm = nn.LayerNorm(dim_out)#, elementwise_affine=True)

    def forward(self, x):

        return self.norm(x.unsqueeze(0).squeeze(-2)).squeeze(0).unsqueeze(-2)


class MLP(nn.Module):
    def __init__(self, dim, dim_out, norm=True, type='layer'):
        super().__init__()

        self.dim_out = dim_out
        if norm:
            if type=='layer':
                self.mlp = nn.Sequential(
                    LayerNormalization(dim),
                    nn.Linear(dim, dim_out),
                    nn.LeakyReLU(),
                    nn.Linear(dim_out, dim_out),
                )
            elif type=='instance':
                self.mlp = nn.Sequential(
                    InstanceNormalization(dim),
                    nn.Linear(dim, dim_out),
                    nn.LeakyReLU(),
                    nn.Linear(dim_out, dim_out),
                )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim_out),
                nn.LeakyReLU(),
                nn.Linear(dim_out, dim_out),
                )

    def forward(self, x):

        h = self.mlp(x)

        return h

class MLP_masked_norm(nn.Module):
    def __init__(self, dim, dim_out, norm=False):
        super().__init__()

        self.dim_out = dim_out
        if norm:
            self.mlp = nn.Sequential(LayerNormalization(dim),
                                     nn.Linear(dim, dim_out),
                                     nn.LeakyReLU(),
                                     nn.Linear(dim_out, dim_out))
        else:
            self.mlp = nn.Sequential(nn.Linear(dim, dim_out),
                                     nn.LeakyReLU(),
                                     nn.Linear(dim_out, dim_out))


    def forward(self, x, mask):
        h = torch.zeros(x.shape[0], x.shape[1], self.dim_out).to(x.device)
        if torch.is_autocast_enabled():
            h = h.to(torch.get_autocast_gpu_dtype())
        h[mask] = self.mlp(x[mask])

        return h


def positionalencoding1d(x, d_model):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """

    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))

    length, N, D = x.shape
    D = D * d_model
    x_pe = repeat(x, 's n d-> s n (d repeat)', repeat=d_model // 2)

    pe = torch.zeros(length, N, D, dtype=x.dtype).to(x.device)

    div_term = torch.exp((torch.arange(0, D, 2, dtype=x.dtype) *
                         -(math.log(10000.0) / D))).to(x.device).to(x.dtype)

    pe[:, :, 0::2] = torch.sin(x_pe * div_term)
    pe[:, :, 1::2] = torch.cos(x_pe * div_term)

    # print('pe values: ', torch.min(pe), torch.max(pe), torch.min(x_pe), torch.max(x_pe))

    return torch.cat((pe, x), dim=-1)



