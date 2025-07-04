
import torch
import torch.nn as nn

from einops import rearrange, pack, unpack, repeat, reduce
from torch import einsum
# from torch.nn.attention import sdpa_kernel, SDPBackend

import torch.nn.functional as F
from src.models.layers import MLP, MLP_masked_norm

class ImageAttention(nn.Module):
    def __init__(
            self,
            in_channels,
            dim_hidden,
            dim_context,
            out_dim,
            dim_head=64,
            heads=8
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        dim_inner = dim_head * heads
        self.heads = heads
        self.pos_encoding_xzy=10
        self.pos_encoding_norm=4
        # self.norm = LayerNormalization(in_channels)
        self.fu_mlp = MLP(in_channels, dim_hidden, norm=False)
        self.fs_mlp = MLP_masked_norm(3, dim_hidden, norm=False)
        self.pu_mlp = MLP(3*self.pos_encoding_xzy+4*self.pos_encoding_norm+7, dim_context, norm=False)
        self.ps_loc_depth_ndv_mlp = MLP_masked_norm(3*self.pos_encoding_xzy+8*self.pos_encoding_norm+11, dim_context, norm=False)

        self.q = nn.Linear(dim_hidden+dim_context, dim_inner, bias=False)

        self.k = nn.Linear(dim_hidden+dim_context, dim_inner, bias=False)

        self.v = nn.Linear(dim_hidden, dim_inner, bias=False)

        self.out = nn.Linear(dim_inner, out_dim, bias=False)

    def forward(self, h, fs, Pu, Ps_loc_depth_ndv, mask=None):

        # Normalize Input
        # h = self.norm(h)

        h = self.fu_mlp(h)
        pu = self.pu_mlp(Pu)
        ps_loc_normals_ndv_mlp = self.ps_loc_depth_ndv_mlp(Ps_loc_depth_ndv, mask)

        q, v = torch.cat((h, pu), dim=-1), self.fs_mlp(fs, mask)

        k = torch.cat((v, ps_loc_normals_ndv_mlp), dim=-1)

        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        q = self.q(q).view(sz_b, len_q, self.heads, self.dim_head).transpose(1, 2)
        k = self.k(k).view(sz_b, len_k, self.heads, self.dim_head).transpose(1, 2)
        v = self.v(v).view(sz_b, len_v, self.heads, self.dim_head).transpose(1, 2)
        mask = mask.view(sz_b, 1, len_q, len_k)

        # with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
        q = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.repeat(1,q.shape[1],1,1))

        if q.shape[0]==0:
            return torch.zeros(q.shape).to(q.device).squeeze(0)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        out = self.out(q)

        return out

class TexAttention(nn.Module):
    def __init__(
            self,
            in_channels,
            dim_hidden,
            dim_context,
            out_dim,
            dim_head=64,
            heads=8
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        dim_inner = dim_head * heads
        self.heads = heads
        self.pos_encoding_xzy = 10
        self.pos_encoding_norm = 4
        # self.norm = LayerNormalization(in_channels)
        self.fu_mlp = MLP(in_channels, dim_hidden, norm=False)
        self.fs_mlp = MLP_masked_norm(in_channels, dim_hidden, norm=False)
        self.pu_mlp = MLP(3 * self.pos_encoding_xzy + 6 * self.pos_encoding_norm + 9, dim_context, norm=False)
        self.ps_loc_depth_ndv_mlp = MLP_masked_norm(3 * self.pos_encoding_xzy + 6 * self.pos_encoding_norm + 9,
                                                    dim_context, norm=False)

        self.q = nn.Linear(dim_hidden + dim_context, dim_inner, bias=False)

        self.k = nn.Linear(dim_hidden + dim_context, dim_inner, bias=False)

        self.v = nn.Linear(dim_hidden, dim_inner, bias=False)

        self.out = nn.Linear(dim_inner, out_dim, bias=False)

    def forward(self, Fu, Fs, Pu, Ps, mask=None):
        # Normalize Input
        h = self.fu_mlp(Fu)
        pu = self.pu_mlp(Pu)

        ps_loc_normals_ndv_mlp = self.ps_loc_depth_ndv_mlp(Ps, mask)

        q, v = torch.cat((h, pu), dim=-1), self.fs_mlp(Fs, mask)
        k = torch.cat((v, ps_loc_normals_ndv_mlp), dim=-1)

        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        q = self.q(q).view(sz_b, len_q, self.heads, self.dim_head).transpose(1, 2)
        k = self.k(k).view(sz_b, len_k, self.heads, self.dim_head).transpose(1, 2)
        v = self.v(v).view(sz_b, len_v, self.heads, self.dim_head).transpose(1, 2)
        mask = mask.view(sz_b, 1, len_q, len_k)

        q = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        if q.shape[0]==0:
            return torch.zeros(q.shape).to(q.device).squeeze(0)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        out = self.out(q)

        return out