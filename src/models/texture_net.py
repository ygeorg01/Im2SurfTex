import math
from functools import partial

import torch
import torch.nn as nn
# from torch.nn.attention import sdpa_kernel, SDPBackend
from einops import rearrange, pack, unpack, repeat, reduce
from torch import einsum
from collections import OrderedDict
import torch.nn.functional as F

# from data_preprocessing.src.utils.blender_process_glb import save_image
from torchvision.utils import save_image
from src.models.modules import ImageAttention
from src.models.layers import positionalencoding1d, MLP, LayerNormalization


def exists(val):
    return val is not None


class TextureNet(nn.Module):
    def __init__(
            self,
            dim_input,
            dim_context,
            dim_loop=3,
            dim_out=3,
            dim_hidden=32,
            dim_head=64,
            heads=8,
            depth=1,
            norm_type=None,
            ff_mult=2,
            neighboors=256
    ):
        super().__init__()
        self.init_layer_tex = MLP(dim_input, dim_loop, norm=False)  # , type='layer')

        self.pos_encoding = 6
        self.dim_loop = dim_loop
        self.dim_out = dim_out
        self.blending_layers = nn.ModuleList([])
        self.neighboors = neighboors
        self.pos_encoding_xyz = 10
        self.pos_encoding_other = 4

        for i in range(depth):
            self.blending_layers.append(nn.ModuleList([
                ImageAttention(in_channels=dim_loop, dim_hidden=dim_hidden, dim_context=dim_context, out_dim=dim_loop,
                               dim_head=dim_head, heads=heads),
                LayerNormalization(dim_loop),
                MLP(dim_loop, dim_loop, norm=True, type='layer'),
                LayerNormalization(dim_loop)
            ]))

        self.rgb_layer = MLP(dim_hidden, dim_out, norm=False, type='layer')

    def forward(self, x, x_features, tex_geom, tex_pixel_geod_info, active_texels, active_view_texels_iter,
                nn_gamma, nn_theta, nn_dists, nn_ids, pos_normal_nn, loc, normals, mask=None, tex_size=256):
        """
        x - (B, S, 1, 8) // channels (R, G, B, M, x, y, z, max(v@n))
        content - (B, S, N(views x K^2), 8) // channels (R, G, B, zbuf, x, y, z, v@n)
        """

        S, N, _ = x.shape
        Pu = tex_geom

        # Pixel color values
        Fs = x_features[..., :3]

        # Texel xyz, normals, n@v values
        Ps = x_features[..., 3:]


        #  Set center points to zero
        Pu[..., :6] = Pu[..., :6] - Pu[..., :6]

        # Center Pixel Information
        h = x
        loc_enc_pu = positionalencoding1d(Pu[..., :3], self.pos_encoding_xyz)
        view_surf_pu = positionalencoding1d(Pu[..., 3:], self.pos_encoding_other)
        Pu = torch.cat((loc_enc_pu, view_surf_pu), dim=-1)

        # Relative Coordinates
        relative_xyz_normals = (Ps -
                                torch.cat((tex_geom[..., :6], torch.zeros(tex_geom.shape[0],
                                                                          tex_geom.shape[1], 2).to(tex_geom.device)),
                                          dim=-1).repeat(1, Ps.shape[1], 1))

        Ps[mask] = relative_xyz_normals[mask]

        # Neighborhood information
        Fs = Fs
        # XYZ information
        loc_enc_ps = positionalencoding1d(Ps[..., :3], self.pos_encoding_xyz)
        # Normals information
        normal_enc_ps = positionalencoding1d(Ps[..., 3:6], self.pos_encoding_other)
        # NV information
        view_surf_enc_ps = positionalencoding1d(Ps[..., 6:], self.pos_encoding_other)
        # Geodesic information encodings
        geod_info_ps = positionalencoding1d(tex_pixel_geod_info, self.pos_encoding_other)

        Ps = torch.cat((loc_enc_ps, normal_enc_ps, view_surf_enc_ps, geod_info_ps[:,:view_surf_enc_ps.shape[1],:]), dim=-1)

        # Initial encoding
        h = self.init_layer_tex(h)

        # Transformer block
        for attn, norm_1, ff, norm_2 in self.blending_layers:
            h = norm_1(attn(h, Fs, Pu, Ps, mask=mask.bool()) + h)
            h = norm_2(ff(h) + h)

        out = self.rgb_layer(h)
        out_tex = torch.zeros(1, tex_size * tex_size, 1, 3).to(h.device)

        out_tex[active_view_texels_iter] = out

        return out_tex, active_view_texels_iter

    def forward_HD(self, x, x_features, tex_geom, tex_pixel_geod_info, active_texels, active_view_texels_iter,
                   nn_gamma, nn_theta, nn_dists, nn_ids, pos_normal_nn, loc, normals, mask=None, tex_size=256):
        """
        x - (B, S, 1, 8) // channels (R, G, B, M, x, y, z, max(v@n))
        content - (B, S, N(views x K^2), 8) // channels (R, G, B, zbuf, x, y, z, v@n)
        """

        S, N, _ = x.shape

        Pu = tex_geom
        # Pixel color values
        Fs = x_features[..., :3]

        # Texel xyz, normals, n@v values
        Ps = x_features[..., 3:]
        # Relative Coordinates
        relative_xyz_normals = (Ps -
                                torch.cat((tex_geom[..., :6], torch.zeros(tex_geom.shape[0],
                                                                          tex_geom.shape[1], 2).to(tex_geom.device)),
                                          dim=-1).repeat(1, Ps.shape[1], 1))

        Ps[mask] = relative_xyz_normals[mask]
        #  Set center points to zero
        Pu[..., :6] = Pu[..., :6] - Pu[..., :6]

        h = x
        Fs = Fs

        # Initial encoding
        h = self.init_layer_tex(h)

        window_bl = 10000
        for attn, norm_1, ff, norm_2 in self.blending_layers:
            for i in range(0, h.shape[0], window_bl):
                #        print('blending idxes: ', attn, i, i+window_bl, Ps.shape)
                loc_enc_ps = positionalencoding1d(Ps[i:i + window_bl, :, :3], self.pos_encoding_xyz)
                normal_enc_ps = positionalencoding1d(Ps[i:i + window_bl, :, 3:6], self.pos_encoding_other)
                view_surf_enc_ps = positionalencoding1d(Ps[i:i + window_bl, :, 6:], self.pos_encoding_other)

                loc_enc_pu = positionalencoding1d(Pu[i:i + window_bl, :, :3], self.pos_encoding_xyz)
                view_surf_pu = positionalencoding1d(Pu[i:i + window_bl, :, 3:], self.pos_encoding_other)
                Pu_iter = torch.cat((loc_enc_pu, view_surf_pu), dim=-1)
                geod_info_ps = positionalencoding1d(tex_pixel_geod_info[i:i + window_bl], self.pos_encoding_other)

                Ps_iter = torch.cat((loc_enc_ps, normal_enc_ps, view_surf_enc_ps,  geod_info_ps[:,:view_surf_enc_ps.shape[1],:]), dim=-1)

                # Transformer block
                h[i:i + window_bl] = norm_1(attn(h[i:i + window_bl], Fs[i:i + window_bl], Pu_iter,
                                                 Ps_iter, mask=mask[i:i + window_bl].bool()) + h[i:i + window_bl])
                h[i:i + window_bl] = norm_2(ff(h[i:i + window_bl]) + h[i:i + window_bl])


        out = self.rgb_layer(h)

        out_tex = torch.zeros(1, tex_size * tex_size, 1, 3).to(h.device)

        out_tex[active_view_texels_iter] = out

        return out_tex, active_view_texels_iter

    def get_neighboors(self, tensor, nn_ids, n=256):

        tensor = tensor  # .unsqueeze(-2)
        nn_ids = nn_ids.unsqueeze(-1)
        tensor_or = torch.cat((tensor, -torch.ones(tensor.shape[0], 1,
                                                   tensor.shape[2], tensor.shape[3]).to(tensor.device)), dim=1)
        nn_ids = nn_ids.reshape(1, -1, 1, 1)
        nn_ids[nn_ids == -1] = tensor_or.shape[1] - 1

        tensor = torch.gather(tensor_or, 1,
                              nn_ids.expand(-1, -1, -1, tensor_or.shape[-1]).long())

        tensor = tensor.reshape(1, -1, n, tensor_or.shape[-1])

        return tensor

    def store_model(self, dir):
        torch.save(self.state_dict(), dir)


