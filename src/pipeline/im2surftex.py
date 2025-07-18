import os

import torch

import pytorch_lightning as pl

# customized
import sys

sys.path.append("./helper")
import lightning as L
import wandb
from einops import rearrange
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
from src.helper.mesh_helper import init_mesh_ours

from src.helper.render_helper import render_depth_normals_tex_with_uvs_normals, render_multi_view, \
    store_masks, get_view_texture

import torch.nn as nn
from pytorch3d.io import save_obj

import torch
import torchvision.transforms.functional as F
import torch.nn.functional as F_nn
from torchvision.transforms import ToPILImage
from PIL import Image
import numpy as np


def exists(val):
    return val is not None

def get_unused_parameters(model: nn.Module):
    # Flatten all parameters in optimizers
    unused = []
    for name, param in model.named_parameters():

        # If it requires grad but has no grad after backward, it's unused
        if param.requires_grad and param.grad is None:
            unused.append(name)
        else:
            pass
            # print('ok name, param: ', name)
    return unused



class Im2SurfTexModel(L.LightningModule):

    def __init__(self,
                 config,
                 stamp,
                 ):

        super().__init__()
        self.norm_type = 'layer'

        self.config = config
        self.stamp = stamp
        self.log_stamp = self.stamp

        print('self config arguments: ', self.config)
        self.log_dir = self.config.log_dir
        self.cross_attention_window = self.config.cross_attention_window

        self.use_amp = self.config.use_amp
        self.amp_dtype = torch.bfloat16 if self.config.use_bf16 else torch.float16

        self.save_hyperparameters()

        pl.seed_everything(self.config.seed)

        self.l1Loss = torch.nn.L1Loss()
        self.num_views = 4

        self.in_channels = 3
        self.hidden_dim = 32
        self.render_channels = 3
        self.out_channels = 3
        self.relative_coords = True

        from src.models.texture_net import TextureNet

        self.number_of_neighboors = 200

        # Blending Module
        self.texture_net = TextureNet(dim_input=self.in_channels, dim_context=self.hidden_dim,
                                      dim_loop=self.hidden_dim, dim_hidden=self.hidden_dim,
                                      dim_out=self.out_channels, heads=2, depth=3, ff_mult=1,
                                      dim_head=self.hidden_dim, norm_type='layer', neighboors=self.number_of_neighboors)

        self.apply(self.init_)

        if config.checkpoint_path is not None:
            print('Loading weights...')
            self.load_state_dict(torch.load(self.config.checkpoint_path))


    def init_(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias != None:
                nn.init.zeros_(m.bias)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.texture_net.parameters(), lr=0.001)  # , weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.num_epochs)
        return [optimizer], [scheduler]

    def _fp16(self, dtype):
        for param in self.parameters():
            param.data = param.data.to(dtype)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(dtype)

    def configure(self, inference_mode=False):
        if not inference_mode:
            self.log_name = self.config.exp_name
            self.version_name = self.config.version_name
            self.log_stamp = self.stamp
            self.log_dir = os.path.join(self.config.log_dir, self.log_name, self.log_stamp + '_' + self.version_name)

            # override config
            self.config.log_name = self.log_name
            self.config.log_stamp = self.log_stamp
            self.config.log_dir = self.log_dir

    def init_inputs(self, tex_features, active_view_texels, locations, normals, tex_input):

        # B, C, N, S
        B, C, N, S = tex_features.shape

        active_view_texels = active_view_texels

        loc = locations.reshape(B, -1, 1, 3)
        normals = normals.reshape(B, -1, 1, 3)

        n_dot_v = tex_features[..., [-1]]
        center_idxs = [i * self.cross_attention_window ** 2 + self.cross_attention_window ** 2 // 2 for i in
                       range(tex_features.shape[2] // self.cross_attention_window ** 2)]

        n_dot_v[n_dot_v == -1] = 0

        mean_dot_v = torch.sum(n_dot_v[:, :, center_idxs, :], dim=-2) / torch.sum(n_dot_v[:, :, center_idxs, :] != 0,
                                                                                  dim=-2)
        mean_dot_v[torch.isnan(mean_dot_v)] = -1

        tex_geom = torch.cat((loc, normals, mean_dot_v.unsqueeze(-1)), dim=-1)

        tex = -torch.ones(tex_input.shape).to(tex_input.device)

        return tex, tex_features, active_view_texels, loc, normals, tex_geom

    # Function to erode a binary mask
    def erode_mask(self, mask, kernel_size=3, iterations=1):
        """
        Erodes a binary mask using a square kernel.

        Args:
            mask (torch.Tensor): Binary mask tensor of shape (H, W) or (1, H, W).
            kernel_size (int): Size of the square kernel for erosion.
            iterations (int): Number of times erosion is applied.

        Returns:
            torch.Tensor: Eroded mask tensor.
        """

        # mask = mask.squeeze(0)
        mask = mask.permute(3, 0, 1, 2)

        # Create a square kernel
        kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32).to(mask.device)
        padding = kernel_size // 2

        # Apply erosion iteratively
        eroded_mask = mask.float()

        for _ in range(iterations):
            eroded_mask = F_nn.conv2d(eroded_mask, kernel, padding=padding)
            eroded_mask = (eroded_mask > 7).float()  # Threshold to binary

        return eroded_mask.permute(1, 2, 3, 0)

    def get_input(self, tex, tex_hd, tex_features, active_texels, loc, normals, in_channels, dtype, first_iter=False,
                  train=True):



        B, H, W, D1 = tex.shape
        N, _, _, D2 = tex_features.shape

        # Reshape features (H*W, N, feature_dims)
        tex_features = rearrange(tex_features, '(b v) h w c-> b v h w c', b=B).reshape(B, N // B, -1, D2).permute(0, 2,
                                                                                                                  1, 3)
        tex_pixel_coordinates = tex_features[..., -1]
        tex_features = tex_features[..., :-1]

        # Reshape current texture (H*W, 1, input_dims)
        tex_real = tex.reshape(B, -1, D1).unsqueeze(-2)
        tex_real_hd = tex_hd.reshape(B, -1, D1).unsqueeze(-2)

        # Initialize input tex
        tex = -torch.ones(B, H, W, D1).to(tex.device).reshape(B, -1, 1, D1)

        # All valid texels
        active_texels = active_texels

        # Valid texels
        mask1 = tex_features[..., -2] != -1
        # mask2 = tex_features[..., -1] > 0.2
        mask2 = tex_features[..., -1] != -1


        mask_comb = torch.logical_and(mask1, mask2)
        active_tokens = torch.logical_and(mask_comb, active_texels)
        # print('torch. sum: ', torch.sum(mask1), torch.sum(mask2), torch.sum(mask_comb), torch.sum(active_texels), torch.sum(active_texels))

        tex_features[torch.logical_not(active_tokens)] = -1

        active_view_texels = torch.logical_and(torch.sum(active_tokens, dim=-1) > 0
                                               ,active_texels.reshape(active_texels.shape[0], -1))

        # active_view_texels = torch.sum(active_tokens, dim=-1) > 0

        return (tex_real.to(dtype), tex_real_hd.to(dtype), tex.to(dtype), tex_features.to(dtype), active_texels,
                active_view_texels, active_tokens, tex_pixel_coordinates)

    def get_neighboors(self, tensor, nn_ids, n=256):

        tensor_or = torch.cat((tensor, -torch.ones(tensor.shape[0], 1,
                                                   tensor.shape[2], tensor.shape[3]).to(tensor.device)), dim=1)
        # nn_ids = nn_ids.reshape(1, -1, 1, 1)
        nn_ids = nn_ids.reshape(1, -1, 1, 1)
        nn_ids[nn_ids == -1] = tensor_or.shape[1] - 1

        tensor = torch.gather(tensor_or, 1,
                              nn_ids.expand(-1, -1, -1, tensor_or.shape[-1]).long())

        tensor = tensor.reshape(1, -1, n, tensor_or.shape[-1])
        # tensor = rearrange(tensor.squeeze(-2), 'b (s n) d-> b s n d', n=n)
        return tensor

    def texture_forward(self, tex_input, tex_features, tex_geom, tex_pixel_geod_info, active_texels, active_view_texels,
                        active_tokens,
                        nn_gamma, nn_theta, nn_dists, nn_ids, loc, normals, texture_map_size=256):

        # Blending Step
        if torch.is_autocast_enabled():
            tex_input = tex_input.to(torch.get_autocast_gpu_dtype())

        # print('Blending inputs: ', tex_input.shape, tex_features.shape, tex_geom.shape, tex_pixel_geod_info.shape, active_tokens.shape, active_view_texels.shape)

        # Tiled Implementation
        split_window = 256
        split_window = split_window ** 2
        texture_map_size = texture_map_size
        geod_info_idx = 0

        for start_idx in range(0, texture_map_size ** 2, split_window):

            active_view_texels_iter = active_view_texels.clone()
            active_view_texels_iter[:, :] = False
            active_view_texels_iter[:, start_idx:start_idx + split_window] = True
            active_view_texels_iter = torch.logical_and(active_view_texels_iter, active_view_texels)

            tex_pixel_geod_info_iter = tex_pixel_geod_info[
                                       geod_info_idx:geod_info_idx + torch.sum(active_view_texels_iter)]
            geod_info_idx = torch.sum(active_view_texels_iter)

            pos_normal_nn = self.get_neighboors(torch.cat((loc, normals), dim=-1),
                                                nn_ids[active_texels[..., 0]].unsqueeze(0).unsqueeze(-1),
                                                n=self.number_of_neighboors)


            if torch.sum(active_view_texels_iter) == 0:
                print('continue')
                continue

            tex_out, inpainting_texels = self.texture_net(tex_input[active_view_texels_iter],
                                                          tex_features[active_view_texels_iter],
                                                          tex_geom[active_view_texels_iter],
                                                          tex_pixel_geod_info_iter,
                                                          active_texels,
                                                          active_view_texels_iter,
                                                          nn_gamma, nn_theta, nn_dists, nn_ids,
                                                          pos_normal_nn,
                                                          loc, normals,
                                                          mask=active_tokens[active_view_texels_iter])


        return tex_out, active_view_texels, inpainting_texels

    def texture_forward_HD(self, tex_input, tex_features, tex_geom, tex_pixel_geod_info, active_texels,
                           active_view_texels, active_tokens,
                           nn_gamma, nn_theta, nn_dists, nn_ids, loc, normals, texture_map_size=256):

        # Blending Step
        if torch.is_autocast_enabled():
            tex_input = tex_input.to(torch.get_autocast_gpu_dtype())

        # print('Blending inputs: ', tex_input.shape, tex_features.shape, tex_geom.shape, tex_pixel_geod_info.shape, active_tokens.shape, active_view_texels.shape)

        # Tiled Implementation
        split_window = 1024
        split_window = split_window ** 2
        texture_map_size = texture_map_size
        geod_info_idx = 0
        tex_out = None

        for start_idx in range(0, texture_map_size ** 2, split_window):

            active_view_texels_iter = active_view_texels.clone()
            active_view_texels_iter[:, :] = False
            active_view_texels_iter[:, start_idx:start_idx + split_window] = True
            active_view_texels_iter = torch.logical_and(active_view_texels_iter, active_view_texels)

            tex_pixel_geod_info_iter = tex_pixel_geod_info[
                                       geod_info_idx:geod_info_idx + torch.sum(active_view_texels_iter)]
            geod_info_idx = torch.sum(active_view_texels_iter)

            pos_normal_nn = self.get_neighboors(torch.cat((loc, normals), dim=-1),
                                                nn_ids[active_texels[:, :, 0]].unsqueeze(0).unsqueeze(-1),
                                                n=self.number_of_neighboors)

            if torch.sum(active_view_texels_iter) == 0:
                print('continue')
                continue

            tex_out, inpainting_texels = self.texture_net.forward_HD(tex_input[active_view_texels_iter],
                                                                     tex_features[active_view_texels_iter],
                                                                     tex_geom[active_view_texels_iter],
                                                                     tex_pixel_geod_info_iter,
                                                                     active_texels,
                                                                     active_view_texels_iter,
                                                                     nn_gamma, nn_theta, nn_dists, nn_ids,
                                                                     pos_normal_nn,
                                                                     loc, normals,
                                                                     mask=active_tokens[active_view_texels_iter],
                                                                     tex_size=1024)

        return tex_out, active_view_texels, inpainting_texels

    def map_to_full_resolutions(self, tensor, indexes, res=256):

        map = -torch.ones(1, (res * res), tensor.shape[-1]).to(tensor.device)
        map[0, indexes[-1].long(), :] = tensor

        return map

    def get_nn_maps(self, nn_texel_ids, nn_ids, nn_gamma, nn_theta, nn_dists, res=256):

        nn_ids[nn_ids == (res * res) - 1] = nn_texel_ids.shape[1]

        nn_texel_ids = torch.cat((nn_texel_ids, -torch.ones((1, 1)).to(nn_texel_ids.device)), dim=-1)
        nn_ids = nn_texel_ids[0, nn_ids.reshape(-1).long()].reshape(1, -1, nn_ids.shape[-1])

        nn_texel_ids = nn_texel_ids[:, :-1]

        nn_ids = self.map_to_full_resolutions(nn_ids, nn_texel_ids, res)  # [...,::2]
        nn_theta = self.map_to_full_resolutions(nn_theta, nn_texel_ids, res)  # [...,::2]
        nn_gamma = self.map_to_full_resolutions(nn_gamma, nn_texel_ids, res)  # [...,::2]
        nn_dists = self.map_to_full_resolutions(nn_dists, nn_texel_ids, res)  # [...,::2]

        nn_ids = nn_ids[:, :, :self.number_of_neighboors]
        nn_gamma = nn_gamma[:, :, :self.number_of_neighboors]
        nn_theta = nn_theta[:, :, :self.number_of_neighboors]
        nn_dists = nn_dists[:, :, :self.number_of_neighboors]

        # Check for invalid neighboors
        nn_ids[torch.isinf(nn_dists)] = -1
        nn_gamma[torch.isinf(nn_dists)] = -1
        nn_theta[torch.isinf(nn_dists)] = -1
        nn_dists[torch.isinf(nn_dists)] = -1

        return nn_ids, nn_gamma, nn_theta, nn_dists

    def gather_geod_info(self, nn_ids, indexes, nn_info, tex_pixel_coords_act_no_zero, active_view_texels, i, window_,
                         window_size, res=256):

        channels = nn_info.shape[-1]
        nn_geod_info = (-1 * torch.ones((nn_ids.shape[0],
                                         window_size,
                                         nn_ids.shape[1], channels))).reshape(nn_ids.shape[0], -1, channels).to(
            nn_ids.device)

        nn_info_act = nn_info[active_view_texels][i:i + window_]
        nn_info_act = nn_info_act.reshape(nn_ids.shape[0], -1, channels)

        indexes = torch.clamp(indexes.long(), min=0, max=nn_geod_info.shape[1] - 1)
        nn_geod_info[:, indexes] = nn_info_act

        nn_geod_info = nn_geod_info.reshape(1, tex_pixel_coords_act_no_zero.shape[0], -1, channels)

        tex_pixel_coords_act_no_zero = torch.clamp(tex_pixel_coords_act_no_zero, min=0, max=(res * res) - 1)
        return torch.gather(nn_geod_info, 2,
                            tex_pixel_coords_act_no_zero.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, channels).long())

    def training_step(self, batch, batch_idx):

        batch = self.get_data(batch)
        # opt_blend, opt_inpaint = self.optimizers()

        camera_params = self.create_camera_params(batch['cam_params'])

        # Render Views Normals and
        images_set, _ = render_depth_normals_tex_with_uvs_normals(batch['mesh'], batch['tex'], batch['tex_hd'],
                                                                  camera_params, self.config.render_size,
                                                                  1, self.cross_attention_window, self.device,
                                                                  gen_images=batch['gen_images'])

        tex_real, tex_real_hd, tex, tex_features, active_texels, active_view_texels, active_tokens, tex_pixel_coordinates = self.get_input(
            batch['tex'],
            batch['tex_hd'],
            images_set['tex_features'],
            images_set['active_texels'],
            images_set['uv_loc'],
            images_set['uv_normals'],
            self.in_channels,
            self.dtype)

        # Initialize inputs
        tex, tex_features, active_view_texels, loc, normals, tex_geom = self.init_inputs(tex_features,
                                                                                         active_view_texels,
                                                                                         images_set['uv_loc'],
                                                                                         images_set['uv_normals'], tex)

        if batch['nn_ids'].shape[-1] != batch['nn_gamma'].shape[-1]:
            loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            return loss

        nn_ids, nn_gamma, nn_theta, nn_dists = self.get_nn_maps(batch['nn_texels_id'], batch['nn_ids'],
                                                                batch['nn_gamma'], batch['nn_theta'],
                                                                batch['nn_dists'])


        tex_pixel_geod_info = self.gather_view_geod_info(tex_pixel_coordinates, nn_ids, nn_theta, nn_dists, nn_gamma,
                                                         active_view_texels)


        if not torch.sum(active_view_texels) or torch.sum(active_view_texels) < 1000:# or torch.sum(
                # active_texels) > 65000:
            print('=====> \n', torch.sum(active_texels))
            print(f'ERROR: No active view texels for {batch["folder_path"]}')
            loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            print('=================================================')
            return loss


        # Run and update blending
        with torch.autocast(device_type=batch['tex'].device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            tex_blend, mask_blend, loss_blend = self.one_train_step(tex, tex_features, tex_geom, tex_pixel_geod_info,
                                                                    active_view_texels, active_tokens, tex_real,
                                                                    active_texels, 0, batch_idx, nn_gamma,
                                                                    nn_theta, nn_dists, nn_ids, loc, normals)

        if torch.isnan(loss_blend).any():
            print('Blend loss error with sample: ', batch['folder_path'])
            loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            return loss

        if self.global_rank == 0:
            self.log("blend_loss", loss_blend, on_epoch=True, prog_bar=True)
            # Log the learning rate
            optimizer = self.optimizers()
            lr = optimizer.param_groups[0]['lr']
            self.log("learning_rate", lr, on_epoch=True, prog_bar=True)

        return loss_blend

    def one_train_step(self, tex, tex_features, tex_geom, tex_pixel_geod_info, active_view_texels, active_tokens,
                       tex_real, active_texels, opt_blend, batch_idx, nn_gamma, nn_theta, nn_dists, nn_ids, loc,
                       normals):

        tex, mask_blend, mask_act_texels = self.texture_forward(tex, tex_features, tex_geom, tex_pixel_geod_info,
                                                                active_texels, active_view_texels,
                                                                active_tokens, nn_gamma, nn_theta, nn_dists, nn_ids,
                                                                loc, normals)

        # Update inpainting module weight
        loss_blend = self.l1Loss(tex[mask_act_texels].squeeze(), tex_real[mask_act_texels].squeeze())

        return tex, mask_blend, loss_blend

    @torch.no_grad()
    def gather_view_geod_info(self, tex_pixel_coordinates, nn_ids, nn_theta, nn_dists, nn_gamma, active_view_texels,
                              res=256, out_dir=None):

        if out_dir != None:
            if os.path.exists(os.path.join(out_dir, 'geod_info_'+str(res)+'.pt')):
                return torch.load(os.path.join(out_dir, 'geod_info_'+str(res)+'.pt'))

        tex_pixel_geod_info = -torch.ones(tex_pixel_coordinates.shape).to(tex_pixel_coordinates.device)[
            active_view_texels]
        tex_pixel_geod_info = tex_pixel_geod_info.unsqueeze(-1).repeat(1, 1, 3)

        # Gather Geodesic information
        window_ = 5000
        for i in range(0, nn_ids[active_view_texels].shape[0], window_):
            nn_ids_acti = nn_ids[active_view_texels].unsqueeze(0)
            index_diff = torch.arange(window_).unsqueeze(0).unsqueeze(-1).to(
                nn_ids.device)  # .repeat(nn_ids.shape[0], 1, window_)
            index_diff = index_diff * nn_ids.shape[1]

            indexes = nn_ids_acti[:, i:i + window_].clone()  # .reshape(nn_ids.shape[0], -1)
            index_diff = index_diff[:, :indexes.shape[1]]
            window_size = indexes.shape[1]

            indexes[indexes != -1] = indexes[indexes != -1] + index_diff.repeat(1, 1, self.number_of_neighboors)[
                indexes != -1]
            indexes = indexes.reshape(1, -1)

            tex_pixel_coords_act = tex_pixel_coordinates[active_view_texels][i:i + window_]
            tex_pixel_coords_act_no_zero = tex_pixel_coords_act.clone()
            tex_pixel_coords_act_no_zero[tex_pixel_coords_act == -1] = res * res

            tex_pixel_geod_info[i:i + window_size] = self.gather_geod_info(nn_ids, indexes,
                                                                           torch.cat((nn_dists.unsqueeze(-1),
                                                                                      nn_theta.unsqueeze(-1),
                                                                                      nn_gamma.unsqueeze(-1)), dim=-1),
                                                                           tex_pixel_coords_act_no_zero,
                                                                           active_view_texels,
                                                                           i, window_, window_size, res)


        tex_pixel_phi = torch.arctan2(torch.cos(tex_pixel_geod_info[..., 2]), torch.cos(tex_pixel_geod_info[..., 1]))

        tex_pixel_phi_1 = tex_pixel_geod_info[..., 0] * torch.cos(tex_pixel_phi)
        tex_pixel_phi_2 = tex_pixel_geod_info[..., 0] * torch.sin(tex_pixel_phi)

        tex_pixel_phi = torch.cat((tex_pixel_phi_1.unsqueeze(-1), tex_pixel_phi_2.unsqueeze(-1)), dim=-1)

        tex_pixel_geod_info = torch.cat((tex_pixel_geod_info[..., [0]], tex_pixel_phi), dim=-1)

        torch.save(tex_pixel_geod_info, os.path.join(out_dir, 'geod_info_'+str(res)+'.pt'))
        return tex_pixel_geod_info

    def forward_inference(self, tex_input, tex_features, tex_pixel_coordinates, nn_ids, active_texels,
                          active_view_texels, active_tokens,
                          locations, normals, nn_gamma, nn_theta, nn_dists, self_iter, out_dir):

        # Initialize inputs
        tex, tex_features, active_view_texels, loc, normals, tex_geom = self.init_inputs(tex_features,
                                                                                         active_view_texels, locations,
                                                                                         normals, tex_input)

        tex_pixel_geod_info = self.gather_view_geod_info(tex_pixel_coordinates, nn_ids, nn_theta, nn_dists, nn_gamma,
                                                         active_view_texels, out_dir=out_dir)

        # Forward Step
        tex, blend_mask, inpainting_mask = self.texture_forward(tex, tex_features, tex_geom, tex_pixel_geod_info,
                                                                active_texels, active_view_texels,
                                                                active_tokens, nn_gamma, nn_theta, nn_dists, nn_ids,
                                                                loc, normals)

        return tex, blend_mask, inpainting_mask

    def forward_inference_HD(self, tex_input, tex_features, tex_pixel_coordinates, nn_ids, active_texels,
                          active_view_texels, active_tokens,
                          locations, normals, nn_gamma, nn_theta, nn_dists, self_iter, out_dir):

        # Initialize inputs
        tex, tex_features, active_view_texels, loc, normals, tex_geom = self.init_inputs(tex_features,
                                                                                         active_view_texels,
                                                                                         locations,
                                                                                         normals,
                                                                                         tex_input)

        tex_pixel_geod_info = self.gather_view_geod_info(tex_pixel_coordinates.to(tex_input.device),
                                                         nn_ids.to(tex_input.device),
                                                         nn_theta.to(tex_input.device),
                                                         nn_dists.to(tex_input.device),
                                                         nn_gamma.to(tex_input.device),
                                                         active_view_texels.to(tex_input.device), res=1024, out_dir=out_dir)


        # Blending Step
        tex, blend_mask, inpainting_mask = self.texture_forward_HD(tex, tex_features, tex_geom, tex_pixel_geod_info,
                                                                   active_texels, active_view_texels,
                                                                   active_tokens, nn_gamma, nn_theta, nn_dists, nn_ids,
                                                                   loc, normals)

        return tex, blend_mask, inpainting_mask



    def get_data(self, batch):

        batch['mesh'] = init_mesh_ours(batch['mesh'], batch['tex'].permute(0, 3, 1, 2))

        batch['tex'] = batch['tex'].to(self.dtype)
        batch['nn_ids'] = batch['nn_ids'].long().to(self.dtype)
        batch['nn_dists'] = batch['nn_dists'].to(self.dtype)
        batch['nn_gamma'] = batch['nn_gamma'].to(self.dtype)
        batch['nn_theta'] = batch['nn_theta'].to(self.dtype)
        batch['gen_images'] = batch['gen_images'].to(self.dtype)
        batch['cam_params'] = batch['cameras_dict']

        return batch

    def validation_step(self, batch, index):
        self.eval()

        batch = self.get_data(batch)
        if self.global_rank == 0:
            camera_params = self.create_camera_params(batch['cam_params'])

            # Render Views Normals and
            images_set, renderer = render_depth_normals_tex_with_uvs_normals(batch['mesh'], batch['tex'],
                                                                             batch['tex_hd'], camera_params,
                                                                             self.config.render_size,
                                                                             1, self.cross_attention_window,
                                                                             batch['mesh'].device,
                                                                             gen_images=batch['gen_images'])

            first_iter = True

            tex_real, tex_real_hd, tex, tex_features, active_texels, active_view_texels, active_tokens, tex_pixel_coordinates = self.get_input(
                batch['tex'],
                batch['tex_hd'],
                images_set['tex_features'],
                images_set['active_texels'],
                images_set['uv_loc'],
                images_set['uv_normals'],
                self.in_channels,
                self.dtype,
                first_iter=first_iter,
                train=False)

            nn_ids, nn_gamma, nn_theta, nn_dists = self.get_nn_maps(batch['nn_texels_id'], batch['nn_ids'],
                                                                    batch['nn_gamma'], batch['nn_theta'],
                                                                    batch['nn_dists'])

            print('=====> \n', torch.sum(active_texels))
            # Forward pass
            tex_blend, mask_blend, inpainting_mask = self.forward_inference(tex, tex_features, tex_pixel_coordinates,
                                                                            nn_ids, active_texels,
                                                                            active_view_texels, active_tokens,
                                                                            images_set['uv_loc'],
                                                                            images_set['uv_normals'],
                                                                            nn_gamma,
                                                                            nn_theta,
                                                                            nn_dists, 1)

            tex_mean = self.mean_tex(tex_features, batch['tex']).unsqueeze(1)

            tex_mean_display = torch.clip(tex_mean.reshape(batch['tex'].shape), min=0, max=1)
            tex_real_display = torch.clip(tex_real.reshape(batch['tex'].shape), min=0, max=1)
            tex_blend_display = torch.clip(tex_blend.reshape(batch['tex'].shape), min=0, max=1)

            store_masks(active_texels, mask_blend, mask_blend, batch['tex'], self.current_epoch, index,
                        curr_mask=None, out_dir=self.log_dir)

            images_gen = images_set['gen_img']

            # tex_synth_display_sm = torch.nn.functional.interpolate(tex_synth_display.permute(0,3,1,2), scale_factor=0.25).permute(0,2,3,1)

            tex_display = torch.clamp(
                torch.cat((tex_real_display, tex_mean_display, tex_blend_display), dim=2), min=0,
                max=1)

            render_multi_view(batch['mesh'], tex_blend_display, tex_mean_display, tex_display, renderer,
                              self.current_epoch,
                              index, images_gen, 512, out_dir=self.log_dir)

            # Calculate and log validation loss
            loss_val = self.l1Loss(tex_blend[active_view_texels].squeeze(), tex_real[active_view_texels].squeeze())
            self.log("blend_loss_val", loss_val, on_epoch=True, prog_bar=True)

        self.train()

    def on_validation_epoch_end(self):
        if self.global_rank == 0:
            # Create the directory
            os.makedirs(os.path.join(self.log_dir, 'checkpoints'), exist_ok=True)

            self.store_model(os.path.join(self.log_dir, 'checkpoints', 'latest_'
                                          + str(self.global_step).zfill(10) + '.pt'))

    def store_model(self, dir):
        torch.save(self.state_dict(), dir)

    def save_mesh(self):
        if self.config.texture_type == 'latent':
            texture = self.guidance.decode_latent_texture(self.texture_mesh.texture.permute(0, 3, 1, 2))
        else:
            texture = self.texture_mesh.texture.permute(0, 3, 1, 2)
        save_obj(
            'meshes/mesh.obj',
            verts=self.texture_mesh.mesh.verts_packed(),
            faces=self.texture_mesh.mesh.faces_packed(),
            # decimal_places=5,
            verts_uvs=self.texture_mesh.mesh.textures.verts_uvs_padded()[0],
            faces_uvs=self.texture_mesh.mesh.textures.faces_uvs_padded()[0],
            texture_map=texture[0].permute(1, 2, 0)
        )

    def create_camera_params(self, camera_params):

        if camera_params is None:
            num_views = torch.randint(3, 7, (1,))

            camera_params = {}

            camera_params["dist"] = [torch.rand(1) * 2 + 1.5 for _ in range(num_views)]
            camera_params["dist"] = [torch.rand(1) * 2 + 1.5 for _ in range(num_views)]
            camera_params["elev"] = [torch.rand(1) * 5 + 20 for _ in range(num_views)]

            rand_azim = torch.rand(1) * 360
            camera_params["azim"] = [rand_azim + ((360 / (num_views)) * i)
                                     for i in range(num_views)]

        else:

            camera_params = camera_params

        return camera_params

    def mean_tex(self, tex_features, tex):

        # mask = tex_features[..., 3] !=-1
        mask = torch.logical_and(  # torch.logical_and(torch.sum(tex_features[..., :3] != 1, dim=-1)>0,
            torch.logical_and(tex_features[..., -1] != -1, tex_features[..., -1] != 0),  # ),
            torch.logical_and(tex_features[..., -2] != -1, tex_features[..., -2] != 0))

        tex_features[torch.logical_not(mask), [-1]] = -1

        # Mean & Weighted
        tex_mean = tex_features[:, :, :, :3]
        tex_minus_one = tex_features[:, :, :, [-1]]

        tex_mean[torch.logical_not(mask)] = torch.tensor([0.0, 0.0, 0.0]).to(tex_mean.device).to(tex_mean.dtype)

        tex_mean = torch.sum(tex_mean, dim=-2) / (torch.sum(mask != 0, dim=-1).unsqueeze(-1).repeat(1, 1, 3) + 0.000001)
        tex_mean[torch.isnan(tex_mean)] = 0.0

        return tex_mean