import os
from datetime import datetime
import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image
import sys
import torchvision

# from src.render_script.gen_view_dataset import latents

sys.path.append("../scripts/")
# from src.data.tex_rgb_dataset import TexRGBDataset

from src.helper.shading_helper import (
    BlendParams,
    init_soft_phong_shader
)

from pytorch3d.structures import (
    Meshes,
    join_meshes_as_scene
)
from src.helper.camera_helper import init_camera_lookat

import argparse

from einops import rearrange, pack, unpack, repeat, reduce

from src.helper.render_helper import init_renderer
# from lib.mesh_helper import init_mesh_ours
from pytorch3d.renderer import TexturesUV

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    exit()

import torch.nn.functional as F

from src.helper.render_helper import (get_relative_depth_map_training, normals_view_shading,
                                      get_view_texture,get_all_1_locations, get_all_4_locations, render_multi_inference)

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, \
    StableDiffusionControlNetInpaintPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import time

from tqdm import tqdm


def get_relative_depth_map(zbuf, pad_value=0):
    absolute_depth = zbuf[..., 0]  # B, H, W
    no_depth = -1

    depth_min, depth_max = absolute_depth[absolute_depth != no_depth].min(), absolute_depth[
        absolute_depth != no_depth].max()

    # target_min, target_max = 20, 255
    target_min, target_max = 0.5, 1.

    # print('absolute depth: ', absolute_depth.shape)
    relative_depth_list = []
    # depth_value_tensor = torch.zeros(absolute_depth.shape)
    for i in range(absolute_depth.shape[0]):
        depth_value = absolute_depth[i][absolute_depth[i] != no_depth]

        # depth_value = depth_max - depth_value  # reverse values

        # depth_value /= (depth_max - depth_min)
        # depth_value = depth_value * (target_max - target_min) + target_min

        # depth_value = ((1 - target_min) * (depth_value - torch.min(depth_value)) / (
        #         torch.max(depth_value) - torch.min(depth_value))) + target_min
        # absolute_depth = zbuf[..., 0]  # B, H, W
        # no_depth = -1

        depth_min, depth_max = depth_value.min(), depth_value.max()
        target_min, target_max = 0.4, 1.

        # depth_value = absolute_depth[absolute_depth != no_depth]
        depth_value = depth_max - depth_value  # reverse values

        depth_value /= (depth_max - depth_min)
        depth_value = depth_value * (target_max - target_min) + target_min

        relative_depth = absolute_depth[i].clone()
        relative_depth[absolute_depth[i] != no_depth] = depth_value
        relative_depth[absolute_depth[i] == no_depth] = pad_value  # not completely black

        relative_depth_list.append(relative_depth)

    relative_depth = torch.stack(relative_depth_list)

    return absolute_depth, relative_depth

# Load controlnet
torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    exit()


def dilate_mask(tex, iter=4):
    for i in range(iter):
        # Expand loc texture
        tex_empty = F.max_pool2d(tex, 3, padding=1, stride=1)
        mask_fill = torch.all(tex == 0, dim=1).unsqueeze(1).repeat(1, tex_empty.shape[1], 1, 1)
        # print('shapes : ', tex.shape, mask_fill.shape, tex_empty.shape)
        tex = tex * (1 - mask_fill.float()) + mask_fill * tex_empty

    tex = tex.permute(0,2,3,1)
    mask_texels_empty_ = torch.all(tex == 0, dim=-1)
    empty_texels = tex[mask_texels_empty_]
    empty_texels[:] = 1.
    empty_texels[..., 1] = 51 / 255

    tex[mask_texels_empty_] = empty_texels

    return tex


def normalize_mesh(vertices, target_scale=1.0, mesh_dy=0.0, mean=None):
    verts = vertices
    if mean == None:
        center = verts.mean(dim=0)
    else:
        center = mean

    verts = verts - center
    scale = torch.max(torch.norm(verts, p=2, dim=1))
    verts = verts / scale
    verts *= target_scale
    verts[:, 1] += mesh_dy

    return verts


def init_mesh(batch, tex, verts):
    # print(batch['mesh']['verts'])
    mesh = Meshes(verts, batch['faces_idx'])

    if tex.shape[1] == 3:
        tex = tex.permute(0, 2, 3, 1)
    tex = tex[..., :3]

    mesh.textures = TexturesUV(
        maps=torch.clip(tex, min=0, max=1),  # B, H, W, C
        # maps=ones,  # B, H, W, C
        faces_uvs=batch['face_uvs_idx'],
        verts_uvs=batch['verts_uvs'],
        sampling_mode="bilinear",
        # align_corners=False
    )

    return mesh

def init_camera(camera_params, image_size, device, paint3d=False):

    cameras = init_camera_lookat(
        camera_params["dist"],
        camera_params["elev"],
        camera_params["azim"],
        image_size,
        device,
        paint3D=paint3d
    )

    return cameras


def render_depth_normals_tex(mesh, tex, cameras, image_size, faces_per_pixel, window_size,
                             device, fragments, gen_images=None, up=False, tex_size=256, paint3d=False, number_of_views=2):
    with (torch.no_grad()):
        images_set = {}


        images_set['gen_img'] = gen_images

        # depth range 0-1 (B H W C)
        relative_depth = get_relative_depth_map_training(fragments).unsqueeze(-1)

        images_set['depth'] = relative_depth.detach()

        # normal dot view 0-1 (B H W C)
        normals_view, normals_global, xyz_global, uv_loc, uv_normal = normals_view_shading(
            mesh, fragments, cameras.get_camera_center(), tex_size)

        images_set['uv_loc'] = uv_loc
        images_set['uv_normals'] = uv_normal
        images_set['active_texels'] = torch.logical_not(
            torch.logical_or(
                torch.logical_or(
                    torch.all(uv_normal == 0, dim=-1),
                    torch.all(tex.to('cuda') == 0, dim=-1)),
                torch.all(uv_loc == 0, dim=-1))
        ).reshape(1, -1, 1)

        images_set['uv_loc'] = uv_loc
        images_set['uv_normal'] = uv_normal

        # texture_map = mesh.textures._maps_padded
        uvs_coords = get_view_texture(mesh, fragments)


        gen_imgs_hd = rearrange(gen_images, 'b v c h w -> (b v) c h w').permute(0, 2, 3, 1)

        feature_map = torch.cat(
            (gen_imgs_hd.to('cuda'), xyz_global.to('cuda'), normals_global.to('cuda'),
             relative_depth.to('cuda'), normals_view.to('cuda')), dim=-1)

        # backproject pixels to texel locations
        tex_features = backproject_features_with_uvs(uvs_coords.to('cuda'), tex.to('cuda'), feature_map.to('cuda'), window_size,
                                            'cuda', tex_size=tex_size)

        # tex feature values: range 0-1 invalid -1
        images_set['tex_features'] = tex_features

    return images_set


@torch.no_grad()
def backproject_features(uv_coords, tex, feature_maps, window_size, device, tex_size=256):
    V, _, _, D = feature_maps.shape
    B, H, W, _ = tex.shape

    uv_coords = uv_coords.reshape(V, -1, uv_coords.shape[-1])
    blend_size = tex_size
    tex_features = -(torch.ones((B, blend_size, blend_size, D * window_size ** 2))).to(device)

    texture_locations_y, texture_locations_x = get_all_4_locations(
        (1 - uv_coords[:, :, 1]).reshape(-1) * (blend_size - 1),
        uv_coords[:, :, 0].reshape(-1) * (blend_size - 1), uv_coords.shape[0]
        # (1 - uv_coords[:, :, 1]).reshape(-1) * (tex.shape[1] - 1),
        # uv_coords[:, :, 0].reshape(-1) * (tex.shape[1] - 1), uv_coords.shape[0]
    )

    # texture_locations_y, texture_locations_x = get_all_1_locations(
    #     (1 - uv_coords[:, :, 1]).reshape(-1) * (tex.shape[1] - 1),
    #     uv_coords[:, :, 0].reshape(-1) * (tex.shape[1] - 1), uv_coords.shape[0]
    # )

    texture_locations_y = texture_locations_y.reshape(uv_coords.shape[0], -1)
    texture_locations_x = texture_locations_x.reshape(uv_coords.shape[0], -1)

    windowmize_kernel = torch.zeros(feature_maps.shape[-1] * window_size ** 2, feature_maps.shape[-1], window_size,
                                    window_size)
    filter_count = 0
    for i in range(window_size ** 2):
        for c in range(feature_maps.shape[-1]):
            windowmize_kernel[filter_count, c, i // window_size, i % window_size] = 1
            filter_count += 1

    feature_maps.requires_grad = False
    with torch.no_grad():
        feature_maps = torch.nn.functional.conv2d(feature_maps.permute(0, 3, 1, 2),
                                                  windowmize_kernel.to(feature_maps.device),
                                                  stride=1, padding='same')

    _, D_N, _, _ = feature_maps.shape

    feature_maps = feature_maps.permute(0, 2, 3, 1).reshape(V, -1, D_N)

    # feature_maps = torch.cat([feature_maps, feature_maps, feature_maps, feature_maps], dim=1)
    # feature_maps = feature_maps.repeat(1,4,1)

    tex_features = tex_features.repeat(V // B, 1, 1, 1)
    f_size = feature_maps.shape[1]

    texture_locations_y = torch.clip(texture_locations_y, min=0, max=blend_size - 1)
    texture_locations_x = torch.clip(texture_locations_x, min=0, max=blend_size - 1)

    # Fix this
    for b in range(tex_features.shape[0]):
        for co_i in range(4):
            tex_features[b, texture_locations_y[b, co_i * f_size: (co_i + 1) * f_size],
            texture_locations_x[b, co_i * f_size: (co_i + 1) * f_size], :] = feature_maps[b]

    tex_features = tex_features.permute(1, 2, 0, 3)
    tex_features = tex_features.reshape(tex_features.shape[0], tex_features.shape[1], -1, D)
    tex_features = tex_features.permute(2, 0, 1, 3)

    return tex_features

@torch.no_grad()
def backproject_features_with_uvs(uv_coords, tex, feature_maps, window_size, device, tex_size=256):
    V, _, _, D = feature_maps.shape
    B, H, W, _ = tex.shape

    uv_coords = uv_coords.reshape(V, -1, uv_coords.shape[-1])
    blend_size = tex_size
    tex_features = -(torch.ones((B, blend_size, blend_size, (D+1) * window_size ** 2))).to(device)

    # texture_locations_y, texture_locations_x = get_all_4_locations(
    #    (1 - uv_coords[:, :, 1]).reshape(-1) * (blend_size - 1),
    #    uv_coords[:, :, 0].reshape(-1) * (blend_size - 1), uv_coords.shape[0]
        # (1 - uv_coords[:, :, 1]).reshape(-1) * (tex.shape[1] - 1),
        # uv_coords[:, :, 0].reshape(-1) * (tex.shape[1] - 1), uv_coords.shape[0]
    # )

    texture_locations_y, texture_locations_x = get_all_1_locations(
        (1 - uv_coords[:, :, 1]).reshape(-1) * (tex.shape[1] - 1),
        uv_coords[:, :, 0].reshape(-1) * (tex.shape[1] - 1), uv_coords.shape[0]
    )

    texture_locations_y = texture_locations_y.reshape(uv_coords.shape[0], -1)
    texture_locations_x = texture_locations_x.reshape(uv_coords.shape[0], -1)

    texture_locations_ids = texture_locations_y+texture_locations_x

    feature_maps = torch.cat((feature_maps, texture_locations_ids.reshape(texture_locations_ids.shape[0], 512, 512, 1)), dim=-1)

    # feature_maps
    windowmize_kernel = torch.zeros(feature_maps.shape[-1] * window_size ** 2, feature_maps.shape[-1], window_size,
                                    window_size)
    filter_count = 0
    for i in range(window_size ** 2):
        for c in range(feature_maps.shape[-1]):
            windowmize_kernel[filter_count, c, i // window_size, i % window_size] = 1
            filter_count += 1

    feature_maps.requires_grad = False
    with torch.no_grad():
        feature_maps = torch.nn.functional.conv2d(feature_maps.permute(0, 3, 1, 2),
                                                  windowmize_kernel.to(feature_maps.device),
                                                  stride=1, padding='same')

    _, D_N, _, _ = feature_maps.shape

    feature_maps = feature_maps.permute(0, 2, 3, 1).reshape(V, -1, D_N)
    tex_features = tex_features.repeat(V // B, 1, 1, 1)
    f_size = feature_maps.shape[1]
    #print('texture loc y: ', torch.min(texture_locations_y), torch.max(texture_locations_y))
    texture_locations_y = torch.clip(texture_locations_y, min=0, max=blend_size - 1)
    texture_locations_x = torch.clip(texture_locations_x, min=0, max=blend_size - 1)


    # Fix this
    for b in range(tex_features.shape[0]):
        for co_i in range(1):
            tex_features[b, texture_locations_y[b, co_i * f_size: (co_i + 1) * f_size],
            texture_locations_x[b, co_i * f_size: (co_i + 1) * f_size], :] = feature_maps[b]


    tex_features = tex_features.permute(1, 2, 0, 3)
    tex_features = tex_features.reshape(tex_features.shape[0], tex_features.shape[1], -1, D+1)
    tex_features = tex_features.permute(2, 0, 1, 3)

    return tex_features

def erode_mask(mask, kernel_size=3):
    """
    Erodes a binary mask using a convolution operation.

    Args:
        mask (torch.Tensor): Binary mask of shape [N, C, H, W] with values 0 or 1.
        kernel_size (int): Size of the square kernel for erosion (default: 3).

    Returns:
        torch.Tensor: Eroded binary mask of the same shape as input.
    """
    # Ensure mask is float32 for convolution
    mask = mask.float()

    # Create the kernel (structuring element) with ones
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32, device=mask.device)

    # Apply padding to maintain output size
    padding = kernel_size // 2

    # Perform convolution with the kernel
    eroded_mask = F.conv2d(mask, kernel, padding=padding)

    # Calculate the threshold: All values in the kernel neighborhood must be 1
    threshold = 5

    # Erode: Check if all values in the neighborhood are 1
    eroded_mask = (eroded_mask > threshold).float()

    return eroded_mask

def compute_depth(config, batch, cameras):

    renderer = init_renderer(cameras,
                             shader=init_soft_phong_shader(
                                 camera=cameras,
                                 # blend_params=BlendParams(),
                                 device=DEVICE),
                             image_size=config.render_size,
                             faces_per_pixel=1
                             ).to(DEVICE)

    latents, fragments = renderer(batch['mesh_struct'].to('cuda').extend(len(cameras)), camera=cameras)  # image: (N, H, W, C)

    _, relative_depth = get_relative_depth_map(fragments.zbuf)

    return relative_depth, fragments, latents[...,:3]

def inpaint_views(inpaint_cnet, latents, sd_cfg, depth_images, output_dir, iteration):
    from src.controlnet.txt2img_inpaint import inpaint_viewpoint


    # Define deinoise strength
    images = inpaint_viewpoint(
                sd_cfg=sd_cfg,
                latents=latents,
                cnet=inpaint_cnet,
                depth_im=depth_images.unsqueeze(1),
                outdir=output_dir,
                iter=iteration)[0]

    # Convert to tensor
    images = transforms.ToTensor()(images)

    return images

def generate_colored_views(depth_cnet, latents, sd_cfg, depth_images, output_dir, iteration):
    from src.controlnet.txt2img import gen_init_view

    # Define deinoise strength
    images = gen_init_view(
                p_cfg=sd_cfg,
                latents=latents,
                cnet=depth_cnet,
                depth_im=depth_images.unsqueeze(1),
                outdir=output_dir,
                iter=iteration,
                save_outputs=True)[0]

    # Convert to tensor
    images = transforms.ToTensor()(images)

    return images


