import os
import torch

import cv2

import numpy as np

from PIL import Image

from torchvision import transforms
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRendererWithFragments,
    MeshRasterizer,
)

from pytorch3d.renderer import TexturesUV
from torchvision.utils import save_image
from torchvision.io import write_video

from pytorch3d.structures import Meshes

import torch.nn.functional as F


# customized
import sys
sys.path.append(".")

from src.helper.camera_helper import init_camera
from src.helper.shading_helper import (
    BlendParams,
    init_soft_phong_shader
)
from einops import rearrange, pack, unpack, repeat, reduce
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes

def init_renderer(camera, shader, image_size, faces_per_pixel):
    raster_settings = RasterizationSettings(image_size=image_size,
                                            faces_per_pixel=faces_per_pixel,
                                            # max_faces_per_bin=1000,
                                            # max_faces_per_bin=1000,
                                            bin_size=-1)
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings

        ),
        shader=shader
    )

    return renderer

def phong_normal_shading(meshes, fragments) -> torch.Tensor:
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)

    faces_normals = vertex_normals[faces]
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )

    return pixel_normals

def normals_view_shading(meshes, fragments, camera_center, tex_size):

    meshes = meshes.extend(len(camera_center) // len(meshes)).to('cuda')
    faces = meshes.faces_padded() # (F, 3)
    vertex_normals = meshes.verts_normals_packed() # (V, 3)

    faces_normals = vertex_normals[faces]
    vertices = meshes.verts_packed() # (V, 3)
    face_positions = vertices[faces]

    view_directions = camera_center.reshape(camera_center.shape[0], 1, 1, 3).to('cuda') - face_positions

    # Shapes (F, 3, 3)
    view_directions = (view_directions / (torch.norm(view_directions, dim=-1).unsqueeze(-1)+0.000001))#.unsqueeze(-1)

    faces_normals = (faces_normals / (torch.norm(faces_normals, dim=-1).unsqueeze(-1)+0.000001))#.unsqueeze(-2)

    dot_similarity = faces_normals.unsqueeze(-2) @ view_directions.unsqueeze(-1)
    dot_similarity = dot_similarity.squeeze()

    # Interpolate similarity view
    pixel_similarity = interpolate_face_attributes(
        fragments.pix_to_face.to('cuda'), fragments.bary_coords.to('cuda'), torch.reshape(dot_similarity.to('cuda'), (-1, 1, 3)).permute(0,2,1)
    ).squeeze(-1)

    # Interpolate Normals view
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face.to('cuda'), fragments.bary_coords.to('cuda'), (torch.reshape(vertex_normals[faces].to('cuda'), (-1, 3, 3)) + 1) / 2
    )

    # Interpolate XYZ view
    pixel_xyz = interpolate_face_attributes(
        fragments.pix_to_face.to('cuda'), fragments.bary_coords.to('cuda'), (torch.reshape(vertices[faces].to('cuda'), (-1, 3, 3)) + 1) / 2
    ).squeeze(-2)

    uvs_vertex = meshes[0].textures.verts_uvs_padded()
    uvs_vertex[..., 0] = 1 - uvs_vertex[..., 0]
    uvs_vertex = torch.concat(((uvs_vertex * 2) - 1, torch.ones(1, meshes[0].textures.verts_uvs_padded().shape[1], 1).to('cuda')),
                              dim=-1)

    # Rasterize texel locations
    face_idxs, zbuf, barycentric_coords, dists = rasterize_meshes(
        Meshes(uvs_vertex, meshes[0].textures.faces_uvs_padded()), image_size=tex_size,
        faces_per_pixel=1, bin_size=-1)
    locations = meshes[0].verts_packed()[meshes[0].faces_packed()]

    uv_loc = interpolate_face_attributes(
        face_idxs, barycentric_coords, (locations.to('cuda')+ 1) / 2
    ).squeeze().unsqueeze(0)

    normals = meshes[0].verts_normals_packed()[meshes[0].faces_packed()]
    uv_normal = interpolate_face_attributes(
        face_idxs, barycentric_coords, (normals.to('cuda')+ 1) / 2
    ).squeeze().unsqueeze(0)

    # print('pixel similarity: ', torch.min(pixel_similarity), torch.max(pixel_similarity))
    # print('pixel normals: ', torch.min(pixel_normals), torch.max(pixel_normals))
    # print('pixel XYZ: ', torch.min(pixel_xyz), torch.max(pixel_xyz))
    # print('uv loc: ', torch.min(uv_loc), torch.max(uv_loc))
    # print('uv normals: ', torch.min(uv_normal), torch.max(uv_normal))

    return (pixel_similarity, pixel_normals.squeeze(-2), pixel_xyz, uv_loc, uv_normal)

def get_relative_depth_map_training(fragments, pad_value=0):
    absolute_depth = fragments.zbuf[..., 0] # B, H, W
    no_depth = -1
    if (absolute_depth != no_depth).any():
        depth_min, depth_max = absolute_depth[absolute_depth != no_depth].min(), absolute_depth[absolute_depth != no_depth].max()
        target_min, target_max = 20/255., 255/255.

        depth_value = absolute_depth[absolute_depth != no_depth]
        depth_value = depth_max - depth_value # reverse values

        depth_value /= (depth_max - depth_min)
        depth_value = depth_value * (target_max - target_min) + target_min

        relative_depth = absolute_depth.clone()
        relative_depth[absolute_depth != no_depth] = depth_value
        relative_depth[absolute_depth == no_depth] = pad_value # not completely black
    else:
        relative_depth = absolute_depth.clone()
        relative_depth[absolute_depth == no_depth] = pad_value

    return relative_depth

def get_uv_coordinates(mesh, fragments):

    xyzs = mesh.verts_padded() # (N, V, 3)
    faces = mesh.faces_padded() # (N, F, 3)

    faces_uvs = mesh.textures.faces_uvs_padded()
    verts_uvs = mesh.textures.verts_uvs_padded()

    # NOTE Meshes are replicated in batch. Taking the first one is enough.
    batch_size, _, _, _ = fragments.pix_to_face.shape
    xyzs, faces, faces_uvs, verts_uvs = xyzs[0], faces[0], faces_uvs[0], verts_uvs[0]
    faces_coords = verts_uvs[faces_uvs] # (F, 3, 2)

    # replicate the coordinates as batch
    faces_coords = faces_coords.repeat(batch_size, 1, 1)

    invalid_mask = fragments.pix_to_face == -1

    target_coords = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_coords.to(fragments.pix_to_face.device)
    ) # (N, H, W, 1, 3)

    _, H, W, K, _ = target_coords.shape
    # target_coords[invalid_mask] = 0
    assert K == 1 # pixel_per_faces should be 1
    target_coords = target_coords.squeeze(3) # (N, H, W, 2)

    return target_coords

def get_view_texture(mesh, fragments, encode=True):

    uv_coords = (get_uv_coordinates(mesh, fragments))

    return uv_coords

def store_masks(all_texels, inpaint_texels, blend_texels, tex, eval_iter, iter_, curr_mask=None, out_dir='eval'):

    # from torchvision.utils import save_image
    B,W,H,_ = tex.shape
    unseen_texels = inpaint_texels.reshape(B, W, H, 1)
    un_seen_img = unseen_texels#.permute(0, 3, 1, 2)
    active_img = all_texels.reshape(unseen_texels.shape)#.permute(0, 3, 1, 2)
    view_img = blend_texels.reshape(unseen_texels.shape)#.permute(0, 3, 1, 2)

    out_dir = os.path.join(out_dir, 'eval')
    # if not os.path.exists(out_dir):
    # Create the directory
    os.makedirs(out_dir, exist_ok=True)

    # if not os.path.exists(os.path.join(out_dir, str(eval_iter).zfill(5))):
        # Create the directory
    os.makedirs(os.path.join(out_dir, str(eval_iter).zfill(5)), exist_ok=True)

    # if not os.path.exists(os.path.join(out_dir, str(eval_iter).zfill(5), 'masks')):
        # Create the directory
    os.makedirs(os.path.join(out_dir, str(eval_iter).zfill(5), 'masks'), exist_ok=True)

    # if not os.path.exists(os.path.join(out_dir, str(eval_iter).zfill(5), 'rgb')):
        # Create the directory
        # os.makedirs(os.path.join(out_dir, str(eval_iter).zfill(5), 'rgb'))

    if curr_mask is not None:
        save_image(curr_mask.reshape(B, W, H, 1).permute(0, 3, 1, 2).float(),  os.path.join(out_dir, str(eval_iter).zfill(5),
                                                                                            'masks', 'curr_tex'+str(iter_).zfill(5)+'.png'))

    rgb_mask = torch.zeros(un_seen_img.shape).repeat(1,1,1,3).to(active_img.device)

    rgb_mask[active_img.squeeze(-1)] = torch.Tensor([1.,0.,0.]).to(rgb_mask.device)
    rgb_mask[un_seen_img.squeeze(-1)] = torch.Tensor([0.,1.,0.]).to(rgb_mask.device)
    rgb_mask[view_img.squeeze(-1)] = torch.Tensor([0.,0.,1.]).to(rgb_mask.device)

    save_image(rgb_mask.permute(0,3,1,2),  os.path.join(out_dir, str(eval_iter).zfill(5), 'masks', 'rgb_tex'+str(iter_).zfill(5)+'.png'))

from pytorch3d.io import save_obj
def render_multi_inference(mesh, tex_synth, renderer, images_train, image_size, out_dir='eval', loop_idx=0):

    mesh.textures = TexturesUV(
        maps=torch.clip(tex_synth, min=0, max=1).float(),  # B, H, W, C
        faces_uvs=[mesh.textures.faces_uvs_padded()[0]],
        verts_uvs=[mesh.textures.verts_uvs_padded()[0]],
        sampling_mode="bilinear",
    )

    save_obj(os.path.join(out_dir[:-5],'mesh.obj'), mesh.verts_packed(), mesh.faces_packed(),
             # normals=mesh.verts_normals_packed(),
             # faces_normals_idx=mesh.faces_normals_padded()[0],
             verts_uvs=mesh.textures.verts_uvs_padded()[0],
             faces_uvs=mesh.textures.faces_uvs_padded()[0],
             texture_map=torch.clip(tex_synth[0], min=0, max=1).float())

    views = 12
    camera_params = {}
    camera_params["dist"] = [1.5 for _ in range(views)]
    camera_params["elev"] = [22 for _ in range(views)]
    rand_azim = torch.rand(1)
    camera_params["azim"] = [((360 / views) * i)
                             for i in range(views)]

    cameras = init_camera(camera_params, image_size, tex_synth.device)

    images, frag = renderer(mesh.extend(views), cameras=cameras)

    images = images[...,:3]
    mask = (frag.zbuf[...,0] != -1).unsqueeze(-1).repeat(1,1,1,images.shape[-1])

    images = images * mask

    out_dir = os.path.join(out_dir)

    write_video(os.path.join(out_dir+'_rgb_'+str(loop_idx)+'.mp4'), (images.cpu()*255).byte(), fps=3)
    save_image(images_train.permute(0,3,1,2), os.path.join(out_dir+'_inputs_'+str(loop_idx)+'.png'))
    save_image(tex_synth.permute(0,3,1,2), os.path.join(out_dir+'_texture_'+str(loop_idx)+'.png'))

    return images, mesh

# def render_multi_view_eval(args, mesh, renderer, device='cpu', multi_view=6):
#
#
#     camera_params = {}
#     camera_params["dist"] = [1.3 for _ in range(multi_view)]
#     camera_params["elev"] = [18 for _ in range(multi_view)]
#     # rand_azim = torch.rand(1)
#     camera_params["azim"] = [((360 / multi_view) * i)
#                              for i in range(multi_view)]
#
#     cameras = init_camera(camera_params, args.render_size, device).to('cuda')
#
#     mesh = mesh.to(device)
#     # image range 0-1 (B H W C)
#     images, frag = renderer(mesh.extend(multi_view), cameras=cameras)
#
#     images = images[..., :3]
#
#     return images

def render_multi_view(mesh, tex_synth, tex_mean, display, renderer, eval_iter, iter_, images_train,
                      image_size, out_dir='eval', multi_view=6):

    # for idx, mesh in enumerate(meshes):
    mesh.textures = TexturesUV(
        # maps=torch.clip(tex_synth, min=0, max=1).float()[...,:3],  # B, H, W, C
        maps=torch.clip(tex_synth.float(), min=0, max=1),  # B, H, W, C
        faces_uvs=mesh.textures.faces_uvs_padded(),
        verts_uvs=mesh.textures.verts_uvs_padded(),
        sampling_mode="bilinear",
    )

    camera_params = {}
    camera_params["dist"] = [1.5 for _ in range(multi_view)] * (len(mesh))
    camera_params["elev"] = [25 for _ in range(multi_view)] * (len(mesh))
    # rand_azim = torch.rand(1)
    camera_params["azim"] = [((360 / multi_view) * i)
                             for i in range(multi_view)] * (len(mesh))

    cameras = init_camera(camera_params, image_size, tex_synth.device)

    # image range 0-1 (B H W C)
    images, frag = renderer(mesh.extend(multi_view), cameras=cameras)

    images = images[...,:3]
    mask = (frag.zbuf[...,0] != -1).unsqueeze(-1).repeat(1,1,1,images.shape[-1])

    images = images * mask
    tex_mean[torch.isnan(tex_mean)] = 0.0
    mesh.textures = TexturesUV(
        maps=tex_mean.float(),  # B, H, W, C
        # maps=torch.clip(tex_mean, min=0, max=1).float()[..., :3],  # B, H, W, C
        faces_uvs=mesh.textures.faces_uvs_padded(),
        verts_uvs=mesh.textures.verts_uvs_padded(),
        sampling_mode="bilinear",
        # align_corners=False
    )
    images_bp, frag = renderer(mesh.extend(multi_view), cameras=cameras)

    images_bp = images_bp[..., :3]
    mask = (frag.zbuf[..., 0] != -1).unsqueeze(-1).repeat(1, 1, 1, images_bp.shape[-1])

    images_bp = images_bp * mask

    out_dir = os.path.join(out_dir, 'eval')
    # if not os.path.exists(out_dir):
    # Create the directory
    os.makedirs(out_dir, exist_ok=True)

    # if not os.path.exists(os.path.join(out_dir, str(eval_iter).zfill(5))):
        # Create the directory
    os.makedirs(os.path.join(out_dir, str(eval_iter).zfill(5)), exist_ok=True)

    # if not os.path.exists(os.path.join(out_dir, str(eval_iter).zfill(5), 'uvs')):
        # Create the directory
    os.makedirs(os.path.join(out_dir, str(eval_iter).zfill(5), 'uvs'), exist_ok=True)

    # if not os.path.exists(os.path.join(out_dir, str(eval_iter).zfill(5), 'rgb')):
        # Create the directory
    os.makedirs(os.path.join(out_dir, str(eval_iter).zfill(5), 'rgb'), exist_ok=True)

    # if not os.path.exists(os.path.join(out_dir, str(eval_iter).zfill(5), 'simple')):
        # Create the directory
    os.makedirs(os.path.join(out_dir, str(eval_iter).zfill(5), 'simple'), exist_ok=True)

    # if not os.path.exists(os.path.join(out_dir, str(eval_iter).zfill(5), 'inputs')):
        # Create the directory
    os.makedirs(os.path.join(out_dir, str(eval_iter).zfill(5), 'inputs'), exist_ok=True)

    os.makedirs(os.path.join(out_dir, str(eval_iter).zfill(5), 'uvs_final'), exist_ok=True)


    save_image(images.permute(0,3,1,2), os.path.join(out_dir,
                                                str(eval_iter).zfill(5), 'rgb', 'renders_'+str(iter_).zfill(5)+'.png'))
    save_image(images_bp.permute(0, 3, 1, 2), os.path.join(out_dir,
                                                        str(eval_iter).zfill(5), 'simple',
                                                        'renders_' + str(iter_).zfill(5) + '.png'))

    # print('dispaly shape: ', display.shape)
    save_image(display[...,:3].permute(0,3,1,2), os.path.join(out_dir,
                                                str(eval_iter).zfill(5), 'uvs', 'uvs_'+str(iter_).zfill(5)+'.png'))


    save_image(torch.clip(tex_synth.float(), min=0, max=1).permute(0,3,1,2), os.path.join(out_dir,
                                                str(eval_iter).zfill(5), 'uvs_final', 'uvs_'+str(iter_).zfill(5)+'.png'))

    save_image(images_train[...,:3].permute(0,3,1,2), os.path.join(out_dir,
                                                str(eval_iter).zfill(5), 'inputs', 'inputs_'+str(iter_).zfill(5)+'.png'))

@torch.no_grad()
def backproject_features_with_uvs(uv_coords, tex, feature_maps, window_size, device, blend_tex_size=256):
    V, _, _, D = feature_maps.shape
    B, H, W, _ = tex.shape

    uv_coords = uv_coords.reshape(V, -1, uv_coords.shape[-1])
    blend_size = blend_tex_size
    tex_features = -(torch.ones((B, blend_size, blend_size, (D+1) * window_size ** 2))).to(device)

    # texture_locations_y, texture_locations_x = get_all_4_locations(
    #     (1 - uv_coords[:, :, 1]).reshape(-1) * (blend_size - 1),
    #     uv_coords[:, :, 0].reshape(-1) * (blend_size - 1), uv_coords.shape[0]
        # (1 - uv_coords[:, :, 1]).reshape(-1) * (tex.shape[1] - 1),
        # uv_coords[:, :, 0].reshape(-1) * (tex.shape[1] - 1), uv_coords.shape[0]
    # )

    texture_locations_y_single, texture_locations_x_single = get_all_1_locations(
        (1 - uv_coords[:, :, 1]).reshape(-1) * (tex.shape[1] - 1),
        uv_coords[:, :, 0].reshape(-1) * (tex.shape[1] - 1), uv_coords.shape[0]
    )

    texture_locations_y = texture_locations_y.reshape(uv_coords.shape[0], -1)
    texture_locations_x = texture_locations_x.reshape(uv_coords.shape[0], -1)

    texture_locations_ids = texture_locations_y_single*256+texture_locations_x_single

    feature_maps = torch.cat((feature_maps, texture_locations_ids.reshape(texture_locations_ids.shape[0], 1024, 1024, 1)), dim=-1)

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
    tex_features = tex_features.reshape(tex_features.shape[0], tex_features.shape[1], -1, D+1)
    tex_features = tex_features.permute(2, 0, 1, 3)

    return tex_features



@torch.no_grad()
def  backproject_features(uv_coords, tex, feature_maps, window_size,  device, blend_tex_size=256):

    V, _, _, D = feature_maps.shape
    B, H, W, _ = tex.shape

    uv_coords = uv_coords.reshape(V, -1, uv_coords.shape[-1])
    blend_size = blend_tex_size
    tex_features = -(torch.ones((B, blend_size, blend_size, D*window_size**2))).to(device)

    # texture_locations_y, texture_locations_x = get_all_4_locations(
    #     (1 - uv_coords[:, :, 1]).reshape(-1) * (blend_size - 1),
    #     uv_coords[:, :, 0].reshape(-1) * (blend_size - 1), uv_coords.shape[0]
        # (1 - uv_coords[:, :, 1]).reshape(-1) * (tex.shape[1] - 1),
        # uv_coords[:, :, 0].reshape(-1) * (tex.shape[1] - 1), uv_coords.shape[0]
    # )

    texture_locations_y, texture_locations_x = get_all_1_locations(
        (1 - uv_coords[:, :, 1]).reshape(-1) * (tex.shape[1] - 1),
        uv_coords[:, :, 0].reshape(-1) * (tex.shape[1] - 1), uv_coords.shape[0]
    )

    texture_locations_y = texture_locations_y.reshape(uv_coords.shape[0], -1)
    texture_locations_x = texture_locations_x.reshape(uv_coords.shape[0], -1)

    windowmize_kernel = torch.zeros(feature_maps.shape[-1] * window_size ** 2, feature_maps.shape[-1], window_size, window_size)
    filter_count = 0
    for i in range(window_size ** 2):
        for c in range(feature_maps.shape[-1]):
            windowmize_kernel[filter_count, c, i // window_size, i % window_size] = 1
            filter_count+=1

    feature_maps.requires_grad = False
    with torch.no_grad():
        feature_maps = torch.nn.functional.conv2d(feature_maps.permute(0,3,1,2), windowmize_kernel.to(feature_maps.device),
                                              stride=1, padding='same')

    _, D_N, _, _ = feature_maps.shape

    feature_maps = feature_maps.permute(0,2,3,1).reshape(V, -1, D_N)

    # feature_maps = torch.cat([feature_maps, feature_maps, feature_maps, feature_maps], dim=1)
    # feature_maps = feature_maps.repeat(1,4,1)

    tex_features = tex_features.repeat(V // B,1,1,1)
    f_size = feature_maps.shape[1]
    # print('texture loc y: ', torch.min(texture_locations_y), torch.max(texture_locations_y))
    texture_locations_y = torch.clip(texture_locations_y, min=0, max=blend_size-1)
    texture_locations_x = torch.clip(texture_locations_x, min=0, max=blend_size-1)

    # Fix this
    for b in range(tex_features.shape[0]):
        # for co_i in range(4):
        #     tex_features[b, texture_locations_y[b, co_i*f_size : (co_i+1)*f_size],
        #     texture_locations_x[b, co_i*f_size : (co_i+1)*f_size], :] = feature_maps[b]

        tex_features[b, texture_locations_y[b, 0:f_size],
        texture_locations_x[b, 0:f_size], :] = feature_maps[b]


    tex_features = tex_features.permute(1,2,0,3)
    tex_features = tex_features.reshape(tex_features.shape[0], tex_features.shape[1], -1, D)
    tex_features = tex_features.permute(2,0,1,3)

    return tex_features

def decode_latent_texture(vae, inputs, use_patches=False):
    outputs = 1 / vae.config.scaling_factor * inputs

    if use_patches:
        # assert guidance.config.latent_texture_size % guidance.config.decode_texture_size == 0
        batch_size = inputs.shape[0]
        latent_texture_size = inputs.shape[-1]
        decode_texture_size = inputs.shape[-1] // 4
        num_iter_x = latent_texture_size // decode_texture_size
        num_iter_y = latent_texture_size // decode_texture_size
        patch_stride = decode_texture_size
        decoded_stride = decode_texture_size * 8
        decoded_size = latent_texture_size * 8
        decoded_texture = torch.zeros(batch_size, 3, decoded_size, decoded_size).to(inputs.device)

        for x in range(num_iter_x):
            for y in range(num_iter_y):
                patch = outputs[:, :, x * patch_stride:(x + 1) * patch_stride,
                        y * patch_stride:(y + 1) * patch_stride]
                patch = vae.decode(patch.contiguous()).sample  # B, 3, H, W

                decoded_texture[:, :, x * decoded_stride:(x + 1) * decoded_stride,
                y * decoded_stride:(y + 1) * decoded_stride] = patch

        outputs = (decoded_texture / 2 + 0.5).clamp(0, 1)

    else:
        outputs = vae.decode(outputs.contiguous()).sample  # B, 3, H, W
        outputs = (outputs / 2 + 0.5).clamp(0, 1)

    return outputs

def get_all_4_locations(values_y, values_x, num_view=1):

    values_y = values_y.reshape((num_view, -1))
    values_x = values_x.reshape((num_view, -1))

    y_0 = torch.floor(values_y)
    y_1 = torch.ceil(values_y)
    x_0 = torch.floor(values_x)
    x_1 = torch.ceil(values_x)
    # print('shapes: ', y_0.shape)
    # print('coords: ', values_x[values_x!=0].shape, values_y[values_x!=0].shape)
    # print('coords: ', values_x[values_x!=0], values_y[values_x!=0])
    # print('x: ', y_0[values_x!=0], y_1[values_x!=0])
    # print('y: ', x_0[values_x!=0], x_1[values_x!=0])
    return (torch.cat([y_0, y_0, y_1, y_1], 1).long().reshape(-1, y_0.shape[-1]),
            torch.cat([x_0, x_1, x_0, x_1], 1).long().reshape(-1, y_0.shape[-1]))

def get_all_1_locations(values_y, values_x, num_view=1):

    values_y = values_y.reshape((num_view, -1))
    values_x = values_x.reshape((num_view, -1))

    y = torch.round(values_y)
    x = torch.round(values_x)

    return (torch.cat([y], 1).long().reshape(-1, y.shape[-1]),
            torch.cat([x], 1).long().reshape(-1, y.shape[-1]))

def render_depth_normals_tex_with_uvs_normals(mesh, tex, tex_hd, camera_params, image_size, faces_per_pixel, window_size,
                             device, gen_images=None, up=True, blend_tex_size=256):

    with (torch.no_grad()):
        images_set = {}

        cameras = init_camera(camera_params, image_size, device).to(mesh.device)

        # render the view
        renderer = init_renderer(cameras,
            shader=init_soft_phong_shader(
                camera=cameras,
                # blend_params=BlendParams(),
                device=device),
            image_size=image_size,
            faces_per_pixel=faces_per_pixel
        ).to(mesh.device)

        _, fragments = renderer(mesh.extend(len(cameras) // len(mesh)))
        # xyz_global = xyz_global[..., :3]
        if gen_images is not None:
            images_set['gen_img'] = gen_images
        else:
            images_set['gen_img'] = torch.zeros(1,4,3,512,512)

        # depth range 0-1 (B H W C)
        relative_depth = get_relative_depth_map_training(fragments).unsqueeze(-1)
        images_set['depth'] = relative_depth.detach()
        # print('relative depth shape: ', images_set['depth'].shape)
        # save_image(images_set['depth'].permute(0,3,1,2), 'depth_test.png')

        # normal dot view 0-1 (B H W C)
        normals_view, normals_global, xyz_global, uv_loc_1024, uv_normal_1024, uv_loc_256, uv_normal_256 = normals_view_shading(mesh, fragments, cameras.get_camera_center())

        # print('normal views shape: ', normals_view.shape, torch.min(normals_view), torch.max(normals_view))
        images_set['normals_view'] = normals_view.detach()
        images_set['uv_loc'] = uv_loc_256
        images_set['uv_normals'] = uv_normal_256

        images_set['active_texels'] = torch.logical_not(
            torch.logical_or(
                torch.logical_or(
                    torch.all(uv_normal_256 == 0, dim=-1),
                    torch.all(tex.to('cuda') == 0, dim=-1)),
                torch.all(uv_loc_256 == 0, dim=-1))
            ).reshape(1,-1, 1)

        images_set['uv_loc_1024'] = uv_loc_1024
        images_set['uv_normal_1024'] = uv_normal_1024

        images_set['active_texels_1024'] = torch.logical_not(torch.logical_or(
            torch.logical_or(
            torch.all(uv_normal_1024 == 0, dim=-1),
            torch.all(tex_hd.to('cuda') == 0, dim=-1)
            ),
            torch.all(uv_loc_1024 == 0, dim=-1)
            )).reshape(1,-1, 1)

        # texture_map = mesh.textures._maps_padded
        uvs_coords = get_view_texture(mesh, fragments)

        # images_set['features'] = features.detach()
        images_set['gen_img'] = rearrange(images_set['gen_img'], 'b v c h w -> (b v) c h w').permute(0, 2, 3, 1)
        if up:
            gen_imgs_hd = torch.nn.functional.interpolate(images_set['gen_img'].permute(0,3,1,2), scale_factor=2).permute(0,2,3,1)
        else:
            gen_imgs_hd = images_set['gen_img']#.permute(0,2,3,1)

        feature_map = torch.cat((gen_imgs_hd.to('cuda'), xyz_global.to('cuda'),
                                 normals_global.to('cuda'), relative_depth.to('cuda'),
                                 normals_view.to('cuda')), dim=-1)

        # backproject pixels to texel locations
        tex_features = backproject_features_with_uvs(uvs_coords.to('cuda'), tex.to('cuda'),
                                                     feature_map.to('cuda'), window_size, 'cuda', blend_tex_size=blend_tex_size)

        # print('tex features shape : ', tex_features.shape)

        images_set['tex_features'] = tex_features

    return images_set, renderer

def render_depth_normals_tex_with_uvs(mesh, tex, tex_hd, camera_params, image_size, faces_per_pixel, window_size,
                             device, gen_images=None, up=True, blend_tex_size=256):

    with (torch.no_grad()):
        images_set = {}

        cameras = init_camera(camera_params, image_size, device).to(mesh.device)

        # render the view
        renderer = init_renderer(cameras,
            shader=init_soft_phong_shader(
                camera=cameras,
                # blend_params=BlendParams(),
                device=device),
            image_size=image_size,
            faces_per_pixel=faces_per_pixel
        ).to(mesh.device)

        _, fragments = renderer(mesh.extend(len(cameras) // len(mesh)))
        # xyz_global = xyz_global[..., :3]
        if gen_images is not None:
            images_set['gen_img'] = gen_images
        else:
            images_set['gen_img'] = torch.zeros(1,4,3,512,512)

        # depth range 0-1 (B H W C)
        relative_depth = get_relative_depth_map_training(fragments).unsqueeze(-1)
        images_set['depth'] = relative_depth.detach()
        # print('relative depth shape: ', images_set['depth'].shape)
        # save_image(images_set['depth'].permute(0,3,1,2), 'depth_test.png')

        # normal dot view 0-1 (B H W C)
        normals_view, normals_global, xyz_global, uv_loc_1024, uv_normal_1024, uv_loc_256, uv_normal_256 = normals_view_shading(mesh, fragments, cameras.get_camera_center())

        # print('normal views shape: ', normals_view.shape, torch.min(normals_view), torch.max(normals_view))
        images_set['normals_view'] = normals_view.detach()
        images_set['uv_loc'] = uv_loc_256
        images_set['uv_normals'] = uv_normal_256

        images_set['active_texels'] = torch.logical_not(
            torch.logical_or(
                torch.logical_or(
                    torch.all(uv_normal_256 == 0, dim=-1),
                    torch.all(tex.to('cuda') == 0, dim=-1)),
                torch.all(uv_loc_256 == 0, dim=-1))
            ).reshape(1,-1, 1)

        images_set['uv_loc_1024'] = uv_loc_1024
        images_set['uv_normal_1024'] = uv_normal_1024

        images_set['active_texels_1024'] = torch.logical_not(torch.logical_or(
            torch.logical_or(
            torch.all(uv_normal_1024 == 0, dim=-1),
            torch.all(tex_hd.to('cuda') == 0, dim=-1)
            ),
            torch.all(uv_loc_1024 == 0, dim=-1)
            )).reshape(1,-1, 1)

        # texture_map = mesh.textures._maps_padded
        uvs_coords = get_view_texture(mesh, fragments)

        # images_set['features'] = features.detach()
        images_set['gen_img'] = rearrange(images_set['gen_img'], 'b v c h w -> (b v) c h w').permute(0, 2, 3, 1)
        if up:
            gen_imgs_hd = torch.nn.functional.interpolate(images_set['gen_img'].permute(0,3,1,2), scale_factor=2).permute(0,2,3,1)
        else:
            gen_imgs_hd = images_set['gen_img']#.permute(0,2,3,1)


        feature_map = torch.cat((gen_imgs_hd.to('cuda'), xyz_global.to('cuda'), relative_depth.to('cuda'), normals_view.to('cuda')), dim=-1)

        # backproject pixels to texel locations
        tex_features = backproject_features_with_uvs(uvs_coords.to('cuda'), tex.to('cuda'), feature_map.to('cuda'), window_size, 'cuda', blend_tex_size=blend_tex_size)

        images_set['tex_features'] = tex_features

    return images_set, renderer

def render_depth_normals_tex(mesh, tex, tex_hd, camera_params, image_size, faces_per_pixel, window_size,
                             device, gen_images=None, up=True, blend_tex_size=256):

    with (torch.no_grad()):
        images_set = {}

        cameras = init_camera(camera_params, image_size, device).to(mesh.device)

        # render the view
        renderer = init_renderer(cameras,
            shader=init_soft_phong_shader(
                camera=cameras,
                # blend_params=BlendParams(),
                device=device),
            image_size=image_size,
            faces_per_pixel=faces_per_pixel
        ).to(mesh.device)

        _, fragments = renderer(mesh.extend(len(cameras) // len(mesh)))
        # xyz_global = xyz_global[..., :3]
        if gen_images is not None:
            images_set['gen_img'] = gen_images
        else:
            images_set['gen_img'] = torch.zeros(1,4,3,512,512)

        # depth range 0-1 (B H W C)
        relative_depth = get_relative_depth_map_training(fragments).unsqueeze(-1)
        images_set['depth'] = relative_depth.detach()
        # print('relative depth shape: ', images_set['depth'].shape)
        # save_image(images_set['depth'].permute(0,3,1,2), 'depth_test.png')

        # normal dot view 0-1 (B H W C)
        normals_view, normals_global, xyz_global, uv_loc_1024, uv_normal_1024, uv_loc_256, uv_normal_256 = normals_view_shading(mesh, fragments, cameras.get_camera_center())

        # print('normal views shape: ', normals_view.shape, torch.min(normals_view), torch.max(normals_view))
        images_set['normals_view'] = normals_view.detach()
        images_set['uv_loc'] = uv_loc_256
        images_set['uv_normals'] = uv_normal_256

        images_set['active_texels'] = torch.logical_not(
            torch.logical_or(
                torch.logical_or(
                    torch.all(uv_normal_256 == 0, dim=-1),
                    torch.all(tex.to('cuda') == 0, dim=-1)),
                torch.all(uv_loc_256 == 0, dim=-1))
            ).reshape(1,-1, 1)

        images_set['uv_loc_1024'] = uv_loc_1024
        images_set['uv_normal_1024'] = uv_normal_1024

        images_set['active_texels_1024'] = torch.logical_not(torch.logical_or(
            torch.logical_or(
            torch.all(uv_normal_1024 == 0, dim=-1),
            torch.all(tex_hd.to('cuda') == 0, dim=-1)
            ),
            torch.all(uv_loc_1024 == 0, dim=-1)
            )).reshape(1,-1, 1)

        # texture_map = mesh.textures._maps_padded
        uvs_coords = get_view_texture(mesh, fragments)

        # images_set['features'] = features.detach()
        images_set['gen_img'] = rearrange(images_set['gen_img'], 'b v c h w -> (b v) c h w').permute(0, 2, 3, 1)
        if up:
            gen_imgs_hd = torch.nn.functional.interpolate(images_set['gen_img'].permute(0,3,1,2), scale_factor=2).permute(0,2,3,1)
        else:
            gen_imgs_hd = images_set['gen_img']#.permute(0,2,3,1)

        # print(gen_imgs_hd.to('cuda').shape, xyz_global.to('cuda').shape, relative_depth.to('cuda').shape, normals_view.to('cuda').shape)
        # save_image(gen_imgs_hd.permute(0,3,1,2), '1.png')
        # save_image(xyz_global.permute(0,3,1,2), '2.png')
        # save_image(relative_depth.permute(0,3,1,2), '3.png')
        # save_image(normals_view.permute(0,3,1,2), '4.png')
        # return
        print(gen_imgs_hd.to('cuda').shape, xyz_global.to('cuda').shape, relative_depth.to('cuda').shape, normals_view.to('cuda').shape)
        feature_map = torch.cat((gen_imgs_hd.to('cuda'), xyz_global.to('cuda'), relative_depth.to('cuda'), normals_view.to('cuda')), dim=-1)
        # feature_map = torch.cat((images_set['gen_img'], zbuf, xyz_global, relative_depth, normals_view), dim=-1)

        # backproject pixels to texel locations
        tex_features = backproject_features(uvs_coords.to('cuda'), tex.to('cuda'), feature_map.to('cuda'), window_size, 'cuda', blend_tex_size=blend_tex_size)

        print('tex features shape : ', tex_features.shape)

        images_set['tex_features'] = tex_features

    return images_set, renderer
    # return depth, normals, rgb
@torch.no_grad()
def render(mesh, renderer, faces_semantics=None, pad_value=10, return_dict=False):
    def phong_normal_shading(meshes, fragments) -> torch.Tensor:
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)

        faces_normals = vertex_normals[faces]

        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_normals
        )

        return pixel_normals

    def get_semantic_shading(fragments, faces_semantics):
        # pixel_semantic = interpolate_face_attributes(
        #     fragments.pix_to_face, fragments.bary_coords, faces_semantics
        # ).int()

        pixel_semantic = faces_semantics[fragments.pix_to_face][:, :, :, :, 0, 0]

        # NOTE background will be -1 in fragments.pix_to_face
        # those pixels will be pad with -1
        pixel_semantic[fragments.pix_to_face == -1] = -1

        return pixel_semantic

    def similarity_shading(meshes, fragments):
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        vertices = meshes.verts_packed()  # (V, 3)
        face_positions = vertices[faces]
        view_directions = torch.nn.functional.normalize((renderer.shader.cameras.get_camera_center().reshape(1, 1, 3) - face_positions), p=2, dim=2)
        cosine_similarity = torch.nn.CosineSimilarity(dim=2)(faces_normals, view_directions)
        pixel_similarity = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, cosine_similarity.unsqueeze(-1)
        )

        return pixel_similarity

    def get_relative_depth_map(fragments, pad_value=pad_value):
        absolute_depth = fragments.zbuf[..., 0] # B, H, W
        no_depth = -1

        depth_min, depth_max = absolute_depth[absolute_depth != no_depth].min(), absolute_depth[absolute_depth != no_depth].max()

        target_min, target_max = 50, 255

        depth_value = absolute_depth[absolute_depth != no_depth]
        depth_value = depth_max - depth_value # reverse values

        depth_value /= (depth_max - depth_min)
        depth_value = depth_value * (target_max - target_min) + target_min

        relative_depth = absolute_depth.clone()
        relative_depth[absolute_depth != no_depth] = depth_value
        relative_depth[absolute_depth == no_depth] = pad_value # not completely black

        return relative_depth

    images, fragments = renderer(mesh)
    normal_maps = phong_normal_shading(mesh, fragments).squeeze(-2)

    similarity_maps = similarity_shading(mesh, fragments).squeeze(-2) # -1 - 1
    depth_maps = get_relative_depth_map(fragments)

    # normalize similarity mask to 0 - 1
    similarity_maps = torch.abs(similarity_maps) # 0 - 1

    if faces_semantics is not None:
        # index semantic labels from 1
        # 0 is invalid
        semantic_maps = get_semantic_shading(fragments, faces_semantics)
        semantic_maps += 1
    else:
        semantic_maps = torch.zeros_like(similarity_maps)

    # HACK erode, eliminate isolated dots
    non_zero_similarity = (similarity_maps > 0).float()
    non_zero_similarity = (non_zero_similarity * 255.).cpu().numpy().astype(np.uint8)[0]
    non_zero_similarity = cv2.erode(non_zero_similarity, kernel=np.ones((3, 3), np.uint8), iterations=2)
    non_zero_similarity = torch.from_numpy(non_zero_similarity).to(similarity_maps.device).unsqueeze(0) / 255.
    similarity_maps = non_zero_similarity.unsqueeze(-1) * similarity_maps
    # save_image(similarity_maps.permute(0,3,1,2), 'similarity_maps_2.png')

    if return_dict:
        return {
            "images": images,
            "depth_maps": depth_maps,
            "normal_maps": normal_maps,
            "similarity_maps": similarity_maps,
            "semantic_maps": semantic_maps,
            "fragments": fragments,
        }
    else:
        return images, depth_maps, normal_maps, similarity_maps, semantic_maps, fragments


@torch.no_grad()
def render_one_view(mesh, cameras,
    image_size, faces_per_pixel, faces_semantics,
    device):

    # cameras = init_camera(camera_params, image_size, device)

    # render the view
    renderer = init_renderer(cameras,
        shader=init_soft_phong_shader(
            camera=cameras,
            blend_params=BlendParams(),
            device=device),
        image_size=image_size,
        faces_per_pixel=faces_per_pixel
    )

    (
        init_images_tensor,
        depth_maps_tensor,
        normal_maps_tensor,
        similarity_tensor,
        semantic_maps_tensor,
        fragments
    ) = render(mesh, renderer, faces_semantics)

    return (
        cameras, renderer,
        init_images_tensor, normal_maps_tensor, similarity_tensor, depth_maps_tensor,
        semantic_maps_tensor,
        fragments
    )



@torch.no_grad()
def check_visible_faces(mesh, fragments):
    pix_to_face = fragments.pix_to_face

    # Indices of unique visible faces
    visible_map = pix_to_face.unique()  # (num_visible_faces)

    return visible_map

