import os
import torch
import torchvision

from torchvision.utils import save_image
# Helper Functions
from src.helper.inference_functions import (
    normalize_mesh,
    init_mesh,
    compute_depth,
    generate_colored_views,
    render_depth_normals_tex,
    dilate_mask,
    inpaint_views)

from pytorch3d.io import save_obj


def generate_text2img_inpaint(config, batch, net, inpaint_cnet, sd_cfg, cameras, mesh_idx, texture, prev_views, iteration):

    # print('mesh: ', batch['mesh'])
    batch['mesh_struct'] = init_mesh(batch['mesh'], texture.to(batch['mesh']['verts'].device), batch['mesh']['verts'])

    # Initialize text
    sd_cfg.prompt = 'turn around, ' + batch['text'][0] + ' , high quality, high detail'

    # Compute depth images
    relative_depth, fragments, latents = compute_depth(config, batch, cameras)


    # Inpaint Views
    out_dir = os.path.join(config.out_dir, str(mesh_idx).zfill(3))
    views_colored = inpaint_views(inpaint_cnet, latents[-2:], sd_cfg, relative_depth[-2:],
                                           out_dir, iteration)


    if iteration==1:
        views_colored = torch.cat((prev_views, views_colored), dim=1)
    elif iteration==2:
        views_colored = torch.cat((prev_views, views_colored), dim=1)

    # Neural Back Projection
    texture, position_uv, textured_texel_mask = neural_backprojection(config, batch, net, views_colored, cameras, fragments, out_dir)

    return texture, position_uv, textured_texel_mask, views_colored


def generate_text2img(config, batch, net, depth_cnet, sd_cfg, cameras, mesh_idx, texture, iteration):

    # print('mesh: ', batch['mesh'])
    batch['mesh_struct'] = init_mesh(batch['mesh'], texture.to(batch['mesh']['verts'].device), batch['mesh']['verts'])

    # Initialize text
    sd_cfg.prompt = 'turn around, '+ batch['text'][0] + ' , high quality, high detail'

    # Compute depth images
    relative_depth, fragments, latents = compute_depth(config, batch, cameras)

    # Generate RGB images
    out_dir = os.path.join(config.out_dir, str(mesh_idx).zfill(3))
    views_colored = generate_colored_views(depth_cnet, latents, sd_cfg, relative_depth, out_dir, iteration)

    # Neural Back Projection
    texture, position_uv, textured_texel_mask = neural_backprojection(config, batch, net, views_colored, cameras, fragments, out_dir)

    return texture, position_uv, textured_texel_mask, views_colored



# def generate_texture_one_iteration(config, batch, net, depth_cnet, inpaint_cnet, sd_cfg, cameras, mesh_idx, iteration):



    # return texture

@torch.no_grad()
def generate_texture(config, batch, net, depth_cnet, inpaint_cnet, sd_cfg, cameras, mesh_idx, paint3D_strategy=False):

    # Initialize Mesh and scale mesh
    batch['mesh']['verts'] = normalize_mesh(batch['mesh']['verts'][0], target_scale=config.scale_mesh,
                                            mesh_dy=config.dy_mesh, mean=None).unsqueeze(0)

    texture=None
    inpaint_set = [[0,1], [0,1,2,3], [0,1,2,3,4,5]]
    for iteration in range(config.inference_iterations):
        if texture == None:
            texture = batch['tex'].permute(0, 3, 1, 2)

        # generate_texture_one_iteration(config, batch, net, depth_cnet, inpaint_cnet, sd_cfg, cameras, mesh_idx, iteration)
        if not paint3D_strategy:

            # Run text to image
            texture, position_uv, textured_texel_mask, _ = generate_text2img(config, batch, net, depth_cnet, sd_cfg.txt2img, cameras, mesh_idx, texture, iteration)

        else:

            if iteration == 0:
                texture, position_uv, textured_texel_mask, prev_views = generate_text2img(config, batch, net, depth_cnet,
                                                                              sd_cfg.txt2img, cameras[[0,1]], mesh_idx,
                                                                              texture, iteration)
            else:
                # Run Inpaint
                texture, position_uv, textured_texel_mask, prev_views =  generate_text2img_inpaint(config, batch, net, inpaint_cnet,
                                                                                       sd_cfg, cameras[inpaint_set[iteration]],
                                                                                       mesh_idx, texture, prev_views, iteration)

        texture = torch.clip(texture, min=0, max=1.)

        # print('texture shape : ', texture.shape)
        texture_og = texture.resize(1, config.tex_size, config.tex_size, 3).permute(0,3,1,2)
        if paint3D_strategy:
            texture = dilate_mask(texture_og, iter=2)
        else:
            texture = dilate_mask(texture_og, iter=4)

    # Store Mesh with Texture
    store_mesh(config, texture_og, batch, mesh_idx)

    store_mask(position_uv, textured_texel_mask.reshape(1,config.tex_size, config.tex_size, 1), os.path.join(config.out_dir, str(mesh_idx).zfill(3)))

    torch.cuda.empty_cache()
    return texture

def store_mesh(config, texture, batch, mesh_idx, dilate_flag=True):
    if dilate_flag and config.tex_size==1024:
        dilated_texture = dilate_mask(texture, iter=4)
    elif dilate_flag and config.tex_size==256:
        dilated_texture = dilate_mask(texture, iter=1)
    else:
        dilated_texture = texture

    save_obj(
        os.path.join(config.out_dir, str(mesh_idx).zfill(3), "mesh.obj"),
        verts=batch['mesh_struct'].verts_packed(),
        faces=batch['mesh_struct'].faces_packed(),
        verts_uvs=batch['mesh_struct'].textures.verts_uvs_padded()[0],
        faces_uvs=batch['mesh_struct'].textures.faces_uvs_padded()[0],
        texture_map=dilated_texture[0]
    )

def store_mask(position_uv, textured_texel_mask, dst_path):

    save_image(position_uv.permute(0,3,1,2), os.path.join(dst_path, 'position_uv.png'))
    save_image(textured_texel_mask.permute(0,3,1,2).float(), os.path.join(dst_path, 'textured_texel_mask.png'))

def unmake_grid(grid_tensor, image_size, padding=2):
    C, H_total, W_total = grid_tensor.shape
    H, W = image_size, image_size
    ncols = (W_total + padding) // (W + padding)
    nrows = (H_total + padding) // (H + padding)

    images = []
    for y in range(nrows):
        for x in range(ncols):
            top = y * (H + padding)
            left = x * (W + padding)
            patch = grid_tensor[:, top:top + H, left:left + W]
            if patch.shape[1] == H and patch.shape[2] == W:
                images.append(patch)

    return torch.stack(images)

def neural_backprojection(config, batch, net, gen_images, cameras, fragments, out_dir):

    gen_images = unmake_grid(gen_images, image_size=config.render_size, padding=0)

    texture = torch.ones(batch['tex_hd'].shape) if config.tex_size==1024 else torch.ones(batch['tex'].shape)

    # Get geometric information
    images_set = render_depth_normals_tex(batch['mesh_struct'],
                                             texture,
                                             cameras, config.render_size,1,
                                             config.cross_attention_window, 'cuda',
                                             fragments, gen_images=gen_images.unsqueeze(0),
                                             tex_size=config.tex_size, number_of_views=len(cameras))

    # Get geodesic neighbor relations
    nn_ids, nn_gamma, nn_theta, nn_dists = net.get_nn_maps(batch['nn_texels_id'], batch['nn_ids'],
                                                       batch['nn_gamma'], batch['nn_theta'],
                                                       batch['nn_dists'], res=config.tex_size)

    # Prepare inputs
    (tex_real, tex_real_hd, tex,
     tex_features, active_texels,
     active_view_texels, active_tokens, tex_pixel_coordinates) = net.get_input(
                                                                texture,
                                                                texture,
                                                                images_set['tex_features'].to('cuda'),
                                                                images_set['active_texels'].to('cuda'),
                                                                images_set['uv_loc'].to('cuda'),
                                                                images_set['uv_normal'].to('cuda'),
                                                                config.cross_attention_window,
                                                                torch.float32)

    tex, tex_features, active_view_texels, loc, normals, tex_geom = net.init_inputs(tex_features.to(tex.device),
                                                                                    active_view_texels.to(tex.device),
                                                                                    images_set['uv_loc'].to(tex.device),
                                                                                    images_set['uv_normal'].to(tex.device),
                                                                                    tex.to(tex.device))

    tex = tex.to('cuda')
    if config.tex_size == 256:
        texture, active_view_texels, valid_mask = net.forward_inference(tex.to(tex.device), tex_features.to(tex.device), tex_pixel_coordinates.to(tex.device),
                                                                  nn_ids.to(tex.device), active_texels.to(tex.device),
                                                                  active_view_texels.to(tex.device), active_tokens.to(tex.device),
                                                                  images_set['uv_loc'].to(tex.device),
                                                                  images_set['uv_normals'].to(tex.device),
                                                                  nn_gamma.to(tex.device),
                                                                  nn_theta.to(tex.device),
                                                                  nn_dists.to(tex.device), 1, out_dir)
    else:
        texture, active_view_texels, valid_mask = net.forward_inference_HD(tex.to(tex.device), tex_features.to(tex.device), tex_pixel_coordinates.to(tex.device),
                                                                  nn_ids.to(tex.device), active_texels.to(tex.device),
                                                                  active_view_texels.to(tex.device), active_tokens.to(tex.device),
                                                                  images_set['uv_loc'].to(tex.device),
                                                                  images_set['uv_normals'].to(tex.device),
                                                                  nn_gamma.to(tex.device),
                                                                  nn_theta.to(tex.device),
                                                                  nn_dists.to(tex.device), 1, out_dir)


    return texture, images_set['uv_loc'], images_set['active_texels']