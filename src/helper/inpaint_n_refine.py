
import sys
import argparse
import os
import cv2
from tqdm import tqdm
import torch
import torchvision
import time
import numpy as np
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf

from src.controlnet.diffusers_cnet_txt2img import txt2imgControlNet
from src.controlnet.diffusers_cnet_inpaint import inpaintControlNet
from src.controlnet.diffusers_cnet_img2img import img2imgControlNet
from paint3d.dataset import init_dataloaders
from paint3d import utils
from paint3d.models.textured_mesh import TexturedMeshModel
from src.pipeline.generate_texture import store_mesh


# from paint3d.trainer import dr_eval, forward_texturing


def UV_inpaint(sd_cfg, cnet, outdir):


    albedo_path = os.path.join(outdir, f"mesh.png")
    UV_pos_path = os.path.join(outdir, f"position_uv.png")

    # processing mask
    mask_dilated = utils.extract_bg_mask(albedo_path, dilate_kernel=2)
    mask_path = os.path.join(outdir, f"mask.png")
    cv2.imwrite(mask_path, mask_dilated)

    # UV inpaint
    p_cfg = sd_cfg.inpaint
    p_cfg.image_path = albedo_path
    p_cfg.mask_path =  mask_path
    p_cfg.controlnet_units[0].condition_image_path = UV_pos_path

    images = cnet.infernece(config=p_cfg)
    res = []
    for i, img in enumerate(images):
        save_path = os.path.join(outdir, f'UV_inpaint_res_{i}.png')
        img.save(save_path)
        res.append((img, save_path))
    return res


def UV_tile(sd_cfg, cnet, outdir,):
    print(f"rendering texture and position map")
    # mesh_model.export_mesh(outdir, export_texture_only=True)
    albedo_path = os.path.join(outdir, f"UV_inpaint_res_0.png")
    # UV_pos = mesh_model.UV_pos_render()
    UV_pos_path = os.path.join(outdir, f"position_uv.png")
    # utils.save_tensor_image(UV_pos.permute(0, 3, 1, 2), UV_pos_path)

    # UV inpaint
    p_cfg = sd_cfg.img2img
    p_cfg.image_path = albedo_path
    p_cfg.controlnet_units[0].condition_image_path = UV_pos_path
    p_cfg.controlnet_units[1].condition_image_path = albedo_path

    images = cnet.infernece(config=p_cfg)
    for i, img in enumerate(images):
        save_path = os.path.join(outdir, f'UV_tile_res_{i}.png')
        img.save(save_path)
    return images


def init_process(opt):
    outdir = opt.outdir
    os.makedirs(outdir, exist_ok=True)

    pathdir, filename = Path(opt.render_config).parent, Path(opt.render_config).stem
    sys.path.append(str(pathdir))
    render_cfg = __import__(filename, ).TrainConfig()
    utils.seed_everything(render_cfg.optim.seed)

    sd_cfg = OmegaConf.load(opt.sd_config)
    render_cfg.log.exp_path = str(outdir)

    if opt.prompt is not None:
        sd_cfg.inpaint.prompt = opt.prompt
        sd_cfg.img2img.prompt = opt.prompt
    if opt.ip_adapter_image_path is not None:
        sd_cfg.inpaint.ip_adapter_image_path = opt.ip_adapter_image_path
        sd_cfg.img2img.ip_adapter_image_path = opt.ip_adapter_image_path
    if opt.mesh_path is not None:
        render_cfg.guide.shape_path = opt.mesh_path
    if opt.texture_path is not None:
        render_cfg.guide.initial_texture = opt.texture_path
        img = Image.open(opt.texture_path)
        render_cfg.guide.texture_resolution = img.size
    return sd_cfg, render_cfg



def inpaint_and_refine(config, batch, mesh_idx, out_dir, sd_config='stable-diffusion/v2-inpainting-inference.yaml'):

    opt = {}
    opt['render_config'] = 'src/config/train_config_paint3d.py'
    opt['sd_config'] = 'src/controlnet/config/UV_based_inpaint_template.yaml'
    opt['outdir'] = out_dir
    opt['texture_path'] = os.path.join(out_dir, 'mesh.png')
    opt['prompt'] = "UV map , "+batch['text'][0]+" , high quality, best quality"
    opt['ip_adapter_image_path'] = None
    opt['mesh_path'] = os.path.join(out_dir, 'mesh.obj')

    from types import SimpleNamespace
    opt = SimpleNamespace(**opt)
    # print('opt outdir: ', opt.outdir)
    sd_cfg, render_cfg = init_process(opt)

    # ===  1. create model
    device = torch.device("cuda")
    # mesh_model = TexturedMeshModel(cfg=render_cfg, device=device,).to(device)
    # dataloaders = init_dataloaders(render_cfg, device)

    UVInpaint_cnet = inpaintControlNet(sd_cfg.inpaint)
    UVtile_cnet = img2imgControlNet(sd_cfg.img2img)

    total_start = time.time()
    # ===  2. UV inpaint
    start_t = time.time()
    UV_inpaint_res = UV_inpaint(
        sd_cfg=sd_cfg,
        cnet=UVInpaint_cnet,
        # mesh_model=mesh_model,
        outdir=opt.outdir,
    )
    print(f"UV Inpainting time: {time.time() - start_t}")


    for i, (_, init_img_path) in enumerate(UV_inpaint_res):
        # outdir = Path(opt.outdir) / f"tile_res_{i}"

        outdir = Path(opt.outdir) #/ f"tile_res_{i}"

        # outdir.mkdir(exist_ok=True)
        # ===  3. UV tile
        start_t = time.time()

        images = UV_tile(
            sd_cfg=sd_cfg,
            cnet=UVtile_cnet,
            outdir=outdir,
        )
        print(f"UV tile time: {time.time() - start_t}")

        print(f"total processed time:{time.time() - total_start}")
        torch.cuda.empty_cache()

    import torchvision.transforms as transforms
    texture = transforms.ToTensor()(images[0]).unsqueeze(0).permute(0,2,3,1)

    store_mesh(config, texture, batch, mesh_idx, dilate_flag=False)

