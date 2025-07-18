import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image
import sys
import os

sys.path.append("./")

from pytorch3d.structures import (
    Meshes,
    join_meshes_as_scene
)

import argparse

from einops import rearrange, pack, unpack, repeat, reduce

from pytorch3d.renderer import TexturesUV

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    exit()

import json

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, \
    StableDiffusionControlNetInpaintPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# helper
from src.pipeline.generate_texture import generate_texture
from src.helper.inference_functions import  init_camera
from src.data.tex_test_dataset import TexRGBDataset
from src.controlnet.txt2img import txt2imgControlNet
from src.controlnet.txt2img_inpaint import inpaintControlNet
from src.helper.inpaint_n_refine import inpaint_and_refine

def init_config(args):
    config = OmegaConf.load(args.config)
    config.geod_neighbors = True
    config.extra_views = args.extra_views
    config.checkpoint_path = args.checkpoint_path
    config.cross_attention_window = args.cross_attention_window
    config.out_dir = args.out_dir
    return config

def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="src/config/template_eval.yaml")
    parser.add_argument("--mesh_json", type=str, default='src/config/mesh.json')
    parser.add_argument("--sd_config", type=str, default='src/controlnet/config/depth_based_inpaint_eval_template.yaml')
    # Model Parameters
    parser.add_argument("--cross_attention_window", type=int, default=3)
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/latest_weights.pt")

    # Texture Loop Parameters
    parser.add_argument("--sa_iter", type=int, default=2)
    parser.add_argument("--loop_iter", type=int, default=2)
    parser.add_argument("--extra_views", type=bool, default=False)
    parser.add_argument("--geod_neighbors", type=bool, default=True)

    # Path Directories
    parser.add_argument("--out_dir", type=str, default="",  required=True)
    parser.add_argument("--guidance_scale", type=float, default=7)

    # Rendering arguments
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--workers_per_gpu", type=int, default=11)
    parser.add_argument("--obj_name", type=str, default='mesh.obj')
    parser.add_argument("--renders_out_dir", type=str, default='')

    args = parser.parse_args()

    return args

def init_model(config, stamp):

    from src.pipeline.im2surftex import Im2SurfTexModel
    model = Im2SurfTexModel(config, stamp).to('cuda')

    return model.eval()

if __name__ == '__main__':

    # Initialize Arguments
    args = init_args()
    print('Args: ', args)

    print("=> loading config file...")
    config = init_config(args)

    print("=> loading json file..." , args.mesh_json)
    with open(args.mesh_json, 'r') as file:
        mesh_list = json.load(file)

    print('Start Generating Textures .... ')

    from torch.utils.data import DataLoader
    data = TexRGBDataset(mesh_list, config, DEVICE, type='full')
    data = DataLoader(data, batch_size=1, shuffle=False)

    stamp = 'test'

    net = init_model(config, stamp)

    # initialize D2I model
    sd_cfg = OmegaConf.load(args.sd_config)
    depth_cnet = txt2imgControlNet(sd_cfg.txt2img)
    inpaint_cnet = None
    if config.cameras_paint3D:
        inpaint_cnet = inpaintControlNet(sd_cfg.inpaint)

    dist, elev, azim = config.camera_locations

    # Initialize Cameras
    cameras = init_camera({'dist': dist, 'elev': elev, 'azim': azim}, config.render_size, DEVICE, paint3d=config.cameras_paint3D)
    for mesh_idx, batch in tqdm(enumerate(data)):

        # Generate UV textures for each shape
        generate_texture(config, batch, net, depth_cnet, inpaint_cnet, sd_cfg, cameras, batch['id'][0], config.inpaint_strategy)

        # Run Inpaint and HD module
        inpaint_and_refine(config, batch, batch['id'][0], os.path.join(config.out_dir, batch['id'][0].zfill(3)))
