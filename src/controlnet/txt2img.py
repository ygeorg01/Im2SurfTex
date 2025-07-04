# -*- coding: utf-8 -*-

import time
import torch
from PIL import Image, PngImagePlugin

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler
import os
import torchvision
import cv2
import numpy as np


class txt2imgControlNet():
    def __init__(self, config, torch_dtype=torch.float16):
        controlnet_list = []
        for cnet_unit in config.controlnet_units:
            controlnet = ControlNetModel.from_pretrained(cnet_unit.controlnet_key, torch_dtype=torch_dtype)
            controlnet_list.append(controlnet)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(config.sd_model_key, controlnet=controlnet_list,
                                                                 torch_dtype=torch_dtype).to("cuda")
        # pipe = StableDiffusionControlNetPipeline.from_pretrained(config.sd_model_key, controlnet=controlnet_list,
        #                                                          torch_dtype=torch_dtype).to("cuda")
        if config.ip_adapter_image_path:
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.safetensors")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(config.sd_model_key, subfolder="scheduler")
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        # pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        self.pipe = pipe


    def infernece(self, config):
        """
        :param config: task config for txt2img
        :return:
        """
        w, h = config.width, config.height

        # condition
        control_img = []
        conditioning_scales = []
        for cnet_unit in config.controlnet_units:
            if cnet_unit.preprocessor == 'none':
                condition_image = Image.open(cnet_unit.condition_image_path)
                condition_image = condition_image.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
            else:
                raise NotImplementedError
            control_img.append(condition_image)
            conditioning_scales.append(cnet_unit.weight)
        conditioning_scales = conditioning_scales[0] if len(conditioning_scales) == 1 else conditioning_scales

        # ip-adapter
        ip_adapter_image = None
        if config.ip_adapter_image_path:
            ip_adapter_image = Image.open(config.ip_adapter_image_path)
            print("using ip adapter...")

        seed = int(time.time()) if config.seed == -1 else config.seed
        generator = torch.manual_seed(int(seed))
        print('strength: ', config.strength)

        latent = Image.open(config.latent_image_path)
        latent = latent.resize(size=(w, h), resample=Image.Resampling.BICUBIC)

        res_image = self.pipe(config.prompt,
                              negative_prompt=config.negative_prompt,
                              image=latent,
                              control_image=control_img,
                              ip_adapter_image=ip_adapter_image,
                              height=h,
                              width=w,
                              num_images_per_prompt=config.num_images_per_prompt,
                              guidance_scale=config.guidance_scale,
                              num_inference_steps=config.num_inference_steps,
                              strength=config.strength,
                              generator=generator,
                              controlnet_conditioning_scale=conditioning_scales).images
        return res_image

# Extra Functions
def gen_init_view(p_cfg, latents, cnet, depth_im, outdir, iter, save_outputs=False):

    init_depth_map = depth_im.repeat(1, 3, 1, 1)

    if init_depth_map.shape[0] == 2:
        nrows = 2
    else:
        nrows = init_depth_map.shape[0] // 3

    print(init_depth_map.shape, nrows)


    init_depth_map = torchvision.utils.make_grid(init_depth_map, nrow=nrows, padding=0)

    # The generated output changing based on the rendered views
    p_cfg.height, p_cfg.width = init_depth_map.shape[-2:]

    save_path = os.path.join(outdir, f"init_depth_render.png")

    save_tensor_image(init_depth_map.unsqueeze(0), save_path=save_path)

    # post-processing depth，dilate
    depth_dilated = dilate_depth_outline(save_path, iters=5, dilate_kernel=3)
    save_path = os.path.join(outdir, f"init_depth_dilated.png")
    cv2.imwrite(save_path, depth_dilated)

    # p_cfg = sd_cfg.txt2img
    p_cfg.controlnet_units[0].condition_image_path = save_path

    save_path = os.path.join(outdir, f"latents_"+str(iter).zfill(2)+".png")
    p_cfg.latent_image_path = save_path

    latents = torchvision.utils.make_grid(latents.permute(0,3,1,2), nrow=nrows, padding=0)

    save_tensor_image(latents.unsqueeze(0), save_path=save_path)

    if iter==0:
        p_cfg.strength = 1.
    else:
        p_cfg.strength = 0.5


    images = cnet.infernece(config=p_cfg)
    if save_outputs:
        img = images[0]
        save_path = os.path.join(outdir, f'init-img-{iter}.png')
        img.save(save_path)

    return images

def save_tensor_image(tensor: torch.Tensor, save_path: str):
    if len(os.path.dirname(save_path)) > 0 and not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)  # [1, c, h, w]-->[c, h, w]
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    tensor = tensor.permute(1, 2, 0).detach().cpu().numpy()  # [c, h, w]-->[h, w, c]
    Image.fromarray((tensor * 255).astype(np.uint8)).save(save_path)


def dilate_depth_outline(path, iters=5, dilate_kernel=3):
    ori_img = cv2.imread(path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

    img = ori_img
    for i in range(iters):
        _, mask = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        mask = mask / 255

        img_dilate = cv2.dilate(img, np.ones((dilate_kernel, dilate_kernel), np.uint8))

        img = (mask * img + (1 - mask) * img_dilate).astype(np.uint8)
    return img