# -*- coding: utf-8 -*-

import time
import numpy as np
import cv2
import torch
from PIL import Image, PngImagePlugin

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from diffusers.schedulers import EulerAncestralDiscreteScheduler


class inpaintControlNet():
    def __init__(self, config, torch_dtype=torch.float16):
        controlnet_list = []
        for cnet_unit in config.controlnet_units:
            controlnet = ControlNetModel.from_pretrained(cnet_unit.controlnet_key, torch_dtype=torch_dtype)
            controlnet_list.append(controlnet)
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(config.sd_model_key, controlnet=controlnet_list,
                                                                        torch_dtype=torch_dtype).to("cuda")
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
        :param config: task config for img2img
        :return:
        """
        w, h = config.width, config.height

        # input
        image = Image.open(config.image_path)
        image = image.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
        mask = Image.open(config.mask_path)
        mask = mask.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
        image = self.fill_image(image, mask)

        # condition
        control_img = []
        conditioning_scales = []
        for cnet_unit in config.controlnet_units:
            if cnet_unit.preprocessor == 'none':
                condition_image = Image.open(cnet_unit.condition_image_path)
                condition_image = condition_image.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
            elif cnet_unit.preprocessor == 'inpaint_global_harmonious':
                condition_image = self.make_inpaint_condition(image, mask)
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
        res_image = self.pipe(config.prompt,
                              negative_prompt=config.negative_prompt,
                              image=image,
                              mask_image=mask,
                              control_image=control_img,
                              ip_adapter_image=ip_adapter_image,
                              height=h,
                              width=w,
                              num_images_per_prompt=config.num_images_per_prompt,
                              guidance_scale=config.guidance_scale,
                              num_inference_steps=config.num_inference_steps,
                              strength=config.denoising_strength,
                              generator=generator,
                              controlnet_conditioning_scale=conditioning_scales).images
        return res_image

    def fill_image(self, image, image_mask, inpaintRadius=3):
        image = np.array(image.convert("RGB"))
        image_mask = (np.array(image_mask.convert("L"))).astype(np.uint8)
        filled_image = cv2.inpaint(image, image_mask, inpaintRadius, cv2.INPAINT_TELEA)

        res_img = Image.fromarray(np.clip(filled_image, 0, 255).astype(np.uint8))
        return res_img

    def make_inpaint_condition(self, image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image

from tqdm import tqdm
import os
from torchvision.utils import save_image
def inpaint_viewpoint(sd_cfg, latents, cnet, depth_im, outdir, iter, inpaint_view_ids=['0', '1']):
    # projection
    # print(f"Project inpaint view {inpaint_view_ids}...")
    # inpaint_view_ids = ['0', '1']
    # t = '_'+inpaint_view_ids[0]+'_'+inpaint_view_ids[1]+'_'
    t = '_'+str(iter)+'_'

    inpaint_used_key = ["image", "depth", "uncolored_mask"]
    # for i, one_batch_id in tqdm(enumerate(inpaint_view_ids)):
    #
    #     one_batch_img = []
    #     one_batch_img.append(latents.permute(0,3,1,2))
    #     one_batch_img.append(depth_im)
    #     # print('latent: ', latents.shape, torch.tensor([1., 51 / 255, 1.]).shape)
    #     print((latents - torch.tensor([1., 51 / 255, 1.])).shape)
    #     masked_img_mesh = torch.sum(latents - torch.tensor([1., 51 / 255, 1.]).to(latents.device), dim=-1)
    #     one_batch_img.append((masked_img_mesh < 0.1).float().unsqueeze(0))
        # print(latents.shape, depth_im.shape, masked_img_mesh.shape)
        # exit()
    # for i, img in enumerate(zip(*one_batch_img)):
    one_batch_img = []
    latents = latents.permute(0,3,1,2)
    one_batch_img.append(torch.cat((latents[[0]],latents[[1]]), dim=-1))
    one_batch_img.append(torch.cat((depth_im[[0]],depth_im[[1]]), dim=-1))
    base_color = torch.tensor([1., 51 / 255, 1.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    # print('base color shape: ', base_color.shape, one_batch_img[0].shape)
    # exit()
    # print(latents)
    masked_img_mesh = torch.sum(torch.abs(one_batch_img[0] - base_color.to(latents.device)), dim=1, keepdim=True)
    # print(masked_img_mesh.shape, (latents - base_color.to(latents.device)).shape)
    # exit()
    one_batch_img.append(masked_img_mesh<0.1)

    for i, img in enumerate(one_batch_img):

        print('imag : ', img.shape)
        # img = torch.cat(img, dim=3)
        if img.size(1) == 1:
            img = img.repeat(1, 3, 1, 1)
        # t = '_'.join(map(str, one_batch_id))
        name = inpaint_used_key[i]
        if name == "uncolored_mask":
            img[img>0] = 1.
        save_path = os.path.join(outdir, f"view_{t}_{name}.png")

        save_image(img.float(), save_path)

    # inpaint view point
    txt_cfg = sd_cfg.txt2img
    img_cfg = sd_cfg.inpaint
    copy_list = ["prompt", "negative_prompt", "seed", ]
    for k in copy_list:
        img_cfg[k] = txt_cfg[k]

    for i, one_batch_id in tqdm(enumerate(inpaint_view_ids)):
        # t = '_'.join(map(str, one_batch_id))
        rgb_path = os.path.join(outdir, f"view_{t}_{inpaint_used_key[0]}.png")
        depth_path = os.path.join(outdir, f"view_{t}_{inpaint_used_key[1]}.png")
        mask_path = os.path.join(outdir, f"view_{t}_{inpaint_used_key[2]}.png")

        # pre-processing inpaint mask: dilate
        mask = cv2.imread(mask_path)
        dilate_kernel = 10

        mask = cv2.dilate(mask, np.ones((dilate_kernel, dilate_kernel), np.uint8))
        mask_path = os.path.join(outdir, f"view_{t}_{inpaint_used_key[2]}_d{dilate_kernel}.png")
        cv2.imwrite(mask_path, mask)

        img_cfg.image_path = rgb_path
        img_cfg.mask_path =  mask_path
        img_cfg.controlnet_units[0].condition_image_path = depth_path
        images = cnet.infernece(config=img_cfg)
        for i, img in enumerate(images):
            save_path = os.path.join(outdir, f"view_{t}_rgb_inpaint_{i}.png")
            img.save(save_path)
    return images