# -*- coding: utf-8 -*-

import time
import numpy as np
import cv2
import torch
from PIL import Image, PngImagePlugin

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from diffusers.schedulers import EulerAncestralDiscreteScheduler
import torchvision.transforms as T
from PIL import ImageFilter
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
        seed = int(time.time()) if config.seed == -1 else config.seed
        self.generator = torch.manual_seed(int(seed))


    def infernece(self, config, image, mask):
        """
        :param config: task config for img2img
        :return:
        """
        w, h = config.width, config.height

        # input
        # image = Image.open(config.image_path)
        # image = image.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
        # mask = Image.open(config.mask_path)
        # mask = mask.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
        image = T.ToPILImage()(image[0].permute(2, 0, 1))
        image = image.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
        image.save('test_input_image.png')
        mask = T.ToPILImage()(torch.logical_not(mask).reshape(1, 256, 256).float())
        mask = mask.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
        mask.save('mask_resize.png')
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

    def predict_t0(self, config, image, mask, UV_positions, cross_attention_out, denoise_strength, prompt, inference_steps=20):
        """
        :param config: task config for img2img
        :return:
        """
        w, h = config.width, config.height

        # print('predict_t0 shapes and min max: ', image.shape, mask.shape, UV_positions.shape, cross_attention_out.shape)
        # print('predict_t0 shapes and min max: ', torch.min(image), torch.max(image), torch.min(mask), torch.max(mask),
        #       torch.min(UV_positions), torch.max(UV_positions), torch.min(cross_attention_out), torch.max(cross_attention_out))

        # input
        # image = Image.open(config.image_path)
        image = T.ToPILImage()(image[0].permute(2,0,1))
        image = image.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
        image.save('input_image_test.png')
        mask = T.ToPILImage()(torch.logical_not(mask).reshape(1,256,256).float())
        # mask = T.ToPILImage()(mask.reshape(1,256,256).float())
        mask = mask.filter(ImageFilter.MaxFilter(3))
        mask = mask.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
        mask.save('mask_test.png')

        image = self.fill_image(image, mask)

        # condition
        control_img = []
        conditioning_scales = []
        for idxs, cnet_unit in enumerate(config.controlnet_units):
            print('idxs: ', idxs, cnet_unit.preprocessor)
            if cnet_unit.preprocessor == 'none':
                # condition_image = Image.open(cnet_unit.condition_image_path)
                condition_image = T.ToPILImage()(UV_positions[0].permute(2, 0, 1))
                condition_image = condition_image.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
                condition_image.save('condition_image.png')
            elif cnet_unit.preprocessor == 'inpaint_global_harmonious':
                # condition_image = T.ToPILImage()(cross_attention_out[0].permute(2, 0, 1))
                condition_image = T.ToPILImage()(cross_attention_out[0].permute(2, 0, 1))
                condition_image = self.make_inpaint_condition(condition_image, mask)
                from torchvision.utils import save_image
                print(condition_image.shape)
                save_image(condition_image , 'condition_image_inpaint.png')
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
        res_image = self.pipe(prompt,
                              negative_prompt=config.negative_prompt,
                              image=image,
                              mask_image=mask,
                              control_image=control_img,
                              ip_adapter_image=ip_adapter_image,
                              height=h,
                              width=w,
                              num_images_per_prompt=config.num_images_per_prompt,
                              guidance_scale=config.guidance_scale,
                              num_inference_steps=inference_steps,
                              strength=denoise_strength,
                              generator=generator,
                              controlnet_conditioning_scale=conditioning_scales,
                              output_type='rgb').images
        return res_image

    def encode_blented_latent(self, config, image, mask):
        """
        :param config: task config for img2img
        :return:
        """
        w, h = config.width, config.height

        # print('predict_t0 shapes and min max: ', image.shape, mask.shape, UV_positions.shape, cross_attention_out.shape)
        # print('predict_t0 shapes and min max: ', torch.min(image), torch.max(image), torch.min(mask), torch.max(mask),
        #       torch.min(UV_positions), torch.max(UV_positions), torch.min(cross_attention_out), torch.max(cross_attention_out))

        # input
        # image = Image.open(config.image_path)
        image = T.ToPILImage()(image[0].permute(2,0,1))
        image = image.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
        # image.save('test_input_image.png')
        mask = T.ToPILImage()(torch.logical_not(mask).reshape(1,256,256).float())
        mask = mask.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
        # mask.save('mask_resize.png')
        image = T.ToTensor()(self.fill_image(image, mask)).to('cuda')
        # print('image min max: ', image.shape,  torch.min(image), torch.max(image))
        blended_latents = self.pipe._encode_vae_image(((image.unsqueeze(0) * 2) - 1).to(torch.float16),
                                                   self.generator)

        return blended_latents


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