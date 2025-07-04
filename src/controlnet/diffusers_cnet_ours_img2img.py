# -*- coding: utf-8 -*-

import time
import torch
from PIL import Image, PngImagePlugin

from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from diffusers import EulerAncestralDiscreteScheduler
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
import PIL
import numpy as np
from diffusers.utils import (
    PIL_INTERPOLATION,
    replace_example_docstring,
)

import torchvision.transforms as T
from PIL import ImageFilter
import cv2

class img2imgControlNet():
    def __init__(self, config, torch_dtype=torch.float16):
        controlnet_list = []
        for cnet_unit in config.controlnet_units:
            controlnet = ControlNetModel.from_pretrained(cnet_unit.controlnet_key, torch_dtype=torch_dtype)
            controlnet_list.append(controlnet)
        print('controlnet_list: ', config)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(config.sd_model_key, controlnet=controlnet_list,
                                                                 torch_dtype=torch_dtype).to("cuda")
        if config.ip_adapter_image_path:
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.safetensors")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(config.sd_model_key, subfolder="scheduler")
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        # pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        self.pipe = pipe
        self.generator = torch.manual_seed(int(config.seed))

    def infernece(self, config):
        """
        :param config: task config for img2img
        :return:
        """
        w, h = config.width, config.height

        # input
        image = Image.open(config.image_path)
        image = image.resize(size=(w, h), resample=Image.Resampling.BICUBIC)

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

        res_image = self.pipe(config.prompt,
                              negative_prompt=config.negative_prompt,
                              image=image,
                              control_image=control_img,
                              ip_adapter_image=ip_adapter_image,
                              height=h,
                              width=w,
                              num_images_per_prompt=config.num_images_per_prompt,
                              guidance_scale=config.guidance_scale,
                              num_inference_steps=config.num_inference_steps,
                              strength=config.denoising_strength,
                              generator=self.generator,
                              controlnet_conditioning_scale=conditioning_scales).images
        return res_image

    def infernece_custom(self, config):
        w, h = config.width, config.height

        # input
        image = Image.open(config.image_path)
        image = image.resize(size=(w, h), resample=Image.Resampling.BICUBIC)

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

        # Set Inputs
        prompt = config.prompt

        image = image
        controlnet_conditioning_image = control_img
        strength = config.denoising_strength
        height = h
        width = w
        num_inference_steps = config.num_inference_steps
        guidance_scale = 7.5
        negative_prompt = config.negative_prompt
        num_images_per_prompt= config.num_images_per_prompt
        eta = 0.0
        generator= generator
        latents = None,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        output_type = "pil",
        return_dict = True,
        callback = None,
        callback_steps: int = 1,
        cross_attention_kwargs = None,
        controlnet_conditioning_scale = conditioning_scales,
        controlnet_guidance_start = 0.0,
        controlnet_guidance_end = 1.0,

        # Pipeline
        # 0. Default height and width to unet
        height, width = self.pipe._default_height_width(height, width, controlnet_conditioning_image)

        # 1. Check inputs. Raise error if not correct
        self.pipe.check_inputs(
            prompt,
            image,
            controlnet_conditioning_image,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            strength,
            controlnet_guidance_start,
            controlnet_guidance_end,
            controlnet_conditioning_scale,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipe._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(self.pipe.controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(self.pipe.controlnet.nets)

        # 3. Encode input prompt
        prompt_embeds = self.pipe._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare image, and controlnet_conditioning_image
        image = prepare_image(image)

        # condition image(s)
        if isinstance(self.pipe.controlnet, ControlNetModel):
            controlnet_conditioning_image = prepare_controlnet_conditioning_image(
                controlnet_conditioning_image=controlnet_conditioning_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=self.pipe.controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
        elif isinstance(self.pipe.controlnet, MultiControlNetModel):
            controlnet_conditioning_images = []

            for image_ in controlnet_conditioning_image:
                image_ = prepare_controlnet_conditioning_image(
                    controlnet_conditioning_image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=self.pipe.controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )

                controlnet_conditioning_images.append(image_)

            controlnet_conditioning_image = controlnet_conditioning_images
        else:
            assert False

        # 5. Prepare timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.pipe.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        if latents is None:
            latents = self.pipe.prepare_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
            )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.pipe.scheduler.order
        with self.pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

                # compute the percentage of total steps we are at
                current_sampling_percent = i / len(timesteps)

                if (
                    current_sampling_percent < controlnet_guidance_start
                    or current_sampling_percent > controlnet_guidance_end
                ):
                    # do not apply the controlnet
                    down_block_res_samples = None
                    mid_block_res_sample = None
                else:
                    # apply the controlnet
                    down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_cond=controlnet_conditioning_image,
                        conditioning_scale=controlnet_conditioning_scale,
                        return_dict=False,
                    )

                # predict the noise residual
                noise_pred = self.pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                latents_dict = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)

                latents_or = torch.nn.functional.interpolate(latents_dict.pred_original_sample.to(torch.float16),
                                                             (256, 256))

                print('latents 1 or shape: ', latents_or.shape)

                latents_or = torch.nn.functional.interpolate(latents_or, (64, 64))

                # Load geodesic nn
                print('latents 2 or shape: ', latents_or.shape)

                # Add Noise
                sigma = self.pipe.scheduler.sigmas[self.pipe.scheduler.step_index - 1]
                sigma_from = self.pipe.scheduler.sigmas[self.pipe.scheduler.step_index - 1]
                sigma_to = self.pipe.scheduler.sigmas[self.pipe.scheduler.step_index]
                sigma_up = (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5
                sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5

                # 2. Convert to an ODE derivative
                derivative = (latents - latents_or) / sigma

                dt = sigma_down - sigma

                prev_sample = latents + derivative * dt

                device = noise_pred.device
                from diffusers.utils.torch_utils import randn_tensor
                noise = randn_tensor(noise_pred.shape, dtype=noise_pred.dtype, device=device, generator=generator)

                prev_sample = prev_sample + noise * sigma_up

                # Cast sample back to model compatible dtype
                latents = prev_sample.to(noise_pred.dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.pipe.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.pipe.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self.pipe, "final_offload_hook") and self.pipe.final_offload_hook is not None:
            self.pipe.unet.to("cpu")
            self.pipe.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.pipe.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.pipe.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.pipe.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.pipe.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.pipe.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self.pipe, "final_offload_hook") and self.pipe.final_offload_hook is not None:
            self.pipe.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept).images

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
        # print('image min max: ', image)
        latents = self.pipe.vae.encode(((image.unsqueeze(0) * 2) - 1).to(torch.float16)).image_embeds

        latents = self.pipe.vae.config.scaling_factor * latents

        return latents

    def fill_image(self, image, image_mask, inpaintRadius=3):
        image = np.array(image.convert("RGB"))
        image_mask = (np.array(image_mask.convert("L"))).astype(np.uint8)
        filled_image = cv2.inpaint(image, image_mask, inpaintRadius, cv2.INPAINT_TELEA)

        res_img = Image.fromarray(np.clip(filled_image, 0, 255).astype(np.uint8))
        return res_img

    def encode_UV(self, config, image):
        w, h = config.width, config.height

        # Cross-Attention input from torch to PIL
        # image = Image.open(config.image_path)
        # print('img shape: ', image.shape, torch.min(image), torch.max(image))
        # transform = T.ToPILImage()
        image = T.ToPILImage()(image[0].permute(2,0,1))
        # print('img size: ', image.size)
        image = image.resize(size=(w, h), resample=Image.Resampling.BICUBIC)

        # 4. Prepare image, and controlnet_conditioning_image
        image = prepare_image(image)

        latents = self.pipe.vae.encode(image.to(torch.float16).to('cuda')).latent_dist.sample(self.generator)

        latents = self.pipe.vae.config.scaling_factor * latents

        return latents

    def decode_UV(self, config, latents):

        # 8. Post-processing
        image = self.pipe.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.pipe.run_safety_checker(image, latents.device, torch.float32)

        # 10. Convert to PIL
        image = self.pipe.numpy_to_pil(image)

        return image

    def predict_t0(self, config, image, mask, UV_positions, cross_attention_out, denoise_strength, prompt, inference_steps=20):

        w, h = config.width, config.height

        # print('predict_t0 shapes and min max: ', image.shape, mask.shape, UV_positions.shape, cross_attention_out.shape)
        # print('predict_t0 shapes and min max: ', torch.min(image), torch.max(image), torch.min(mask), torch.max(mask),
        #       torch.min(UV_positions), torch.max(UV_positions), torch.min(cross_attention_out),
        #       torch.max(cross_attention_out))

        # input
        # image = Image.open(config.image_path)
        image = T.ToPILImage()(image[0].permute(2, 0, 1))
        image = image.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
        # image.save('input_image_test.png')
        mask = T.ToPILImage()(torch.logical_not(mask).reshape(1, 1024, 1024).float())
        # mask = T.ToPILImage()(mask.reshape(1,256,256).float())
        # mask = mask.filter(ImageFilter.MaxFilter(3))
        mask = mask.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
        # mask.save('mask_test.png')

        image = self.fill_image(image, mask)

        # condition
        control_img = []
        conditioning_scales = []
        for idxs, cnet_unit in enumerate(config.controlnet_units):
            print('idxs: ', idxs, cnet_unit.preprocessor)
            if cnet_unit.preprocessor == 'none':
                print(cnet_unit)
                # condition_image = Image.open(cnet_unit.condition_image_path)
                if idxs==0:
                    condition_image = T.ToPILImage()(UV_positions[0].permute(2, 0, 1))
                    condition_image = condition_image.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
                elif idxs==1:
                    # condition_image = T.ToPILImage()(UV_positions[0].permute(2, 0, 1))
                    # condition_image = condition_image.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
                    condition_image = image
                # condition_image.save('condition_image.png')
            elif cnet_unit.preprocessor == 'inpaint_global_harmonious':
                # condition_image = T.ToPILImage()(cross_attention_out[0].permute(2, 0, 1))
                condition_image = T.ToPILImage()(cross_attention_out[0].permute(2, 0, 1))
                condition_image = self.make_inpaint_condition(condition_image, mask)
                # from torchvision.utils import save_image
                # print(condition_image.shape)
                # save_image(condition_image, 'condition_image_inpaint.png')
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
                              control_image=control_img,
                              ip_adapter_image=ip_adapter_image,
                              height=h,
                              width=w,
                              num_images_per_prompt=config.num_images_per_prompt,
                              guidance_scale=config.guidance_scale,
                              num_inference_steps=inference_steps,
                              strength=denoise_strength,
                              generator=generator,
                              controlnet_conditioning_scale=conditioning_scales).images
        return res_image


def prepare_controlnet_conditioning_image(
    controlnet_conditioning_image,
    width,
    height,
    batch_size,
    num_images_per_prompt,
    device,
    dtype,
    do_classifier_free_guidance,
):
    if not isinstance(controlnet_conditioning_image, torch.Tensor):
        if isinstance(controlnet_conditioning_image, PIL.Image.Image):
            controlnet_conditioning_image = [controlnet_conditioning_image]

        if isinstance(controlnet_conditioning_image[0], PIL.Image.Image):
            controlnet_conditioning_image = [
                np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"]))[None, :]
                for i in controlnet_conditioning_image
            ]
            controlnet_conditioning_image = np.concatenate(controlnet_conditioning_image, axis=0)
            controlnet_conditioning_image = np.array(controlnet_conditioning_image).astype(np.float32) / 255.0
            controlnet_conditioning_image = controlnet_conditioning_image.transpose(0, 3, 1, 2)
            controlnet_conditioning_image = torch.from_numpy(controlnet_conditioning_image)
        elif isinstance(controlnet_conditioning_image[0], torch.Tensor):
            controlnet_conditioning_image = torch.cat(controlnet_conditioning_image, dim=0)

    image_batch_size = controlnet_conditioning_image.shape[0]

    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt

    controlnet_conditioning_image = controlnet_conditioning_image.repeat_interleave(repeat_by, dim=0)

    controlnet_conditioning_image = controlnet_conditioning_image.to(device=device, dtype=dtype)

    if do_classifier_free_guidance:
        controlnet_conditioning_image = torch.cat([controlnet_conditioning_image] * 2)

    return controlnet_conditioning_image

def prepare_image(image):
    if isinstance(image, torch.Tensor):
        # Batch single image
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(dtype=torch.float32)
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image