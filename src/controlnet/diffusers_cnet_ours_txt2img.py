# -*- coding: utf-8 -*-

import time
import torch
from PIL import Image, PngImagePlugin

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers.utils import (
    deprecate,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
import inspect

import PIL
import numpy as np
import torchvision.transforms as T

class txt2imgControlNet():
    def __init__(self, config, torch_dtype=torch.float16):
        controlnet_list = []

        for cnet_unit in config.controlnet_units:
            controlnet = ControlNetModel.from_pretrained(cnet_unit.controlnet_key, torch_dtype=torch_dtype)
            controlnet_list.append(controlnet)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(config.sd_model_key, controlnet=controlnet_list,
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
        res_image = self.pipe(config.prompt,
                              negative_prompt=config.negative_prompt,
                              image=control_img,
                              ip_adapter_image=ip_adapter_image,
                              height=h,
                              width=w,
                              num_images_per_prompt=config.num_images_per_prompt,
                              guidance_scale=config.guidance_scale,
                              num_inference_steps=config.num_inference_steps,
                              generator=generator,
                              controlnet_conditioning_scale=conditioning_scales).images
        return res_image

    def infernece_custom(self, config):
        """
        :param config: task config for txt2img
        :return:
        """

        with torch.no_grad():

            w, h = config.width, config.height

            # condition
            control_img = []
            conditioning_scales = []
            for cnet_unit in config.controlnet_units:
                print('cnet unit: ', cnet_unit)
                if cnet_unit.preprocessor == 'none':
                    condition_image = Image.open(cnet_unit.condition_image_path)
                    condition_image = condition_image.resize(size=(w, h), resample=Image.Resampling.BICUBIC)
                    print('condition_image: ', w,h)
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


            prompt = config.prompt
            image = control_img
            height = h
            width = w
            num_inference_steps = config.num_inference_steps
            timesteps = None
            sigmas = None
            guidance_scale = config.guidance_scale
            negative_prompt = config.negative_prompt
            num_images_per_prompt = config.num_images_per_prompt
            eta: float = 0.0
            generator = generator
            latents = None
            prompt_embeds = None
            negative_prompt_embeds = None
            ip_adapter_image =ip_adapter_image
            ip_adapter_image_embeds = None
            output_type = "pil"
            return_dict = True
            cross_attention_kwargs = None
            controlnet_conditioning_scale = conditioning_scales
            guess_mode = False
            control_guidance_start = 0.0
            control_guidance_end = 1.0
            clip_skip = None
            callback_on_step_end = None
            callback_on_step_end_tensor_inputs = ["latents"]

            # config.prompt,
            # negative_prompt = config.negative_prompt,
            # image = control_img,
            # ip_adapter_image = ip_adapter_image,
            # height = h,
            # width = w,
            # num_images_per_prompt = config.num_images_per_prompt,
            # guidance_scale = config.guidance_scale,
            # num_inference_steps = config.num_inference_steps,
            # generator = generator,
            # controlnet_conditioning_scale = conditioning_scales


            callback = None
            callback_steps = None

            if callback is not None:
                deprecate(
                    "callback",
                    "1.0.0",
                    "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )
            if callback_steps is not None:
                deprecate(
                    "callback_steps",
                    "1.0.0",
                    "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )
            PipelineCallback = int
            MultiPipelineCallbacks = int
            if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
                callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

            controlnet = self.pipe.controlnet._orig_mod if is_compiled_module(self.pipe.controlnet) else self.pipe.controlnet

            # align format for control guidance
            if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
                control_guidance_start = len(control_guidance_end) * [control_guidance_start]
            elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
                control_guidance_end = len(control_guidance_start) * [control_guidance_end]
            elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
                mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
                control_guidance_start, control_guidance_end = (
                    mult * [control_guidance_start],
                    mult * [control_guidance_end],
                )

            # 1. Check inputs. Raise error if not correct
            self.pipe.check_inputs(
                prompt,
                image,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
                ip_adapter_image,
                ip_adapter_image_embeds,
                controlnet_conditioning_scale,
                control_guidance_start,
                control_guidance_end,
                callback_on_step_end_tensor_inputs,
            )

            self.pipe._guidance_scale = guidance_scale
            self.pipe._clip_skip = clip_skip
            self.pipe._cross_attention_kwargs = cross_attention_kwargs
            self.pipe._interrupt = False

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self.pipe._execution_device

            if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
                controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

            global_pool_conditions = (
                controlnet.config.global_pool_conditions
                if isinstance(controlnet, ControlNetModel)
                else controlnet.nets[0].config.global_pool_conditions
            )
            guess_mode = guess_mode or global_pool_conditions
            # self.pipe.cross_attention_kwargs=None
            # 3. Encode input prompt
            text_encoder_lora_scale = (
                self.pipe.cross_attention_kwargs.get("scale", None) if self.pipe.cross_attention_kwargs is not None else None
            )
            # text_encoder_lora_scale = None
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.pipe.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
                clip_skip=self.pipe.clip_skip,
            )
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if self.pipe.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                    self.pipe.do_classifier_free_guidance,
                )

            # 4. Prepare image
            if isinstance(controlnet, ControlNetModel):
                image = self.pipe.prepare_image(
                    image=image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.pipe.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                height, width = image.shape[-2:]

            elif isinstance(controlnet, MultiControlNetModel):
                images = []

                # Nested lists as ControlNet condition
                if isinstance(image[0], list):
                    # Transpose the nested image list
                    image = [list(t) for t in zip(*image)]

                for image_ in image:
                    image_ = self.pipe.prepare_image(
                        image=image_,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=controlnet.dtype,
                        do_classifier_free_guidance=self.pipe.do_classifier_free_guidance,
                        guess_mode=guess_mode,
                    )

                    images.append(image_)

                image = images
                height, width = image[0].shape[-2:]
            else:
                assert False

            # 5. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(
                self.pipe.scheduler, num_inference_steps, device, timesteps, sigmas
            )

            self.pipe._num_timesteps = len(timesteps)

            # 6. Prepare latent variables
            num_channels_latents = self.pipe.unet.config.in_channels
            latents = self.pipe.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            # 6.5 Optionally get Guidance Scale Embedding
            timestep_cond = None
            if self.pipe.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(self.pipe.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
                timestep_cond = self.pipe.get_guidance_scale_embedding(
                    guidance_scale_tensor, embedding_dim=self.pipe.unet.config.time_cond_proj_dim
                ).to(device=device, dtype=latents.dtype)

            # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)

            # 7.1 Add image embeds for IP-Adapter
            added_cond_kwargs = (
                {"image_embeds": image_embeds}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None
                else None
            )

            # 7.2 Create tensor stating which controlnets to keep
            controlnet_keep = []
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

            # 8. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.pipe.scheduler.order
            is_unet_compiled = is_compiled_module(self.pipe.unet)
            is_controlnet_compiled = is_compiled_module(self.pipe.controlnet)
            is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
            latents_or_sample_list = []
            print('timesteps: ', timesteps)
            with self.pipe.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    print('i, t, ', i , t)
                    # if self.pipe.interrupt:
                    #     continue

                    # Relevant thread:
                    # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                    if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                        torch._inductor.cudagraph_mark_step_begin()

                    # expand the latents if we are doing classifier free guidance
                    print('latent shape: ', latents.shape)

                    latent_model_input = torch.cat([latents] * 2) if self.pipe.do_classifier_free_guidance else latents
                    latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)


                    # controlnet(s) inference
                    if guess_mode and self.pipe.do_classifier_free_guidance:
                        # Infer ControlNet only for the conditional batch.
                        control_model_input = latents
                        control_model_input = self.pipe.scheduler.scale_model_input(control_model_input, t)
                        controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                    else:
                        control_model_input = latent_model_input
                        controlnet_prompt_embeds = prompt_embeds

                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[i]

                    down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=image,
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )

                    if guess_mode and self.pipe.do_classifier_free_guidance:
                        # Inferred ControlNet only for the conditional batch.
                        # To apply the output of ControlNet to both the unconditional and conditional batches,
                        # add 0 to the unconditional batch to keep it unchanged.
                        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                    # print('latent model input: ', latent_model_input.shape)

                    # predict the noise residual
                    noise_pred = self.pipe.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.pipe.cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    print('noise_pred shape: ', noise_pred.shape)

                    # perform guidance
                    if self.pipe.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    # latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    latents_dict = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
                    latents_or = torch.nn.functional.interpolate(latents_dict.pred_original_sample.to(torch.float16), (256,256))
                    print('latents 1 or shape: ', latents_or.shape)

                    latents_or = torch.nn.functional.interpolate(latents_or, (64,64))

                    # Load geodesic nn
                    print('latents 2 or shape: ', latents_or.shape)

                    # Add Noise
                    sigma = self.pipe.scheduler.sigmas[self.pipe.scheduler.step_index-1]
                    sigma_from = self.pipe.scheduler.sigmas[self.pipe.scheduler.step_index-1]
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



                    # latents = latents_dict.prev_sample
                    latents_or_sample_list.append(latents_dict.pred_original_sample)
                    # print('latents shape: ', latents_dict)
                    # # break

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self.pipe, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

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

            if not output_type == "latent":
                image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False, generator=generator)[
                    0
                ]
                image, has_nsfw_concept = self.pipe.run_safety_checker(image, device, prompt_embeds.dtype)
            else:
                image = latents
                has_nsfw_concept = None

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = self.pipe.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

            # Offload all models
            self.pipe.maybe_free_model_hooks()

            if not return_dict:
                return (image, has_nsfw_concept)

            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept).images, torch.cat(latents_or_sample_list, dim=0)

    def encode_UV(self, config, image):
        w, h = config.width, config.height

        # Cross-Attention input from torch to PIL
        # image = Image.open(config.image_path)
        print('img shape: ', image.shape, torch.min(image), torch.max(image))
        # transform = T.ToPILImage()
        image = T.ToPILImage()(image[0].permute(2,0,1))
        print('img size: ', image.size)
        image = image.resize(size=(w, h), resample=Image.Resampling.BICUBIC)

        # 4. Prepare image, and controlnet_conditioning_image
        image = prepare_image(image)

        latents = self.pipe.vae.encode(image.to(torch.float16).to('cuda')).latent_dist.sample(self.generator)

        latents = self.pipe.vae.config.scaling_factor * latents

        return latents

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps

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