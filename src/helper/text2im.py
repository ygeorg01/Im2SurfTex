import os
from tqdm import tqdm
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

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


def gen_init_view(sd_cfg, cnet, depth_im, latent_im, outdir, idx):

    save_path = os.path.join(outdir, f"depth_render_" + str(idx) + ".png")
    save_image(depth_im / 255., save_path)

    # post-processing depth，dilate
    depth_dilated = dilate_depth_outline(save_path, iters=1, dilate_kernel=3)
    save_path = os.path.join(outdir, f"depth_dilated_" + str(idx) + ".png")
    cv2.imwrite(save_path, depth_dilated)

    p_cfg = sd_cfg.txt2img
    p_cfg.controlnet_units[0].condition_image_path = save_path

    save_path = os.path.join(outdir, f"latent_render_" + str(idx) + ".png")
    save_image(latent_im, save_path)
    p_cfg.latent_image_path = save_path
    p_cfg.denoising_strength = sd_cfg.denoising_strength

    images = cnet.infernece(config=p_cfg)
    for i, img in enumerate(images):
        save_path = os.path.join(outdir, f'init-img-{i}.png')
        img.save(save_path)
    return images

def inpaint_viewpoint(sd_cfg, cnet, save_result_dir, depth_im, latent_im, masked_img_mesh, idx, inpaint_views=[2, 3]):
    # projection
    print(f"Project inpaint view ..")
    # view_angle_info = {i:data for i, data in enumerate(dataloaders['train'])}
    print('latent im shape: ', latent_im.shape, depth_im.shape)
    inpaint_used_key = ["image", "depth", "uncolored_mask"]
    one_batch_img = []
    for i, one_batch_id in tqdm(enumerate(inpaint_views)):

        # for view_id in one_batch_id:
        #     data = view_angle_info[view_id]
        #     theta, phi, radius = data['theta'], data['phi'], data['radius']
        #     outputs = mesh_model.render(theta=theta, phi=phi, radius=radius)
        #     view_img_info = [outputs[k] for k in inpaint_used_key]
        # mask = (latent_im-torch.tensor([1., 51/255, 1.]).to(latent_im.device).view(1,3,1,1)).abs().sum(axis=1)
        # mask = (mask < 0.1).float().unsqueeze(0)

        one_batch_img.append(latent_im)
        one_batch_img.append(depth_im)
        one_batch_img.append((masked_img_mesh < 0.1).float())

        # for i, img in enumerate(zip(*one_batch_img)):
        for i, img in enumerate(one_batch_img):

            if img.size(1) == 1:
                img = img.repeat(1, 3, 1, 1)
            t = '_'.join(map(str, one_batch_id))
            name = inpaint_used_key[i]
            if name == "uncolored_mask":
                img[img > 0] = 1
            save_path = os.path.join(save_result_dir, f"view_{t}_{name}.png")
            save_tensor_image(img, save_path=save_path)

    # inpaint view point
    txt_cfg = sd_cfg.txt2img
    img_cfg = sd_cfg.inpaint
    copy_list = ["prompt", "negative_prompt", "seed", ]
    for k in copy_list:
        img_cfg[k] = txt_cfg[k]

    for i, one_batch_id in tqdm(enumerate(inpaint_views)):
        t = '_'.join(map(str, one_batch_id))
        rgb_path = os.path.join(save_result_dir, f"view_{t}_{inpaint_used_key[0]}.png")
        depth_path = os.path.join(save_result_dir, f"view_{t}_{inpaint_used_key[1]}.png")
        mask_path = os.path.join(save_result_dir, f"view_{t}_{inpaint_used_key[2]}.png")

        # pre-processing inpaint mask: dilate
        mask = cv2.imread(mask_path)
        dilate_kernel = 10
        mask = cv2.dilate(mask, np.ones((dilate_kernel, dilate_kernel), np.uint8))
        mask_path = os.path.join(save_result_dir, f"view_{t}_{inpaint_used_key[2]}_d{dilate_kernel}.png")
        cv2.imwrite(mask_path, mask)

        img_cfg.image_path = rgb_path
        img_cfg.mask_path = mask_path
        img_cfg.controlnet_units[0].condition_image_path = depth_path
        images = cnet.infernece(config=img_cfg)
        for i, img in enumerate(images):
            save_path = os.path.join(save_result_dir, f"view_{t}_rgb_inpaint_{i}.png")
            img.save(save_path)
    return images