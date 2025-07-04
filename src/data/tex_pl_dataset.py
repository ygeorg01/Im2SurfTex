import torch
import os
import json
from pytorch3d.renderer import TexturesUV
from src.helper.mesh_helper import init_mesh_training
from torchvision.io import read_image, ImageReadMode
from src.helper.mesh_helper import init_mesh_ours
import lightning as L
from torch.utils.data import DataLoader
from src.data.tex_rgb_dataset import TexRGBDataset

def collate_fn(batch):

    verts = []
    faces_idx = []
    verts_uvs = []
    faces_uvs_idx = []
    folder_path = []
    # normal = torch.zeros(len(batch), batch[0]['normal'].shape[0], batch[0]['normal'].shape[1], batch[0]['normal'].shape[2])
    gen_images = torch.zeros(len(batch), batch[0]['gen_images'].shape[0], batch[0]['gen_images'].shape[1], batch[0]['gen_images'].shape[2], batch[0]['gen_images'].shape[3])
    # loc = torch.zeros(len(batch), batch[0]['loc'].shape[0], batch[0]['loc'].shape[1], batch[0]['loc'].shape[2])
    tex_hd = torch.zeros(len(batch), batch[0]['tex_hd'].shape[0], batch[0]['tex_hd'].shape[1], batch[0]['tex_hd'].shape[2])
    # curr_tex = torch.zeros(len(batch), batch[0]['curr_tex'].shape[0])
    nn_ids = torch.zeros(len(batch), batch[0]['nn_ids'].shape[0], batch[0]['nn_ids'].shape[1])
    nn_dists = torch.zeros(len(batch), batch[0]['nn_ids'].shape[0], batch[0]['nn_ids'].shape[1])
    nn_gamma = torch.zeros(len(batch), batch[0]['nn_gamma'].shape[0], batch[0]['nn_gamma'].shape[1])
    nn_texels_id = torch.zeros(len(batch), batch[0]['nn_texels_id'].shape[0])
    nn_theta = torch.zeros(len(batch), batch[0]['nn_theta'].shape[0], batch[0]['nn_theta'].shape[1])
    tex = torch.zeros(len(batch), batch[0]['tex'].shape[0], batch[0]['tex'].shape[1], batch[0]['tex'].shape[2])
    cameras_dict = {'dist': [], 'elev': [], 'azim': []}
    text = {'text': []}

    verts_max = 0
    faces_idx_max = 0
    verts_uvs_max = 0
    faces_uvs_idx_max = 0

    for idx, items in enumerate(batch):
        for k, v in items.items():
            if k == 'mesh':
                verts.append(v['verts'])
                if verts_max < v['verts'].shape[0]:
                    verts_max = v['verts'].shape[0]
                faces_idx.append(v['faces_idx'])
                if faces_idx_max < v['faces_idx'].shape[0]:
                    faces_idx_max = v['faces_idx'].shape[0]
                verts_uvs.append(v['verts_uvs'])
                if verts_uvs_max < v['verts_uvs'].shape[0]:
                    verts_uvs_max = v['verts_uvs'].shape[0]
                faces_uvs_idx.append(v['face_uvs_idx'])
                if faces_uvs_idx_max < v['face_uvs_idx'].shape[0]:
                    faces_uvs_idx_max = v['face_uvs_idx'].shape[0]

            # elif k == 'normal':
            #     normal[idx] = v
            elif k == 'gen_images':
                gen_images[idx] = v
            elif k == 'tex_hd':
                tex_hd[idx] = v
            elif k == 'folder_path':
                folder_path.append(v)
            # curr_tex
            elif k == 'nn_ids':
                nn_ids[idx] = v
            elif k == 'nn_gamma':
                nn_gamma[idx] = v
            elif k == 'nn_texels_id':
                nn_texels_id[idx] = v
            elif k == 'nn_theta':
                nn_theta[idx] = v
            elif k == 'nn_dists':
                nn_dists[idx] = v
            elif k == 'tex':
                tex[idx] = v
            elif k == 'cameras_dict':
                cameras_dict['dist'].extend(v['dist'])
                cameras_dict['elev'].extend(v['elev'])
                cameras_dict['azim'].extend(v['azim'])
            elif k == 'text':
                # print('text: ', v)
                text['text'] = v
                # cameras_dict.append(v)
            # elif k == 'curr_tex':
            #     curr_tex[idx] = v

    verts_tensor = -torch.ones(len(batch), verts_max, batch[0]['mesh']['verts'].shape[-1])
    faces_idx_tensor = -torch.ones(len(batch), faces_idx_max, batch[0]['mesh']['faces_idx'].shape[-1])
    verts_uvs_tensor = -torch.ones(len(batch), verts_uvs_max, batch[0]['mesh']['verts_uvs'].shape[-1])
    faces_uvs_idx_tensor = -torch.ones(len(batch), faces_uvs_idx_max, batch[0]['mesh']['face_uvs_idx'].shape[-1])

    for idx, (verts_, faces_idx_, verts_uvs_, faces_uvs_idx_) in enumerate(zip(verts, faces_idx, verts_uvs, faces_uvs_idx)):
        verts_tensor[idx, 0:verts_.shape[0]] = verts_
        faces_idx_tensor[idx, 0:faces_idx_.shape[0]] = faces_idx_
        verts_uvs_tensor[idx, 0:verts_uvs_.shape[0]] = verts_uvs_
        faces_uvs_idx_tensor[idx, 0:faces_uvs_idx_.shape[0]] = faces_uvs_idx_

    return {
        'mesh':{
            'verts':verts_tensor,
            'faces_idx':faces_idx_tensor.long(),
            'verts_uvs':verts_uvs_tensor,
            'face_uvs_idx':faces_uvs_idx_tensor.long()
        },
        'folder_path':folder_path,
        # 'normal':normal,
        'gen_images':gen_images,
        'tex_hd':tex_hd,
        'text':text,
        # 'loc':loc,
        # 'loc_tex':loc_tex,
        # 'curr_tex':curr_tex,
        'nn_texels_id':nn_texels_id,
        'nn_ids':nn_ids,
        'nn_gamma':nn_gamma,
        'nn_theta':nn_theta,
        'nn_dists':nn_dists,
        'tex':tex,
        'cameras_dict':cameras_dict
    }

class TexDataModule(L.LightningDataModule):

    def __init__(self, config, batch_size, num_workers, device):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.config = config
        self.device = device

    def prepare_data(self):
        # single gpu
        pass

    def setup(self, stage):
        # multiple gpu
        self.train_dataset = TexRGBDataset(self.config, self.device, type='train')
        self.val_dataset = TexRGBDataset(self.config, self.device, type='val')

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=2,
                          collate_fn=collate_fn)