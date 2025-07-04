import torch
import os
import json
from pytorch3d.renderer import TexturesUV
from src.helper.mesh_helper import init_mesh_training
from torchvision.io import read_image, ImageReadMode
import pandas as pd
import h5py
import numpy as np
import json
import logging
# import objaverse

def normalize_vertices(vertices: torch.Tensor, mesh_scale: float = 1.0, dy: float = 0.0) -> torch.Tensor:
    vertices -= vertices.mean(dim=0)[None, :]
    vertices /= vertices.norm(dim=1).max()
    vertices *= mesh_scale
    vertices[:, 1] += dy
    return vertices

# def set_objaverse_paths(path):
#     # Change objaverse BASE_PATH and _VERSIONED_PATH (where the objects and annotations will be downloaded)
#     base_path = os.path.join(path)
#     logging.info(f'Change objaverse.BASE_PATH: {objaverse.BASE_PATH} -> {base_path}')
#     objaverse.BASE_PATH = base_path
#
#     versioned_path = os.path.join(objaverse.BASE_PATH, 'hf-objaverse-v1')
#     logging.info(f'Change objaverse._VERSIONED_PATH: {objaverse._VERSIONED_PATH} -> {versioned_path}')
#     objaverse._VERSIONED_PATH = versioned_path

class TexRGBDataset(torch.utils.data.Dataset):

    def __init__(self, mesh_list, config, device=None, type='train', size=None):


        # self.render_path = config.renders_dir
        print('mesh list: ', mesh_list)
        print('config: ', config)
        self.log_dir = None

        self.geod_neighbors = config.geod_neighbors

        # self.extra_views = config.extra_views

        # self.obj_id_list = [f.strip() for f in open(config.train_split).readlines()]

        # Change Objaverse base paths
        # set_objaverse_paths(config.dataset_dir)
        # self.data_path = os.path.join(objaverse._VERSIONED_PATH, 'objs')

        # Set Objaverse obj directory
        # self.path_objs = os.path.join(objaverse._VERSIONED_PATH, 'objs')

        # if not os.path.exists(self.path_objs):
        #     print('Objaverse objs directory not found! Please run the Objaverse preprocess task first')
        #     return

        self.device = device
        self.inference = True

        data_paths=[]
        for mesh_info in mesh_list:
            print('mesh info: ', mesh_info)
            if os.path.exists(os.path.join(mesh_info["root_dir"], mesh_info["id"], "metadata_"+str(config.tex_size), "metadata.h5")):
                data_paths.append({
                    'folder_path': os.path.join(mesh_info["root_dir"], mesh_info["id"]),
                    'metadata': os.path.join(mesh_info["root_dir"], mesh_info["id"], "metadata_"+str(config.tex_size)),
                    'text': mesh_info['text'],
                    'id': mesh_info['id']

                })

        print('all data: ', len(data_paths))
        print('data_paths: ', data_paths)
        self.data_paths = pd.DataFrame(data_paths)

    def __len__(self):
        return len(self.data_paths)

    def _init_mesh(self, paths):

        if os.path.exists(os.path.join(paths['folder_path'], 'baked_mesh.obj')):
            mesh_path = os.path.join(paths['folder_path'], 'baked_mesh.obj')
        else:
            mesh_path = os.path.join(paths['folder_path'], 'mesh.obj')

        mesh_path = {
                "path": mesh_path,
        }

        print('mesh path: ', mesh_path)

        mesh_dict = init_mesh_training(
            mesh_path,
            str(self.log_dir),
            'cpu',
            join_mesh=False,
            subdivide_factor=0,
            return_dict=True,
            is_force=True
        )

        tex_img = list(mesh_dict['aux'].texture_images.values())[0]

        return mesh_dict, tex_img

    def apply_texture_to_mesh(self, mesh, faces, aux, texture_tensor, sampling_mode="nearest"):
        # new_mesh = mesh.clone() # in-place operation - DANGER!!!
        mesh.textures = TexturesUV(
            maps=torch.clip(texture_tensor, min=0, max=1), # B, H, W, C
            faces_uvs=faces.textures_idx[None, ...],
            verts_uvs=aux.verts_uvs[None, ...],
            sampling_mode=sampling_mode,
            # align_corners=False
        )

        return mesh

    def __getitem__(self, idx):

        batch = self.data_paths.loc[idx]

        # Load mesh and textures
        mesh_dict, texture_im = self._init_mesh(batch)

        batch['verts'] = mesh_dict["verts"]
        batch['faces_idx'] = mesh_dict["faces"].verts_idx
        batch['verts_uvs'] = mesh_dict["aux"].verts_uvs
        batch['face_uvs_idx'] = mesh_dict["faces"].textures_idx

        batch['tex'] = torch.nn.functional.interpolate((texture_im.permute(2,0,1)[:3].unsqueeze(0)).float(), size=(256, 256)).permute(
            0, 2, 3, 1).squeeze()

        batch['tex_hd'] = texture_im[..., :3].squeeze()

        batch['text']

        # Load Metadata
        metadata = h5py.File(os.path.join(batch['metadata'], 'metadata.h5'))
        # print('metadata keys: ', metadata.keys())
        if self.geod_neighbors:
            batch['nn_ids'] = torch.from_numpy(metadata['geodesic_neighborhoods'][:].astype(np.float32))
            batch['nn_dist'] = torch.from_numpy(metadata['geodesic_distances'][:])

            # Normalize
            batch['nn_dist'][batch['nn_dist']!=-1] = ((batch['nn_dist'][batch['nn_dist']!=-1] - torch.min(batch['nn_dist'][batch['nn_dist']!=-1])) /
                                                      (torch.max(batch['nn_dist'][batch['nn_dist']!=-1])-torch.min(batch['nn_dist'][batch['nn_dist']!=-1])))

        else:
            batch['nn_ids'] = torch.from_numpy(metadata['euclidean_neighborhoods'][:].astype(np.float32))
            batch['nn_dist'] = torch.from_numpy(metadata['euclidean_distances'][:])

        batch['texel_ids'] = torch.from_numpy(metadata['texel_ids'][:].astype(np.float32))
        batch['nn_gamma'] = torch.from_numpy(metadata['polar_coords_gamma'][:]).float()
        batch['nn_gamma'] = torch.arctan2(torch.sin(batch['nn_gamma']),torch.cos(batch['nn_gamma']))


        batch['nn_theta'] = torch.from_numpy(metadata['polar_coords_theta'][:]).float()
        batch['nn_theta'] = torch.arctan2(torch.sin(batch['nn_theta']), torch.cos(batch['nn_theta']))

        idxs = torch.arange(0, batch['nn_ids'].shape[0])
        batch['nn_ids'] = torch.cat((idxs.unsqueeze(1), batch['nn_ids']), axis=1)

        zeros = torch.zeros(batch['texel_ids'].shape)
        batch['nn_gamma'] = torch.cat((zeros.unsqueeze(1), batch['nn_gamma']), axis=1)
        batch['nn_theta'] = torch.cat((zeros.unsqueeze(1), batch['nn_theta']), axis=1)
        batch['nn_dist'] = torch.cat((zeros.unsqueeze(1), batch['nn_dist']), axis=1)

        # Open and read the JSON file
        # with open(batch['text'], 'r') as file:
        #     batch['text'] = json.load(file)['name']
        # batch['gen_images'] = 0

        # batch['cameras_dict'] = 0

        print('batch text : ', batch['text'])

        return {
            'mesh':{'verts':batch['verts'], 'faces_idx':batch['faces_idx'], 'verts_uvs':batch['verts_uvs'], 'face_uvs_idx':batch['face_uvs_idx']},
            # 'normal':batch['normal'],
            # 'gen_images':batch['gen_images'],
            'tex_hd':batch['tex_hd'],
            # 'loc_tex':batch['loc_tex'],
            # 'tex_or':batch['tex_or'],
            'folder_path':batch['folder_path'],
            # 'curr_tex':batch['curr_tex'],
            'nn_ids':batch['nn_ids'],
            'nn_dists':batch['nn_dist'],
            'nn_gamma':batch['nn_gamma'],
            'nn_theta':batch['nn_theta'],
            'nn_texels_id':batch['texel_ids'],
            'tex':batch['tex'],
            # 'cameras_dict':batch['cameras_dict'],
            'id': batch['id'],
            'text': batch['text']
        }
