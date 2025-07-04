import torch

from pytorch3d.structures import (
    Meshes,
    join_meshes_as_scene
)
from pytorch3d.io import (
    load_obj,
    load_objs_as_meshes
)

from pytorch3d.renderer import TexturesUV

def init_mesh_ours(batch, tex):
    # print(batch['mesh']['verts'])
    mesh = Meshes(batch['verts'], batch['faces_idx'])

    if tex.shape[1] == 3:
        tex=tex.permute(0,2,3,1)
    tex = tex[..., :3]

    mesh.textures = TexturesUV(
        maps=torch.clip(tex, min=0, max=1),  # B, H, W, C
        # maps=ones,  # B, H, W, C
        faces_uvs=batch['face_uvs_idx'],
        verts_uvs=batch['verts_uvs'],
        sampling_mode="bilinear",
        # align_corners=False
    )

    return mesh
#
#
def init_mesh_training(scene_config, output_dir, device, join_mesh=True, subdivide_factor=0, is_force=False,
                                  return_mesh=True, return_dict=False):
    """
        Load a list of meshes and reparameterize the UVs
        NOTE: all UVs are packed into one texture space
        Those input meshes will be combined as one mesh for the scene.

        scene_config: {
            "0": {              # instance ID
                "name": ...,    # instance name
                "type": ...,    # instance type ("things" or "stuff")
                "path": ...,    # path to obj file
                "prompt": ...   # description for the instance (left empty for "stuff")
            }
        }

    """

    # load the combined scene
    # NOTE only the combined scene mesh is needed here
    scene_mesh_path = scene_config['path']

    verts, faces, aux = load_obj(scene_mesh_path, device=device)

    return {
        "verts": verts,
        "faces": faces,
        "aux": aux,
        "scene_config": scene_config,
    }
