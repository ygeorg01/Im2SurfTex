from typing import NamedTuple, Sequence
#
# from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.renderer import (
    AmbientLights,
    SoftPhongShader
)

class BlendParams(NamedTuple):
    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Sequence = (1, 1, 1)

def init_soft_phong_shader(camera, device, blend_params=BlendParams()):

    lights = AmbientLights()
    shader = SoftPhongShader(
        cameras=camera,
        lights=lights,
        blend_params=blend_params
    )

    return shader
