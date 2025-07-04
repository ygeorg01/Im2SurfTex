<div align="center">
<h1 style="font-size: 3em;"> # 📄 Im2SurfTex: Surface Texture Generation via Neural Backprojection of Multi-View Images </h1>
</div>

![Paper Illustration](gallery.png)

> **Authors:** Yiangos Georgiou, Marios Loizou, Melinos Averkiou, Evangelos Kalogerakis
>
> **Affiliations:** University of Cyprus / CYENS CoE / Technical University of Crete
>
> **Published in:** CGF (SGP2025)
>
> **ArXiv:** https://arxiv.org/abs/2502.14006


## 🧠 Abstract
Im2SurfTex is a method that generates textures for input 3D shapes by learning to aggregate multi-view image outputs produced by 2D image diffusion models onto the shapes' texture space. Unlike existing texture generation techniques that use ad hoc backprojection and averaging schemes to blend multiview images into textures, often resulting in texture seams and artifacts, our approach employs a trained neural module to boost texture coherency. The key ingredient of our module is to leverage neural attention and appropriate positional encodings of image pixels based on their corresponding 3D point positions, normals, and surface-aware coordinates as encoded in geodesic distances within surface patches. These encodings capture texture correlations between neighboring surface points, ensuring better texture continuity. Experimental results show that our module improves texture quality, achieving superior performance in high-resolution texture generation.

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/ygeorg01/Im2SurfTex.git
cd Im2SurfTex
```

### 1. Set up the enviroment

```bash
conda env create -f environment.yaml
source activate 3d_aware_texturing
```
## 🎨 Generating Texture

After setting up the environment, you can generate textures using the provided scripts.

### 1. Prepare Input Data

Click this [link](https://ucy-my.sharepoint.com/:u:/g/personal/ygeorg01_ucy_ac_cy/ERYX_oc8AIxJiKbzhEL3HwABfDXwjYzSJQx3kT4G97li3w?e=tgeQle) to download the assets and unzip the file into im2surftex folder

Download the geodesic information by pressing this [here](https://ucy-my.sharepoint.com/:u:/g/personal/ygeorg01_ucy_ac_cy/EUp3pzlWYP1Ds7SRu6Lo9REBWLNB3i2PcO0aW-1CQ1ITeA?e=Dzz9GT) and use this directory as your output directory.

If you want to change output folder move also the geodesic information to the new folder

Also, if there is a need to change camera locations new geodesic information must be computer which takes time

Download pretrained weight [here](https://ucy-my.sharepoint.com/:u:/g/personal/ygeorg01_ucy_ac_cy/Ebpuy4doOzZJpQUhmuAeZJUB6LfZtkcF8Oo2Y29qKnNTJQ?e=jh7IOL)

### 2. Mesh dictionary and config file
1) Define the Meshes that will be used for texturing in src/config/mesh.json. Here you can define the meshes and their corresponding texts prompts.
2) Define config file arguments src/config/template_eval.yaml
   Important arguments: tex_size : [256|1024] , inpaint_strategy : [True|False] , if inpaint_strategy==True change inference_iterations to 3 otherwise the default is 2
### 3. Run the Texture Generation Script

```bash
python src/scripts/texture_mesh.py --out_dir ./textured_shapes --checkpoint_path <checkpoint path>
```

## 🙌 Acknowledgments

- **[Paint3D]**, This project used multiple parts form paint3D repository. Please also consider also this work.


## 📎 Citation
If you found this project or code useful, please cite:

```bibtex
@article{georgiou2025im2surftex,
  title={Im2SurfTex: Surface Texture Generation via Neural Backprojection of Multi-View Images},
  author={Georgiou, Yiangos and Loizou, Marios and Averkiou, Melinos and Kalogerakis, Evangelos},
  journal={arXiv preprint arXiv:2502.14006},
  year={2025}
}
```
