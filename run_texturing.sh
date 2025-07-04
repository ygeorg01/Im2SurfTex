#!/bin/env bash
#
#SBATCH -J texture_shapes
#SBATCH --output texture_shapes_%j.txt
#SBATCH -e texture_shapes_%j.err
#SBATCH --ntasks-per-node=1
#
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --qos=normal-a6000
#SBATCH --partition=a6000

export PYTHONPATH="${PYTHONPATH}:./"
export HF_HOME="/lustreFS/data/vcg/yiangos/HF/"

python src/scripts/texture_mesh.py --out_dir ./textured_shapes --checkpoint_path ./checkpoints/weights.pt
