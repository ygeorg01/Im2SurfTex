import os
import argparse

import torch
torch.set_float32_matmul_precision('high')
# import torchvision

# import pytorch_lightning as pl

from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
import lightning as L

import sys
sys.path.append(".")
from src.pipeline.texture_full_model import TextureGeodModel
from torch.utils.data import DataLoader
from src.data.tex_rgb_dataset import TexRGBDataset

# Setup
if torch.cuda.is_available():
    # DEVICE = torch.device("cuda:0")
    DEVICE = "cuda"
else:
    print("no gpu avaiable")
    exit()

from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from src.data.tex_pl_dataset import TexDataModule

# DEVICE = torch.device("cpu")

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/optimize_texture.yaml")
    parser.add_argument("--stamp", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--checkpoint_step", type=int, default=1)
    parser.add_argument("--texture_size", type=int, default=1024)
    parser.add_argument("--ca_checkpoint_path", type=str, default=None)

    # only with template
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--version_name", type=str, default="version")
    parser.add_argument("--geod_neighbors", type=bool, default=False)
    parser.add_argument("--extra_views", type=bool, default=False)

    # Net
    parser.add_argument("--cross_attention_window", type=int, default=3)

    args = parser.parse_args()

    if args.stamp is None:
        setattr(args, "stamp", "{}_{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "debug"))

    return args

def init_config(args):

    config = OmegaConf.load(args.config)
    config.log_dir = args.log_dir
    config.exp_name = args.exp_name
    config.cross_attention_window = args.cross_attention_window
    config.geod_neighbors = args.geod_neighbors
    config.extra_views = args.extra_views
    config.version_name = 'cross_attention_v_ca' + str(config.cross_attention_window) + '_gn_' + str(
        config.geod_neighbors) + '_ev_' + str(config.extra_views) + '_size_' + str(args.texture_size)

    return config

def init_pipeline(
        config,
        stamp,
        device=DEVICE,
        inference_mode=False
    ):
    pipeline = TextureGeodModel(
        config=config,
        stamp=stamp,
    ).to(device)

    pipeline.configure(inference_mode=inference_mode)

    return pipeline


if __name__ == "__main__":

    L.seed_everything(42, workers=True)

    torch.backends.cudnn.benchmark = True

    args = init_args()

    inference_mode = len(args.checkpoint_dir) > 0

    print("=> loading config file...")
    config = init_config(args)
    config.cross_attention_window = args.cross_attention_window
    config.ca_checkpoint_path = args.ca_checkpoint_path

    print("=> initializing pipeline...")
    model = init_pipeline(config=config, stamp=args.stamp, inference_mode=inference_mode)

    print("=> initializing datamodule...")
    data_module = TexDataModule(config, batch_size=1, num_workers=8, device=DEVICE)

    wandb_logger = WandbLogger( project=config.exp_name,
                                name=config.log_stamp,
                                save_dir=config.log_dir,
                                # log_model="all"
                            )

    trainer = L.Trainer(accelerator='gpu',
                        devices=config.n_gpus,
                        strategy='ddp' if config.ddp else 'auto',
                        max_epochs=config.num_epochs,
                        logger=wandb_logger,
                        # accumulate_grad_batches=4*config.n_gpus,
                        accumulate_grad_batches=1,
                        log_every_n_steps=5,
                        val_check_interval=5000,
                        # val_check_interval=5,
                        check_val_every_n_epoch=args.checkpoint_step,
                        )

    # trainer
    if not inference_mode:

        print("=> start training...")
        with torch.autograd.set_detect_anomaly(True):
            trainer.fit(model=model, datamodule=data_module)
    else:
        print("=> loading checkpoint...")
