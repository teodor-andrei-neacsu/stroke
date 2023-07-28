import os
import wandb
import torch
import lightning.pytorch as pl

from key_datamodule import KeyDataModule
from key_module import KeyModule
from lightning.pytorch.loggers import WandbLogger

from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler
# from pytorch_lightning.profiler import AdvancedProfiler
from lightning.pytorch.loggers import TensorBoardLogger

import time

from omegaconf import DictConfig, OmegaConf

def main():

  torch.set_float32_matmul_precision("high")

  config = {
    # data
    "user_cnt": 2000,
    "max_seq_len": 50,
    "test_size": 3,
    "replace_prob": [.1, .8, .1],
    

    "batch_size": 1024,
    "val_batch_size": 16,

    "dataset_multiplier": 10,
    "max_epochs": 300,
    # ConstantLR + CossineAnnealingLR
    # set num_epochs for each in key_module.py 
     
    "num_workers": 8,
    # model
    "feat_cnt": 4,
    "key_cnt": 256,
    "key_emb_size": 128,
    "dim_ff": 512,
    "num_heads": 6,
    "num_layers": 6,
    "trf_dropout": 0.2,
    
    "causal_att": False,

    "use_user_emb": True,
    "user_prob": 0.75,
    # 1 - replace tokens to all users (usr_target==1)
    # 0 - replace tokens to none of the users (usr_target==0)

    # optimizer
    "lr": 5e-4,

    # training
    "check_val_every_n_epoch": 10,
    "num_gpus": 1,

    "pretrain": True,
    
    "wandb_log": True,
    "wandb_project": "fuck_it",

  }

  dm = KeyDataModule(
        root_dir="./Keystrokes/",
        user_cnt=config["user_cnt"],
        max_seq_len=config["max_seq_len"],
        test_size=config["test_size"],
        replace_prob=config["replace_prob"],
        user_prob=config["user_prob"],
        batch_size=config["batch_size"],
        val_batch_size=config["val_batch_size"],
        num_workers=config["num_workers"],
        dataset_multiplier=config["dataset_multiplier"],
    )
  
  model = KeyModule(
      user_cnt=dm.user_cnt,
      feat_cnt=config["feat_cnt"],
      key_cnt=config["key_cnt"],
      key_emb_size=config["key_emb_size"],
      dim_ff=config["dim_ff"],
      num_heads=config["num_heads"],
      num_layers=config["num_layers"],
      dropout=config["trf_dropout"],
      lr=config["lr"],
      causal_att=config["causal_att"],
      use_user_emb=config["use_user_emb"],
  )

  # model = torch.compile(model)


  if  config["pretrain"]:

    if config["wandb_log"]:
      checkpoint_callback = pl.callbacks.ModelCheckpoint(
          monitor="val_eer",
          dirpath=os.getcwd() + "/checkpoints",
          filename='{epoch}_{val_err:.2f}',
          save_top_k=1,
          mode="max",
      )
      wandb.login(key="63faf0d0b57a1855a357085c29f385f911743759")
      trainer = pl.Trainer(
          accelerator="gpu",
          precision="bf16-mixed",
          devices=config["num_gpus"],
          max_epochs=config["max_epochs"],
          callbacks=[LearningRateMonitor(logging_interval="step"),
                     checkpoint_callback],
          check_val_every_n_epoch=config["check_val_every_n_epoch"],
          logger=WandbLogger(
              project=config["wandb_project"],
              config=config,
              name="train_key_emb",
              save_dir=os.getcwd(),
              offline=False,
          ),
          gradient_clip_val=1.0,
          log_every_n_steps=5,
        )
    else:
      trainer = pl.Trainer(
          accelerator="gpu",
          devices=config["num_gpus"],
          max_epochs=config["max_epochs"],
          callbacks=[LearningRateMonitor(logging_interval="step")],
          check_val_every_n_epoch=config["check_val_every_n_epoch"],
          logger=TensorBoardLogger("tb_logs", name="my_model"),
          log_every_n_steps=5,
          profiler="pytorch",
      )
    trainer.fit(model, dm)

  


if __name__ == "__main__":
  main()





