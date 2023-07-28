import hydra
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
from lightning.pytorch.loggers import TensorBoardLogger

from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def stroke(cfg: DictConfig) -> None:

  torch.set_float32_matmul_precision("medium")
  
  dm = KeyDataModule(
    root_dir=cfg.ds_params.data_path,
    user_cnt=cfg.ds_params.user_cnt,
    max_seq_len=cfg.ds_params.max_seq_len,
    test_size=cfg.ds_params.test_size,
    replace_prob=cfg.ds_params.replace_prob,
    user_prob=cfg.ds_params.user_prob,
    batch_size=cfg.training_params.batch_size,
    val_batch_size=cfg.training_params.val_batch_size,
    num_workers=cfg.ds_params.num_workers,
    dataset_multiplier=cfg.ds_params.dataset_multiplier
  )

  model = KeyModule(
      user_cnt=dm.user_cnt,
      feat_cnt=cfg.ds_params.feat_cnt,
      key_cnt=cfg.ds_params.key_cnt,
      key_emb_size=cfg.model_params.key_emb_size,
      dim_ff=cfg.model_params.dim_ff,
      num_heads=cfg.model_params.num_heads,
      num_layers=cfg.model_params.num_layers,
      dropout=cfg.model_params.trf_dropout,
      lr=cfg.training_params.lr,
      causal_att=cfg.model_params.causal_att,
      use_user_emb=cfg.model_params.use_user_emb,
  )

  if cfg.training_params.train:
    if cfg.training_params.wandb_log:
      checkpoint_callback = pl.callbacks.ModelCheckpoint(
          monitor="val_token_acc",
          dirpath=os.getcwd(),
          filename="best_model",
          save_top_k=1,
          mode="max",
      )
      wandb.login(key="63faf0d0b57a1855a357085c29f385f911743759")
      trainer = pl.Trainer(
          accelerator="gpu",
          precision="bf16-mixed",
          devices=cfg.training_params.num_gpus,
          max_epochs=cfg.training_params.max_epochs,
          callbacks=[LearningRateMonitor(logging_interval="step")
                    #  checkpoint_callback
                     ],
          check_val_every_n_epoch=cfg.training_params.check_val_every_n_epoch,
          logger=WandbLogger(
              project=cfg.training_params.wandb_project,
              config=cfg,
              name="train_key_emb",
              save_dir=os.getcwd(),
              offline=False,
          ),
          log_every_n_steps=2,
          strategy="ddp",
        )
      
      trainer.fit(model, dm)
      wandb.finish()


if __name__ == "__main__":
  stroke()