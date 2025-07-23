import hydra
import os
import sys
import torch
import wandb
import time

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# 현재 파일 기준으로 프로젝트 루트 경로 찾기
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from hydra.utils import instantiate
from src.dataset.datamodule_onehot import DocumentDataModule

@hydra.main(config_path="../../configs", config_name="config")
def train(cfg):
    wandb.login()
    # 모델 저장 패스 생성
    if not os.path.exists(os.path.join(ROOT_DIR, "artifacts")):
        os.makedirs(os.path.join(ROOT_DIR, "artifacts"))

    if not os.path.exists(os.path.join(ROOT_DIR, "artifacts", cfg.model.name)):
        os.makedirs(os.path.join(ROOT_DIR, "artifacts", cfg.model.name))

    start_time = time.time()

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    early_stop_callback = EarlyStopping(
        monitor="val/f1",   # 기준 지표 (v alidation loss 또는 val_acc 등)
        mode="max",          
        patience=4,           # 성능이 개선되지 않는 epoch 수
        verbose=True
    )

    model_path = os.path.join(ROOT_DIR, "artifacts", cfg.model.name)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        dirpath=model_path,
        filename=f"{cfg.model.name}"
    )

    dm = DocumentDataModule(**cfg.data)
    model = instantiate(cfg.model)
    trainer = Trainer(**cfg.train, callbacks=[early_stop_callback, checkpoint_callback], logger=wandb_logger)
    trainer.fit(model, datamodule=dm)
   
    
    # 학습 시간 및 best score 기록
    end_time = time.time()
    total_time_sec = end_time - start_time
    total_time_min = total_time_sec / 60
    best_f1 = checkpoint_callback.best_model_score
    print(f"Training time: {total_time_min:.2f} minutes")
    print(f"Best F1 score: {best_f1:.4f}")

    best_model_path = checkpoint_callback.best_model_path
    best_model = instantiate(cfg.model)
    best_model.load_state_dict(torch.load(best_model_path)['state_dict'])

    wandb_logger.experiment.log({"training_time_sec": total_time_sec, "training_time_min": total_time_min, "best_f1": best_f1})

    # .pt로 저장 (inference용)
    pt_path = os.path.join(model_path, f"{cfg.model.name}.pt")
    torch.save(best_model.state_dict(), pt_path)

    # model_path = os.path.join(ROOT_DIR, "artifacts", f"{cfg.model.name}.pt")
    # torch.save(model.state_dict(), model_path)
    artifact = wandb.Artifact(name = f"{cfg.model.name}", type="model")
    artifact.add_file(pt_path)
    wandb_logger.experiment.log_artifact(artifact)
    
    wandb.finish()

if __name__ == "__main__":
    train()