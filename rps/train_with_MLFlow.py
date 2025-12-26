import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import lightning as L
import hydra
from omegaconf import DictConfig, OmegaConf
import git  # pip install GitPython

from rps.data import RPSDataModule
from rps.module import RPSModule

# ✅ Правильный импорт MLFlowLogger
from lightning.pytorch.loggers import MLFlowLogger


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # ✅ MLflow настройка (опционально)
    try:
        repo = git.Repo(search_parent_directories=True)
        print(f"Git commit: {repo.head.object.hexsha}")
    except:
        print("Git не найден")
    
    datamodule = RPSDataModule(
        cfg.data.train_data_dir,
        cfg.data.test_data_dir,  # ✅ Теперь должно быть в конфиге
        cfg.data.train_batch_size,
        cfg.data.test_batch_size,
    )
    
    model = RPSModule(num_classes=cfg.model.num_classes)
    
    # ✅ ИСПОЛЬЗУЕМ ТОЛЬКО ОДИН LOGGER!
    logger = MLFlowLogger(
        experiment_name="rps-classifier",
        tracking_uri="http://127.0.0.1:8080",  # ✅ Сервер задания
        # tracking_uri="file:./ml-runs"  # Локально
    )
    
    trainer = L.Trainer(
        max_epochs=cfg.num_epochs,
        logger=logger,  # ✅ Только MLFlow!
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=16,  # ✅ Экономия MPS памяти
    )
    
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    
    # ✅ Сохранение модели
    torch.save(model.state_dict(), cfg.output_file)
    
    print(f"✅ Модель сохранена: {cfg.output_file}")
    print(f"✅ MLflow: http://127.0.0.1:8080")


if __name__ == "__main__":
    train()
