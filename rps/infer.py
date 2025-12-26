import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

from rps.data import RPSDataModule
from rps.module import RPSModule

# Классы RPS для интерпретации предсказаний
CLASS_NAMES = ["rock", "paper", "scissors"]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def infer(cfg: DictConfig):
    print("Загрузка модели и данных...")

    # === BATCH ТЕСТИРОВАНИЕ ===
    datamodule = RPSDataModule(
        cfg.data.train_data_dir,
        cfg.data.test_data_dir,
        cfg.data.val_data_dir,
        cfg.data.train_batch_size,
        cfg.data.test_batch_size,
    )

    # ✅ ИСПРАВЛЕНО: Загружаем В ПРЯМОЙ МОДУЛЬ, а не module.model
    module = RPSModule(num_classes=cfg.model.num_classes)
    module.load_state_dict(torch.load(cfg.output_file, weights_only=True))  # ← ЗДЕСЬ!
    module.eval()

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=16,
    )

    trainer.test(module, datamodule=datamodule)
    print("✓ Batch тест завершен")

    # === ОДИНОЧНЫЕ ПРЕДСКАЗАНИЯ ===
    single_predict(module, cfg)
    return module


def single_predict(module, cfg: DictConfig):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_images = ["test_rock.jpg", "test_paper.jpg", "test_scissors.jpg"]
    CLASS_NAMES = ["rock", "paper", "scissors"]

    module.eval()
    device = next(module.parameters()).device

    for img_path in test_images:
        if Path(img_path).exists():
            image = Image.open(img_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = module(img_tensor)  # ← ПРЯМО module!
                probs = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()

            print(f"🖼️ {img_path}: {CLASS_NAMES[pred_class]} ({probs.max():.1%})")


if __name__ == "__main__":
    # Запуск с конфигом (как в train)
    model = infer()

    print("\n🎉 Inference завершен!")
    print("Для одиночных предсказаний положите изображения в текущую папку")
    print("Файлы: test_rock.jpg, test_paper.jpg, test_scissors.jpg, my_photo.jpg")
