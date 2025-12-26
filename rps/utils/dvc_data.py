"""Module for downloading RPS dataset from Google Drive archive."""

import shutil
import zipfile
from pathlib import Path

import gdown


def download_data(data_dir: str = "data", force: bool = False) -> None:
    """Download and extract RPS dataset from Google Drive archive.

    Args:
        data_dir: Directory to save the data
        force: Force redownload even if data exists
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    # Debug: show what's already there
    existing_files = list(data_path.glob("*"))
    print(f"Found {len(existing_files)} items in {data_path}:")
    for f in existing_files[:5]:
        print(f"  - {f.name}")

    if not force and len(existing_files) > 0:
        print(f"⚠️  Data already exists in {data_path}")
        print("Use force=True to redownload")
        return

    if force or len(existing_files) > 0:
        shutil.rmtree(data_path)
        data_path.mkdir(exist_ok=True)
        print("🧹 Cleared data directory")

    # Google Drive file ID
    file_id = "1L-dwnigLiWw5_4nNn4yfzHKbZoAX-OdG"
    file_url = f"https://drive.google.com/uc?id={file_id}"

    print("📥 Downloading RPS dataset archive...")
    archive_path = data_path / "rps_dataset.zip"

    try:
        gdown.download(file_url, str(archive_path), quiet=False)
        print("✅ Archive downloaded!")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return

    # Extract archive
    print("📦 Extracting...")
    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(data_path)
        archive_path.unlink()
        print("✅ Extraction completed!")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return

    # 🔥 АВТО-ПЕРЕМЕЩЕНИЕ: поднимаем train/test/val на уровень data/
    print("🗂️  Organizing structure...")

    # Возможные вложенные папки
    nested_folders = ["Rock-Paper-Scissors", "RPS", "rps", "dataset"]

    for nested in nested_folders:
        nested_path = data_path / nested
        if nested_path.exists():
            print(f"Found nested folder: {nested}")

            # Перемещаем train, test, validation наверх
            for split in ["train", "test", "validation", "val"]:
                src = nested_path / split
                dst = data_path / split
                if src.exists():
                    shutil.rmtree(dst, ignore_errors=True)
                    shutil.move(str(src), str(dst))
                    print(f"  ✓ Moved {split}/ to data/{split}/")

            # Удаляем пустую вложенную папку
            shutil.rmtree(nested_path, ignore_errors=True)
            break

    # 🔥 АВТО-ОБРАБОТКА validation/ — собираем файлы по именам
    validation_path = data_path / "validation"
    if validation_path.exists() and len(list(validation_path.glob("*"))) == 0:
        print("🔍 Auto-filling validation/ from files with class prefixes...")

        # Создаём папки классов
        for cls in ["rock", "paper", "scissors"]:
            (validation_path / cls).mkdir(exist_ok=True)

        # Ищем файлы по паттернам везде в data/
        patterns = [
            "*rock*",
            "*Rock*",
            "rockval*",
            "rock_val*",
            "*paper*",
            "*Paper*",
            "paperval*",
            "paper_val*",
            "*scissor*",
            "*Scissor*",
            "scissorval*",
            "scissor_val*",
        ]

        moved_count = 0
        for pattern in patterns:
            for file_path in data_path.rglob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                }:
                    # Определяем класс по паттерну
                    if any(p in file_path.name.lower() for p in ["rock", "rockval"]):
                        cls = "rock"
                    elif any(
                        p in file_path.name.lower() for p in ["paper", "paperval"]
                    ):
                        cls = "paper"
                    elif any(
                        p in file_path.name.lower() for p in ["scissor", "scissorval"]
                    ):
                        cls = "scissors"
                    else:
                        continue

                    # Перемещаем в validation/cls/
                    dest = validation_path / cls / file_path.name
                    if not dest.exists():
                        shutil.move(str(file_path), str(dest))
                        moved_count += 1
                        print(f"    ✓ Moved {file_path.name} → validation/{cls}/")

        print(f"    📊 Moved {moved_count} files to validation/")

    # Финальная проверка структуры
    print("\n📁 Final structure:")
    for split in ["train", "test", "validation", "val"]:
        split_path = data_path / split
        if split_path.exists():
            classes = [d.name for d in split_path.iterdir() if d.is_dir()]
            class_counts = {c: len(list((split_path / c).glob("*.*"))) for c in classes}
            print(f"  {split}/: {len(classes)} classes - {class_counts}")
        else:
            print(f"  ❌ Missing {split}/")

    print("\n🎉 Dataset ready in correct structure!")


if __name__ == "__main__":
    download_data(force=True)
