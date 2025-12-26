import torch
from torchvision import datasets, transforms
import lightning as L
from torch.utils.data import DataLoader


class NormalizeToMinusOneOne:
    """Custom transform вместо lambda"""

    def __call__(self, tensor):
        return tensor * 2.0 - 1.0


def init_dataset(path: str):
    """Initialize torch dataset from folder"""
    transformer = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            NormalizeToMinusOneOne(),
        ]
    )
    return datasets.ImageFolder(path, transform=transformer)


def init_dataloader(
    dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0
):  # ← 0 workers!
    """Initialize torch dataloader from dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )


class RPSDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_dir: str,
        test_data_dir: str,
        train_batch_size: int,
        test_batch_size: int,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = init_dataset(self.train_data_dir)
        if stage == "test":
            self.test_dataset = init_dataset(self.test_data_dir)

    def train_dataloader(self):
        return init_dataloader(
            self.train_dataset, self.train_batch_size, shuffle=True, num_workers=0
        )

    def test_dataloader(self):
        return init_dataloader(
            self.test_dataset, self.test_batch_size, shuffle=False, num_workers=0
        )
