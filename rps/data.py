import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NormalizeToMinusOneOne:
   def __call__(self, tensor):
        return tensor * 2.0 - 1.0


def init_dataset(path: str):
    transformer = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            NormalizeToMinusOneOne(),
        ]
    )
    return datasets.ImageFolder(path, transform=transformer)


def init_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )


class RPSDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_dir: str,
        test_data_dir: str,
        val_data_dir: str,
        train_batch_size: int,
        test_batch_size: int,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.val_data_dir = val_data_dir
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = init_dataset(self.train_data_dir)
            self.val_dataset = init_dataset(self.val_data_dir)
        if stage == "test" or stage is None:
            self.test_dataset = init_dataset(self.test_data_dir)

    def train_dataloader(self):
        return init_dataloader(
            self.train_dataset,
            self.train_batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return init_dataloader(
            self.val_dataset,
            self.test_batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return init_dataloader(
            self.test_dataset,
            self.test_batch_size,
            shuffle=False,
            num_workers=4,
        )

    def predict_dataloader(self):
        return init_dataloader(
            self.test_dataset, self.test_batch_size, shuffle=False, num_workers=4
        )
