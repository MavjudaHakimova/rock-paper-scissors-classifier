import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from model import FeatureExtractor


class RPSModule(L.LightningModule):
    def __init__(self, num_classes: int = 3, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.feature_extractor = FeatureExtractor()
        self.classifier_head = torch.nn.Linear(1280, num_classes)
        
        # Метрики для всех стадий
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier_head(features)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        
        # Логируем и train accuracy
        self.train_accuracy(preds, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Добавлен validation_step"""
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        
        self.val_accuracy(preds, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        
        self.test_accuracy(preds, labels)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_accuracy, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
