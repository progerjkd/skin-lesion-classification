"""
Model Training Script for SageMaker Training Job

This script handles:
- Model architecture selection
- Data loading and augmentation
- Training loop with validation
- Checkpointing
- Metrics logging
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkinLesionModel:
    """Skin lesion classification model wrapper."""

    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 8,
        pretrained: bool = True,
    ):
        """
        Initialize model.

        Args:
            model_name: Model architecture name
            num_classes: Number of output classes
            pretrained: Use pretrained weights
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = self._build_model(pretrained)

    def _build_model(self, pretrained: bool) -> nn.Module:
        """Build model architecture."""
        logger.info(f"Building {self.model_name} with {self.num_classes} classes")

        if self.model_name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name == "resnet101":
            model = models.resnet101(pretrained=pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=pretrained)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name == "efficientnet_b4":
            model = models.efficientnet_b4(pretrained=pretrained)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name == "densenet121":
            model = models.densenet121(pretrained=pretrained)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=pretrained)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, self.num_classes)

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return model

    def get_model(self) -> nn.Module:
        """Get the model."""
        return self.model


class Trainer:
    """Model trainer."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int,
        device: torch.device,
        learning_rate: float = 0.001,
        epochs: int = 50,
        checkpoint_dir: str = "/opt/ml/checkpoints",
        model_dir: str = "/opt/ml/model",
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            num_classes: Number of classes
            device: Training device
            learning_rate: Learning rate
            epochs: Number of epochs
            checkpoint_dir: Directory for checkpoints
            model_dir: Directory for final model
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.epochs = epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_dir = Path(model_dir)

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=5
        )

        # Tracking
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def train_epoch(self) -> tuple:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 10 == 0:
                logger.info(
                    f"Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate(self) -> tuple:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct / total

        # Calculate additional metrics
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)

        # AUC (one-vs-rest for multiclass)
        try:
            if self.num_classes == 2:
                auc = roc_auc_score(all_labels, all_probabilities[:, 1])
            else:
                auc = roc_auc_score(
                    all_labels, all_probabilities, multi_class="ovr", average="weighted"
                )
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
            auc = 0.0

        return epoch_loss, epoch_acc, auc

    def train(self):
        """Run the training loop."""
        logger.info("Starting training...")

        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")

            # Train
            train_loss, train_acc = self.train_epoch()
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Train Accuracy: {train_acc:.4f}")

            # Validate
            val_loss, val_acc, val_auc = self.validate()
            logger.info(f"Validation Loss: {val_loss:.4f}")
            logger.info(f"Validation Accuracy: {val_acc:.4f}")
            logger.info(f"Validation AUC: {val_auc:.4f}")

            # Update scheduler
            self.scheduler.step(val_loss)

            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Save checkpoint if best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"Saved best model with accuracy: {val_acc:.4f}")

            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)

        # Save final model
        self.save_model()
        logger.info("Training complete!")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save a checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
        }

        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"

        torch.save(checkpoint, path)

    def save_model(self):
        """Save the final model."""
        # Save model weights
        model_path = self.model_dir / "model.pth"
        torch.save(self.model.state_dict(), model_path)

        # Save training history
        history_path = self.model_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Model saved to {self.model_dir}")


def get_data_loaders(
    train_dir: str, val_dir: str, batch_size: int = 32, num_workers: int = 4
) -> tuple:
    """
    Create data loaders.

    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        Training and validation data loaders, class count, and class names
    """
    # Training transforms (with augmentation)
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Number of classes: {len(train_dataset.classes)}")

    return train_loader, val_loader, len(train_dataset.classes), train_dataset.classes


def main():
    """Main training function."""
    parser = argparse.ArgumentParser()

    # Data parameters
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument(
        "--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION")
    )

    # Model parameters
    parser.add_argument("--model-architecture", type=str, default="resnet50")
    parser.add_argument("--pretrained", type=int, default=1)

    # Training parameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=4)

    # SageMaker parameters
    parser.add_argument(
        "--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"),
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Get data loaders
    train_loader, val_loader, num_classes, class_names = get_data_loaders(
        args.train, args.validation, args.batch_size, args.num_workers
    )

    # Build model
    model_builder = SkinLesionModel(
        model_name=args.model_architecture,
        num_classes=num_classes,
        pretrained=bool(args.pretrained),
    )
    model = model_builder.get_model()

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=device,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        model_dir=args.model_dir,
    )

    trainer.train()

    model_config = {
        "model_architecture": args.model_architecture,
        "num_classes": num_classes,
        "class_names": class_names,
        "image_size": 224,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    config_path = Path(args.model_dir) / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    logger.info(f"Saved model configuration to {config_path}")


if __name__ == "__main__":
    main()
