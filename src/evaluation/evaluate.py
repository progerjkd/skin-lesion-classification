"""
Model Evaluation Script for SageMaker Processing Job

This script handles:
- Loading trained model
- Running inference on test set
- Computing comprehensive metrics
- Generating evaluation report
"""

import argparse
import json
import logging
from pathlib import Path
import tarfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Model evaluator for skin lesion classification."""

    def __init__(
        self,
        model_path: str,
        test_dir: str,
        output_dir: str,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model
            test_dir: Test data directory
            output_dir: Output directory for evaluation results
            batch_size: Batch size for inference
            device: Device to use for inference
        """
        self.model_path = Path(model_path)
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_config = self._load_model_config()
        self.model = self._load_model()

        # Create data loader
        self.test_loader, dataset_classes = self._create_data_loader()
        self.class_names = self.model_config.get("class_names") or dataset_classes

    def _load_model_config(self) -> dict:
        """Load model configuration metadata if available."""
        tar_path = self.model_path / "model.tar.gz"
        if tar_path.exists():
            logger.info("Extracting model.tar.gz for evaluation")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(self.model_path)

        config_path = self.model_path / "model_config.json"
        if not config_path.exists():
            logger.warning("Model config not found; using defaults")
            return {}

        with open(config_path, "r") as f:
            return json.load(f)

    def _build_model(self, model_name: str, num_classes: int) -> nn.Module:
        """Build model architecture to match training."""
        if model_name == "resnet50":
            model = models.resnet50(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "resnet101":
            model = models.resnet101(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=False)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        elif model_name == "efficientnet_b4":
            model = models.efficientnet_b4(pretrained=False)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        elif model_name == "densenet121":
            model = models.densenet121(pretrained=False)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
        elif model_name == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=False)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return model

    def _load_model(self) -> nn.Module:
        """Load the trained model state_dict."""
        logger.info(f"Loading model from {self.model_path}")

        model_file = self.model_path / "model.pth"
        if not model_file.exists():
            model_files = list(self.model_path.glob("*.pth"))
            if not model_files:
                raise FileNotFoundError(f"No model file found in {self.model_path}")
            model_file = model_files[0]

        model_name = self.model_config.get("model_architecture", "resnet50")
        num_classes = self.model_config.get("num_classes")
        if not num_classes:
            num_classes = len(self.model_config.get("class_names", [])) or 2

        model = self._build_model(model_name, num_classes)

        state_dict = torch.load(model_file, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)

        logger.info("Model loaded successfully")
        return model

    def _create_data_loader(self) -> tuple:
        """Create test data loader."""
        test_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        test_dataset = datasets.ImageFolder(self.test_dir, transform=test_transforms)

        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

        logger.info(f"Test samples: {len(test_dataset)}")
        logger.info(f"Number of classes: {len(test_dataset.classes)}")

        return test_loader, test_dataset.classes

    def evaluate(self) -> dict:
        """
        Evaluate the model on test set.

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Starting evaluation...")

        all_labels = []
        all_predictions = []
        self.model.eval()

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                _, predicted = outputs.max(1)

                all_labels.extend(labels.numpy())
                all_predictions.extend(predicted.cpu().numpy())

        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        metrics = self._calculate_metrics(
            all_labels, all_predictions
        )
        self._save_report(metrics)

        logger.info("Evaluation complete!")
        return metrics

    def _calculate_metrics(
        self, labels: np.ndarray, predictions: np.ndarray
    ) -> dict:
        """Calculate evaluation metrics without external dependencies."""
        logger.info("Calculating metrics...")

        num_classes = len(self.class_names)
        accuracy = float(np.mean(labels == predictions))

        per_class_metrics = {}
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0
        total_support = 0

        confusion = np.zeros((num_classes, num_classes), dtype=int)
        for true_label, pred_label in zip(labels, predictions):
            confusion[int(true_label), int(pred_label)] += 1

        for idx, class_name in enumerate(self.class_names):
            tp = confusion[idx, idx]
            fp = int(confusion[:, idx].sum() - tp)
            fn = int(confusion[idx, :].sum() - tp)
            support = int(confusion[idx, :].sum())

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )

            per_class_metrics[class_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "support": support,
            }

            weighted_precision += precision * support
            weighted_recall += recall * support
            weighted_f1 += f1 * support
            total_support += support

        if total_support:
            weighted_precision /= total_support
            weighted_recall /= total_support
            weighted_f1 /= total_support

        metrics = {
            "metrics": {
                "accuracy": accuracy,
                "precision": float(weighted_precision),
                "recall": float(weighted_recall),
                "f1_score": float(weighted_f1),
                "auc": 0.0,
            },
            "per_class_metrics": per_class_metrics,
            "class_names": self.class_names,
            "confusion_matrix": confusion.tolist(),
        }

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {weighted_precision:.4f}")
        logger.info(f"Recall: {weighted_recall:.4f}")
        logger.info(f"F1 Score: {weighted_f1:.4f}")

        return metrics

    def _save_report(self, metrics: dict):
        """Save evaluation report."""
        report_path = self.output_dir / "evaluation.json"
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Evaluation report saved to {report_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="/opt/ml/processing/model"
    )
    parser.add_argument("--test-path", type=str, default="/opt/ml/processing/test")
    parser.add_argument(
        "--output-path", type=str, default="/opt/ml/processing/evaluation"
    )
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run evaluation
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        test_dir=args.test_path,
        output_dir=args.output_path,
        batch_size=args.batch_size,
        device=device,
    )

    metrics = evaluator.evaluate()

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(json.dumps(metrics["metrics"], indent=2))


if __name__ == "__main__":
    main()
