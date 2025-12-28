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
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

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

        # Load model
        self.model = self._load_model()

        # Create data loader
        self.test_loader, self.class_names = self._create_data_loader()

    def _load_model(self) -> nn.Module:
        """Load the trained model."""
        logger.info(f"Loading model from {self.model_path}")

        # Find model file
        model_files = list(self.model_path.glob("*.pth"))
        if not model_files:
            raise FileNotFoundError(f"No model file found in {self.model_path}")

        model_file = model_files[0]

        # Load model
        # Note: In production, you'd need to reconstruct the model architecture
        # For now, we'll assume the model is saved with torch.save(model, path)
        try:
            model = torch.load(model_file, map_location=self.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # If the above fails, you need to reconstruct the architecture
            # and load state_dict
            raise

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
        all_probabilities = []

        self.model.eval()

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_labels.extend(labels.numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)

        # Calculate metrics
        metrics = self._calculate_metrics(
            all_labels, all_predictions, all_probabilities
        )

        # Generate visualizations
        self._generate_visualizations(all_labels, all_predictions, all_probabilities)

        # Save evaluation report
        self._save_report(metrics, all_labels, all_predictions)

        logger.info("Evaluation complete!")
        return metrics

    def _calculate_metrics(
        self, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
    ) -> dict:
        """Calculate evaluation metrics."""
        logger.info("Calculating metrics...")

        num_classes = len(self.class_names)

        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average="weighted")
        recall = recall_score(labels, predictions, average="weighted")
        f1 = f1_score(labels, predictions, average="weighted")

        # AUC
        try:
            if num_classes == 2:
                auc = roc_auc_score(labels, probabilities[:, 1])
            else:
                auc = roc_auc_score(
                    labels, probabilities, multi_class="ovr", average="weighted"
                )
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
            auc = 0.0

        # Per-class metrics
        per_class_precision = precision_score(labels, predictions, average=None)
        per_class_recall = recall_score(labels, predictions, average=None)
        per_class_f1 = f1_score(labels, predictions, average=None)

        metrics = {
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "auc": float(auc),
            },
            "per_class_metrics": {
                self.class_names[i]: {
                    "precision": float(per_class_precision[i]),
                    "recall": float(per_class_recall[i]),
                    "f1_score": float(per_class_f1[i]),
                }
                for i in range(num_classes)
            },
        }

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"AUC: {auc:.4f}")

        return metrics

    def _generate_visualizations(
        self, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
    ):
        """Generate evaluation visualizations."""
        logger.info("Generating visualizations...")

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png")
        plt.close()

        logger.info("Visualizations saved")

    def _save_report(self, metrics: dict, labels: np.ndarray, predictions: np.ndarray):
        """Save evaluation report."""
        # Save metrics JSON
        report_path = self.output_dir / "evaluation.json"
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save classification report
        report = classification_report(
            labels, predictions, target_names=self.class_names
        )
        report_text_path = self.output_dir / "classification_report.txt"
        with open(report_text_path, "w") as f:
            f.write(report)

        logger.info(f"Evaluation report saved to {report_path}")
        logger.info(f"Classification report saved to {report_text_path}")


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
