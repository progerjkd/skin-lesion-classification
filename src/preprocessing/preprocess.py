"""
Data Preprocessing Script for SageMaker Processing Job

This script handles:
- Data loading from S3
- Data validation and cleaning
- Train/val/test splitting
- Image preprocessing and augmentation
- Saving processed data
"""

import argparse
import os
import json
import logging
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkinLesionPreprocessor:
    """Preprocessor for skin lesion images."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        train_split: float = 0.8,
        val_split: float = 0.1,
        image_size: Tuple[int, int] = (224, 224),
        random_state: int = 42,
    ):
        """
        Initialize preprocessor.

        Args:
            input_dir: Input data directory
            output_dir: Output data directory
            train_split: Train split ratio
            val_split: Validation split ratio
            image_size: Target image size
            random_state: Random seed
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = 1 - train_split - val_split
        self.image_size = image_size
        self.random_state = random_state

        # Create output directories
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "validation"
        self.test_dir = self.output_dir / "test"

        for directory in [self.train_dir, self.val_dir, self.test_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def load_metadata(self) -> pd.DataFrame:
        """
        Load metadata file.

        Returns:
            DataFrame with image metadata
        """
        metadata_files = list(self.input_dir.glob("*.csv"))

        if not metadata_files:
            logger.warning("No metadata file found. Creating from directory structure.")
            return self._create_metadata_from_directory()

        metadata_path = metadata_files[0]
        logger.info(f"Loading metadata from {metadata_path}")

        df = pd.read_csv(metadata_path)
        logger.info(f"Loaded {len(df)} records")

        return df

    def _create_metadata_from_directory(self) -> pd.DataFrame:
        """
        Create metadata from directory structure.

        Expected structure:
        input_dir/
            class1/
                image1.jpg
                image2.jpg
            class2/
                image3.jpg
        """
        data = []

        for class_dir in self.input_dir.iterdir():
            if class_dir.is_dir():
                for image_path in class_dir.glob("*.jpg"):
                    data.append(
                        {
                            "image": image_path.name,
                            "label": class_dir.name,
                            "path": str(image_path),
                        }
                    )

        return pd.DataFrame(data)

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean data.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info("Validating data...")

        # Check for required columns
        if "image" not in df.columns:
            raise ValueError("Metadata must contain 'image' column")

        # Remove duplicates
        original_len = len(df)
        df = df.drop_duplicates(subset=["image"])
        logger.info(f"Removed {original_len - len(df)} duplicate records")

        # Remove records with missing images
        valid_records = []
        for idx, row in df.iterrows():
            image_path = self.input_dir / row["image"]
            if image_path.exists():
                valid_records.append(row)
            else:
                logger.warning(f"Image not found: {image_path}")

        df = pd.DataFrame(valid_records)
        logger.info(f"Valid records: {len(df)}")

        return df

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Input DataFrame

        Returns:
            Train, validation, and test DataFrames
        """
        logger.info("Splitting data...")

        # First split: train + val vs test
        train_val, test = train_test_split(
            df,
            test_size=self.test_split,
            random_state=self.random_state,
            stratify=df["label"] if "label" in df.columns else None,
        )

        # Second split: train vs val
        val_ratio = self.val_split / (self.train_split + self.val_split)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=train_val["label"] if "label" in train_val.columns else None,
        )

        logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

        return train, val, test

    def preprocess_image(self, image_path: Path) -> np.ndarray:
        """
        Preprocess a single image.

        Args:
            image_path: Path to image

        Returns:
            Preprocessed image array
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Resize
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)

        return image

    def process_split(
        self, df: pd.DataFrame, output_dir: Path, split_name: str
    ) -> None:
        """
        Process and save a data split.

        Args:
            df: DataFrame for this split
            output_dir: Output directory
            split_name: Name of split (train/val/test)
        """
        logger.info(f"Processing {split_name} split...")

        # Create class directories
        if "label" in df.columns:
            for label in df["label"].unique():
                (output_dir / label).mkdir(parents=True, exist_ok=True)

        # Process each image
        processed_count = 0
        for idx, row in df.iterrows():
            try:
                # Find image path
                image_path = self.input_dir / row["image"]
                if not image_path.exists():
                    # Try alternative paths
                    for ext in [".jpg", ".jpeg", ".png"]:
                        alt_path = self.input_dir / f"{row['image']}{ext}"
                        if alt_path.exists():
                            image_path = alt_path
                            break

                # Preprocess image
                image = self.preprocess_image(image_path)

                # Save to appropriate directory
                if "label" in row:
                    save_path = output_dir / row["label"] / image_path.name
                else:
                    save_path = output_dir / image_path.name

                image.save(save_path)
                processed_count += 1

                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count}/{len(df)} images")

            except Exception as e:
                logger.error(f"Error processing {row['image']}: {str(e)}")

        logger.info(f"Processed {processed_count} images for {split_name}")

        # Save metadata
        metadata_path = output_dir / "metadata.csv"
        df.to_csv(metadata_path, index=False)
        logger.info(f"Saved metadata to {metadata_path}")

    def run(self) -> None:
        """Run the preprocessing pipeline."""
        logger.info("Starting preprocessing pipeline...")

        # Load metadata
        df = self.load_metadata()

        # Validate data
        df = self.validate_data(df)

        # Split data
        train_df, val_df, test_df = self.split_data(df)

        # Process each split
        self.process_split(train_df, self.train_dir, "train")
        self.process_split(val_df, self.val_dir, "validation")
        self.process_split(test_df, self.test_dir, "test")

        # Save split statistics
        stats = {
            "train_count": len(train_df),
            "val_count": len(val_df),
            "test_count": len(test_df),
            "total_count": len(df),
            "train_split": self.train_split,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "image_size": self.image_size,
        }

        if "label" in df.columns:
            stats["class_distribution"] = train_df["label"].value_counts().to_dict()

        stats_path = self.output_dir / "preprocessing_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved preprocessing statistics to {stats_path}")
        logger.info("Preprocessing complete!")


def main():
    """Main function for SageMaker processing job."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-split", type=float, default=0.8, help="Train split ratio"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--image-size", type=int, default=224, help="Target image size"
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # SageMaker paths
    input_dir = "/opt/ml/processing/input"
    train_dir = "/opt/ml/processing/train"
    val_dir = "/opt/ml/processing/validation"
    test_dir = "/opt/ml/processing/test"

    # Create output directory structure
    output_base = Path("/opt/ml/processing")

    # Run preprocessing
    preprocessor = SkinLesionPreprocessor(
        input_dir=input_dir,
        output_dir=output_base,
        train_split=args.train_split,
        val_split=args.val_split,
        image_size=(args.image_size, args.image_size),
        random_state=args.random_state,
    )

    preprocessor.run()


if __name__ == "__main__":
    main()
