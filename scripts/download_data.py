"""
Script to download ISIC skin lesion datasets

Supports:
- ISIC 2019
- HAM10000
- BCN20000
"""

import argparse
import logging
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import kaggle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Download skin lesion datasets."""

    def __init__(self, output_dir: str):
        """
        Initialize downloader.

        Args:
            output_dir: Directory to save downloaded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_isic2019(self):
        """
        Download ISIC 2019 dataset from Kaggle.

        Requires: kaggle API credentials configured
        """
        logger.info("Downloading ISIC 2019 dataset...")

        try:
            # Download using Kaggle API
            kaggle.api.competition_download_files(
                "isic-2019",
                path=str(self.output_dir),
                quiet=False
            )

            # Extract
            zip_file = self.output_dir / "isic-2019.zip"
            if zip_file.exists():
                logger.info("Extracting dataset...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(self.output_dir)
                zip_file.unlink()  # Remove zip file

            logger.info(f"ISIC 2019 dataset downloaded to {self.output_dir}")

        except Exception as e:
            logger.error(f"Error downloading ISIC 2019: {e}")
            logger.info("Please ensure Kaggle API is configured:")
            logger.info("1. Create account at kaggle.com")
            logger.info("2. Go to Account -> Create New API Token")
            logger.info("3. Place kaggle.json in ~/.kaggle/")
            raise

    def download_ham10000(self):
        """
        Download HAM10000 dataset from Kaggle.
        """
        logger.info("Downloading HAM10000 dataset...")

        try:
            kaggle.api.dataset_download_files(
                "kmader/skin-cancer-mnist-ham10000",
                path=str(self.output_dir),
                unzip=True
            )

            logger.info(f"HAM10000 dataset downloaded to {self.output_dir}")

        except Exception as e:
            logger.error(f"Error downloading HAM10000: {e}")
            raise

    def download_bcn20000(self):
        """
        Download BCN20000 dataset.

        Note: This dataset may require manual download and agreement to terms.
        """
        logger.info("BCN20000 dataset requires manual download")
        logger.info("Please visit: https://challenge2020.isic-archive.com/")
        logger.info("Download and extract to the output directory")

    def verify_dataset(self, dataset_name: str) -> bool:
        """
        Verify that dataset was downloaded correctly.

        Args:
            dataset_name: Name of the dataset

        Returns:
            True if dataset appears valid
        """
        logger.info(f"Verifying {dataset_name} dataset...")

        # Check for image files
        image_files = list(self.output_dir.rglob("*.jpg")) + \
                     list(self.output_dir.rglob("*.jpeg")) + \
                     list(self.output_dir.rglob("*.png"))

        if not image_files:
            logger.error("No image files found!")
            return False

        logger.info(f"Found {len(image_files)} image files")

        # Check for metadata
        csv_files = list(self.output_dir.rglob("*.csv"))
        if csv_files:
            logger.info(f"Found {len(csv_files)} metadata files")
        else:
            logger.warning("No metadata CSV files found")

        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download skin lesion datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["isic2019", "ham10000", "bcn20000"],
        help="Dataset to download"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify dataset after download"
    )

    args = parser.parse_args()

    # Create downloader
    downloader = DatasetDownloader(output_dir=args.output_dir)

    # Download dataset
    if args.dataset == "isic2019":
        downloader.download_isic2019()
    elif args.dataset == "ham10000":
        downloader.download_ham10000()
    elif args.dataset == "bcn20000":
        downloader.download_bcn20000()

    # Verify if requested
    if args.verify:
        if downloader.verify_dataset(args.dataset):
            logger.info("Dataset verification successful!")
        else:
            logger.error("Dataset verification failed!")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
