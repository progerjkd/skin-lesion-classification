"""
Upload dataset files to S3.
"""

import argparse
import logging
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def iter_files(input_dir: Path):
    for path in input_dir.rglob("*"):
        if path.is_file():
            yield path


def upload_directory(input_dir: Path, bucket: str, prefix: str, region: str, dry_run: bool):
    s3 = boto3.client("s3", region_name=region)
    files = list(iter_files(input_dir))
    if not files:
        logger.warning("No files found to upload.")
        return

    for path in files:
        rel_path = path.relative_to(input_dir).as_posix()
        key = f"{prefix.rstrip('/')}/{rel_path}" if prefix else rel_path

        if dry_run:
            logger.info(f"DRY RUN: {path} -> s3://{bucket}/{key}")
            continue

        try:
            s3.upload_file(str(path), bucket, key)
        except ClientError as exc:
            logger.error(f"Failed to upload {path}: {exc}")
            raise

    logger.info(f"Uploaded {len(files)} files to s3://{bucket}/{prefix}")


def main():
    parser = argparse.ArgumentParser(description="Upload dataset files to S3.")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument(
        "--input-dir",
        default="data/raw",
        help="Local dataset directory to upload",
    )
    parser.add_argument(
        "--prefix",
        default="data/raw",
        help="S3 prefix to upload into",
    )
    parser.add_argument("--region", default=None, help="AWS region override")
    parser.add_argument("--dry-run", action="store_true", help="Print actions only")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    upload_directory(input_dir, args.bucket, args.prefix, args.region, args.dry_run)


if __name__ == "__main__":
    main()
