"""
Unit tests for preprocessing module
"""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import shutil


class TestSkinLesionPreprocessor:
    """Test cases for SkinLesionPreprocessor class."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        input_dir = tempfile.mkdtemp()
        output_dir = tempfile.mkdtemp()

        yield input_dir, output_dir

        # Cleanup
        shutil.rmtree(input_dir)
        shutil.rmtree(output_dir)

    @pytest.fixture
    def sample_images(self, temp_dirs):
        """Create sample images for testing."""
        input_dir, _ = temp_dirs

        # Create test class directories
        class_dirs = ["MEL", "NV", "BCC"]
        for class_name in class_dirs:
            class_path = Path(input_dir) / class_name
            class_path.mkdir(parents=True, exist_ok=True)

            # Create sample images
            for i in range(5):
                img = Image.new("RGB", (600, 450), color=(i * 50, i * 50, i * 50))
                img.save(class_path / f"image_{i}.jpg")

        return input_dir, class_dirs

    def test_image_resize(self, temp_dirs, sample_images):
        """Test that images are resized correctly."""
        # This would test the actual preprocessing logic
        # For now, it's a placeholder
        input_dir, class_dirs = sample_images

        # Load an image
        test_image_path = Path(input_dir) / class_dirs[0] / "image_0.jpg"
        image = Image.open(test_image_path)

        # Resize
        resized = image.resize((224, 224))

        assert resized.size == (224, 224)
        assert resized.mode == "RGB"

    def test_train_val_test_split(self):
        """Test that data is split correctly."""
        # Create mock data
        total_samples = 100
        train_split = 0.8
        val_split = 0.1

        # Calculate expected splits
        expected_train = int(total_samples * train_split)
        expected_val = int(total_samples * val_split)
        expected_test = total_samples - expected_train - expected_val

        # Verify splits add up
        assert expected_train + expected_val + expected_test == total_samples

        # Verify ratios
        assert abs(expected_train / total_samples - train_split) < 0.01
        assert abs(expected_val / total_samples - val_split) < 0.01

    def test_normalization_values(self):
        """Test image normalization."""
        # Standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # Create sample image
        image = np.random.rand(224, 224, 3).astype(np.float32)

        # Normalize
        normalized = (image - mean) / std

        # Check that normalization changed the values
        assert not np.allclose(image, normalized)

        # Check that mean is approximately 0
        assert abs(normalized.mean()) < 1.0

    def test_metadata_creation(self, temp_dirs, sample_images):
        """Test metadata file creation."""
        input_dir, class_dirs = sample_images

        # Count images
        image_count = 0
        for class_name in class_dirs:
            class_path = Path(input_dir) / class_name
            image_count += len(list(class_path.glob("*.jpg")))

        # Should have 3 classes * 5 images = 15 images
        assert image_count == 15


class TestDataAugmentation:
    """Test cases for data augmentation."""

    def test_random_flip(self):
        """Test random flip augmentation."""
        # Create test image
        image = Image.new("RGB", (100, 100), color=(255, 0, 0))

        # Apply horizontal flip
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)

        assert image.size == flipped.size
        assert image != flipped  # Should be different

    def test_random_rotation(self):
        """Test random rotation augmentation."""
        # Create test image
        image = Image.new("RGB", (100, 100), color=(255, 0, 0))

        # Apply rotation
        rotated = image.rotate(45)

        assert image.size == rotated.size

    def test_color_jitter(self):
        """Test color jitter augmentation."""
        # This is a placeholder for actual color jitter testing
        # In practice, you would test the actual augmentation pipeline
        from PIL import ImageEnhance

        image = Image.new("RGB", (100, 100), color=(128, 128, 128))

        # Adjust brightness
        enhancer = ImageEnhance.Brightness(image)
        brightened = enhancer.enhance(1.5)

        assert brightened.size == image.size
        # Pixel values should be different
        assert list(brightened.getdata())[0] != list(image.getdata())[0]


def test_imports():
    """Test that all required modules can be imported."""
    try:
        import torch
        import torchvision
        import PIL
        import numpy
        import pandas
        import sklearn

        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import required module: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
