# Example Images

This directory should contain example dermoscopic images for testing the demo app.

## Where to Get Example Images

### Option 1: ISIC Archive (Recommended)
1. Visit [ISIC Archive](https://www.isic-archive.com/)
2. Browse the gallery
3. Download a few sample images
4. Save them in this directory

### Option 2: DermNet
1. Visit [DermNet NZ](https://dermnetnz.org/)
2. Search for skin lesion images
3. Download examples (respect copyright)
4. Save them in this directory

### Option 3: Use Your Own Dataset
If you've downloaded the full ISIC 2019 dataset:
```bash
# Copy a few sample images from your dataset
cp ../data/raw/ISIC_*.jpg ./
```

## Recommended Images

For best demo results, include examples of:
- Melanoma (MEL)
- Melanocytic Nevus (NV)
- Basal Cell Carcinoma (BCC)
- Different image qualities and angles

## File Naming

Use descriptive names:
- `melanoma_example_1.jpg`
- `nevus_example_1.jpg`
- `bcc_example_1.jpg`

## Copyright Notice

Ensure you have proper rights to use any images. The ISIC Archive provides images
with appropriate licenses for research and educational purposes.
