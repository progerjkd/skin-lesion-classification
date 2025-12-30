# Skin Lesion Classification - Interactive Demo

An interactive Streamlit web application for demonstrating skin lesion classification using deep learning.

![Demo Screenshot](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

## 🎯 What This Demo Does

This interactive web app allows you to:
- ✨ Upload dermoscopic images of skin lesions
- 🔍 Get instant predictions with confidence scores
- 📊 View probability distributions across all 8 lesion types
- 🧠 Switch between different model architectures (ResNet, EfficientNet, DenseNet, MobileNet)
- 📚 Learn about different types of skin lesions

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

1. **Navigate to the demo directory**:
   ```bash
   cd demo
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Demo

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`.

## 📖 How to Use

1. **Select Model Architecture** (sidebar):
   - Choose from ResNet50, EfficientNet, DenseNet121, etc.

2. **Upload an Image**:
   - Click "Browse files" or drag and drop
   - Supported formats: JPG, JPEG, PNG
   - Recommended: Dermoscopic images of skin lesions

3. **Analyze**:
   - Click "🔍 Analyze Image" button
   - Wait for the model to process (usually < 5 seconds)

4. **View Results**:
   - See predicted lesion type with confidence score
   - Review probability distribution for all classes
   - Read description of the predicted lesion type

## 🏥 Supported Lesion Types

The model classifies skin lesions into 8 categories:

| Code | Name | Description |
|------|------|-------------|
| MEL | Melanoma | Serious skin cancer requiring immediate attention |
| NV | Melanocytic Nevus | Common mole, usually benign |
| BCC | Basal Cell Carcinoma | Most common skin cancer, rarely spreads |
| AK | Actinic Keratosis | Precancerous condition from sun damage |
| BKL | Benign Keratosis | Non-cancerous skin growth |
| DF | Dermatofibroma | Benign fibrous growth |
| VASC | Vascular Lesion | Blood vessel-related lesions |
| SCC | Squamous Cell Carcinoma | Second most common skin cancer |

## 🧪 Testing the Demo

### Option 1: Download Sample Images

Visit [ISIC Archive](https://www.isic-archive.com/) to download sample dermoscopic images.

### Option 2: Use Example Images

Place test images in the `demo/examples/` directory:

```bash
# Example directory structure
demo/
├── examples/
│   ├── melanoma_sample.jpg
│   ├── nevus_sample.jpg
│   └── bcc_sample.jpg
└── app.py
```

See [`examples/README.md`](examples/README.md) for more information.

## ⚙️ Model Information

### Architecture Options

The demo supports multiple deep learning architectures:

- **ResNet50** (default): 25.6M parameters, good balance of speed and accuracy
- **ResNet101**: 44.5M parameters, more capacity
- **EfficientNet-B0**: 5.3M parameters, lightweight and fast
- **EfficientNet-B4**: 19M parameters, high accuracy
- **DenseNet121**: 8M parameters, efficient feature reuse
- **MobileNet V2**: 3.5M parameters, mobile-optimized

### Transfer Learning

All models use:
- **Pretrained Weights**: ImageNet initialization
- **Input Size**: 224x224 pixels
- **Normalization**: ImageNet mean/std
- **Output**: 8-class softmax

### Important Note

⚠️ **This demo uses pretrained ImageNet weights** for demonstration purposes. In production, you would load the actual trained weights from your SageMaker model:

```python
# Production usage
model.load_state_dict(torch.load("path/to/trained_weights.pth"))
```

To use your trained model:
1. Download model weights from S3 or SageMaker Model Registry
2. Save to `demo/models/model.pth`
3. Update the `load_model()` function in `app.py` to load these weights

## 🎨 Demo Features

### User Interface
- 📱 Responsive layout (works on mobile)
- 🎨 Professional design with color-coded predictions
- 📊 Interactive probability charts
- ℹ️ Informative tooltips and descriptions

### Technical Features
- ⚡ Model caching for fast predictions
- 🔄 Multiple architecture support
- 🖼️ Image preprocessing pipeline
- 📈 Confidence thresholds with warnings

## 🛡️ Disclaimer

**Medical Disclaimer**: This application is for **demonstration and educational purposes only**. It should **NOT** be used for medical diagnosis. Always consult a qualified dermatologist or healthcare professional for evaluation of skin lesions.

## 🎯 For Portfolio/Interviews

This demo showcases:

### Technical Skills
- ✅ Streamlit web application development
- ✅ PyTorch model integration
- ✅ Image preprocessing and computer vision
- ✅ Model architecture understanding
- ✅ User experience design

### MLOps Integration
While this demo runs locally, it's part of a larger MLOps pipeline:
- Models trained on AWS SageMaker
- Automated retraining via Step Functions
- Model Registry for version control
- Infrastructure as Code with Terraform
- CI/CD with GitHub Actions

### Use in Interviews
Prepare to discuss:
- How you built the UI/UX
- Trade-offs between model architectures
- How this integrates with the production pipeline
- Potential improvements (API backend, database, user auth)

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Free)
```bash
# Push to GitHub, then deploy on streamlit.io
# https://streamlit.io/cloud
```

### Option 2: Docker Container
```bash
# Build container
docker build -t skin-lesion-demo .

# Run container
docker run -p 8501:8501 skin-lesion-demo
```

### Option 3: AWS App Runner
Deploy as a containerized web service on AWS.

## 📊 Performance

- **Inference Time**: ~2-5 seconds per image (CPU)
- **Model Loading**: ~10-15 seconds (first run, then cached)
- **Memory Usage**: ~500MB - 2GB depending on model
- **Image Size**: Recommended < 10MB

## 🔧 Troubleshooting

### Model Takes Too Long to Load
- First load downloads ImageNet weights (~100-500MB depending on model)
- Subsequent loads are cached by Streamlit
- Consider using lighter models (MobileNet, EfficientNet-B0)

### Out of Memory Errors
- Close other applications
- Use a lighter model architecture
- Reduce image upload size

### Image Upload Fails
- Check file format (JPG, JPEG, PNG only)
- Ensure file size < 200MB
- Try converting to RGB format

## 🤝 Contributing

Ideas for improvements:
- [ ] Add batch prediction support
- [ ] Save prediction history
- [ ] Export results as PDF report
- [ ] Add image preprocessing options (contrast, brightness)
- [ ] Implement GradCAM for model explainability
- [ ] Add comparison mode (multiple models side-by-side)

## 📞 Contact

**Roger Vasconcelos**
- Email: proger.mv@gmail.com
- GitHub: [@progerjkd](https://github.com/progerjkd)
- Project: [github.com/progerjkd/skin-lesion-classification](https://github.com/progerjkd/skin-lesion-classification)

## 📄 License

MIT License - See main repository for details.

---

**Part of the Skin Lesion Classification MLOps Pipeline**
[View Full Project](https://github.com/progerjkd/skin-lesion-classification) | [Documentation](../README.md) | [Architecture](../ARCHITECTURE.md)
