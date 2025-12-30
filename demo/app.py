"""
Streamlit demo app for Skin Lesion Classification.

This app demonstrates the trained model for classifying skin lesions
into 8 different categories including melanoma.
"""

import io
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# Page configuration
st.set_page_config(
    page_title="Skin Lesion Classification Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
CLASS_NAMES = [
    "MEL - Melanoma",
    "NV - Melanocytic Nevus",
    "BCC - Basal Cell Carcinoma",
    "AK - Actinic Keratosis",
    "BKL - Benign Keratosis",
    "DF - Dermatofibroma",
    "VASC - Vascular Lesion",
    "SCC - Squamous Cell Carcinoma",
]

CLASS_DESCRIPTIONS = {
    "MEL - Melanoma": "A serious form of skin cancer that develops in melanocytes. Early detection is critical.",
    "NV - Melanocytic Nevus": "Commonly known as a mole. Usually benign but should be monitored for changes.",
    "BCC - Basal Cell Carcinoma": "The most common type of skin cancer. Rarely spreads but should be treated.",
    "AK - Actinic Keratosis": "A precancerous skin condition caused by sun damage. Can develop into cancer.",
    "BKL - Benign Keratosis": "Non-cancerous skin growth, also called seborrheic keratosis.",
    "DF - Dermatofibroma": "A benign skin growth commonly found on the legs.",
    "VASC - Vascular Lesion": "Lesions related to blood vessels, usually benign.",
    "SCC - Squamous Cell Carcinoma": "Second most common skin cancer. Can spread if not treated.",
}

MODEL_ARCHITECTURES = {
    "ResNet50": "resnet50",
    "ResNet101": "resnet101",
    "EfficientNet-B0": "efficientnet_b0",
    "EfficientNet-B4": "efficientnet_b4",
    "DenseNet121": "densenet121",
    "MobileNet V2": "mobilenet_v2",
}


@st.cache_resource
def load_model(model_name: str) -> Tuple[nn.Module, transforms.Compose]:
    """
    Load a pretrained model for demonstration.

    Note: This is a demo function. In production, you would load
    the actual trained weights from S3 or local storage.
    """
    # Build transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Build model architecture
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(CLASS_NAMES))
    elif model_name == "efficientnet_b4":
        model = models.efficientnet_b4(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(CLASS_NAMES))
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, len(CLASS_NAMES))
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(CLASS_NAMES))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Set to evaluation mode
    model.eval()

    # Note: In production, load actual trained weights here:
    # model.load_state_dict(torch.load("path/to/weights.pth"))

    return model, transform


def predict(image: Image.Image, model: nn.Module, transform: transforms.Compose) -> Dict:
    """
    Make prediction on an image.

    Args:
        image: PIL Image
        model: PyTorch model
        transform: Image transformation pipeline

    Returns:
        Dictionary with prediction results
    """
    # Transform image
    img_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()

    # Get all class probabilities
    all_probs = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": confidence,
        "all_probabilities": all_probs,
    }


def main():
    """Main Streamlit application."""

    # Header
    st.title("🔬 Skin Lesion Classification Demo")
    st.markdown("""
    This demo showcases a deep learning model for classifying skin lesions into 8 categories,
    including melanoma. Upload an image to get instant predictions with confidence scores.

    ⚠️ **Disclaimer**: This is a demonstration model for portfolio purposes only.
    Not for medical diagnosis. Always consult a dermatologist for professional evaluation.
    """)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model selection
        selected_model_name = st.selectbox(
            "Model Architecture",
            options=list(MODEL_ARCHITECTURES.keys()),
            index=0,
            help="Select which deep learning architecture to use"
        )

        st.markdown("---")

        # About section
        st.header("📊 About")
        st.markdown("""
        **Dataset**: ISIC 2019
        **Images**: 25,331 dermoscopic images
        **Classes**: 8 skin lesion types
        **Framework**: PyTorch 2.1+
        **Transfer Learning**: ImageNet pretrained

        [View on GitHub](https://github.com/progerjkd/skin-lesion-classification)
        """)

        st.markdown("---")

        # Model info
        with st.expander("🧠 Model Information"):
            st.markdown(f"""
            **Architecture**: {selected_model_name}

            **Training Details**:
            - Optimizer: Adam
            - Learning Rate: 0.001
            - Batch Size: 32
            - Data Augmentation: Yes
            - Image Size: 224x224

            **Production Pipeline**:
            - AWS SageMaker Training
            - Automated Retraining
            - Drift Detection
            - Model Registry
            """)

        # Class info
        with st.expander("📚 Lesion Types"):
            for class_name, description in CLASS_DESCRIPTIONS.items():
                st.markdown(f"**{class_name}**")
                st.markdown(f"_{description}_")
                st.markdown("")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Image")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a skin lesion image",
            type=["jpg", "jpeg", "png"],
            help="Upload a dermoscopic image of a skin lesion"
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Add analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Loading model..."):
                    model, transform = load_model(MODEL_ARCHITECTURES[selected_model_name])

                with st.spinner("Analyzing image..."):
                    results = predict(image, model, transform)

                # Store results in session state
                st.session_state["results"] = results
                st.session_state["analyzed"] = True
        else:
            st.info("👆 Upload an image to get started")

            # Show example images info
            st.markdown("---")
            st.markdown("""
            **Don't have an image?**

            You can download sample dermoscopic images from:
            - [ISIC Archive](https://www.isic-archive.com/)
            - [DermNet](https://dermnetnz.org/)

            Or use images from the `demo/examples/` directory in the repository.
            """)

    with col2:
        st.header("📊 Results")

        if st.session_state.get("analyzed", False):
            results = st.session_state["results"]

            # Predicted class
            st.subheader("Prediction")
            predicted_class = results["predicted_class"]
            confidence = results["confidence"]

            # Color code based on confidence
            if confidence > 0.8:
                color = "green"
            elif confidence > 0.6:
                color = "orange"
            else:
                color = "red"

            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: rgba(0,128,0,0.1); border-left: 5px solid {color};">
                <h3 style="margin: 0; color: {color};">{predicted_class}</h3>
                <p style="margin: 5px 0 0 0; font-size: 18px;">
                    Confidence: {confidence:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown(f"_{CLASS_DESCRIPTIONS[predicted_class]}_")

            # Confidence warning
            if confidence < 0.6:
                st.warning("⚠️ Low confidence prediction. Consider consulting a professional.")

            st.markdown("---")

            # All probabilities
            st.subheader("All Class Probabilities")

            # Sort by probability
            sorted_probs = sorted(
                results["all_probabilities"].items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Display as progress bars
            for class_name, prob in sorted_probs:
                st.markdown(f"**{class_name}**")
                st.progress(prob)
                st.caption(f"{prob:.1%}")

            st.markdown("---")

            # Disclaimer
            st.error("""
            ⚠️ **Medical Disclaimer**

            This prediction is for demonstration purposes only and should NOT be used for
            medical diagnosis. Always consult a qualified dermatologist for professional
            evaluation of skin lesions.
            """)

        else:
            st.info("Upload and analyze an image to see results here.")

            # Project highlights
            st.markdown("---")
            st.subheader("🚀 Project Highlights")
            st.markdown("""
            **MLOps Pipeline Features**:
            - ✅ Automated training pipeline
            - ✅ Model versioning & registry
            - ✅ Drift detection
            - ✅ Automated retraining
            - ✅ CI/CD with GitHub Actions
            - ✅ Infrastructure as Code (Terraform)
            - ✅ Docker containerization
            - ✅ CloudWatch monitoring

            **Technical Stack**:
            - AWS SageMaker
            - PyTorch
            - Terraform
            - Docker
            - Step Functions
            - S3 + ECR
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>
            Built by <a href="https://github.com/progerjkd">Roger Vasconcelos</a> |
            <a href="https://github.com/progerjkd/skin-lesion-classification">View on GitHub</a> |
            <a href="mailto:proger.mv@gmail.com">Contact</a>
        </p>
        <p style="font-size: 12px;">
            This is a portfolio demonstration project showcasing MLOps best practices with AWS SageMaker.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
