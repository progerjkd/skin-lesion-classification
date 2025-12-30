"""
SageMaker inference script for skin lesion classification.
"""

import base64
import io
import json
import tarfile
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models


def _load_model_config(model_dir: str) -> dict:
    config_path = Path(model_dir) / "model_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def _build_model(model_name: str, num_classes: int) -> nn.Module:
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


def _build_transform(config: dict) -> transforms.Compose:
    image_size = int(config.get("image_size", 224))
    normalization = config.get("normalization", {})
    mean = normalization.get("mean", [0.485, 0.456, 0.406])
    std = normalization.get("std", [0.229, 0.224, 0.225])

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def model_fn(model_dir: str):
    tar_path = Path(model_dir) / "model.tar.gz"
    if tar_path.exists():
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(model_dir)

    config = _load_model_config(model_dir)
    model_name = config.get("model_architecture", "resnet50")
    num_classes = config.get("num_classes")
    if not num_classes:
        num_classes = len(config.get("class_names", [])) or 2

    model = _build_model(model_name, num_classes)

    model_path = Path(model_dir) / "model.pth"
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "model": model,
        "class_names": config.get("class_names"),
        "transform": _build_transform(config),
    }


def input_fn(request_body, content_type):
    if content_type in ("application/x-image", "application/octet-stream"):
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        return image

    if content_type in ("image/jpeg", "image/png"):
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        return image

    if content_type == "application/json":
        payload = json.loads(request_body)
        image_b64 = payload.get("image")
        if not image_b64:
            raise ValueError("Missing 'image' field in JSON payload")
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image

    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model_bundle):
    model = model_bundle["model"]
    transform = model_bundle["transform"]
    class_names = model_bundle.get("class_names")

    tensor = transform(input_data).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0).tolist()
        predicted_idx = int(torch.argmax(outputs, dim=1).item())

    response = {
        "predicted_index": predicted_idx,
        "probabilities": probabilities,
    }
    if class_names and predicted_idx < len(class_names):
        response["predicted_label"] = class_names[predicted_idx]

    return response


def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps(prediction), accept

    return json.dumps(prediction), "application/json"
