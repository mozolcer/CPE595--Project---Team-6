from pathlib import Path

import timm
import torch
from torch import nn
from PIL import Image
from torchvision import transforms


CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

MODEL_ALIASES = {
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "convnext_tiny": "convnext_tiny",
    "swin_tiny_patch4_window7_224": "swin_tiny_patch4_window7_224",
    "vit_base_patch16_224": "vit_base_patch16_224",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_device(device=None):
    if device and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_name(model_key):
    return MODEL_ALIASES.get(model_key, model_key)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.net(x) + self.shortcut(x)


class SortSmartCNN(nn.Module):
    def __init__(self, num_classes=len(CLASSES), width=48, dropout=0.35):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            ConvBlock(width, width, dropout=0.05),
            ConvBlock(width, width * 2, stride=2, dropout=0.08),
            ConvBlock(width * 2, width * 2, dropout=0.08),
            ConvBlock(width * 2, width * 4, stride=2, dropout=0.12),
            ConvBlock(width * 4, width * 4, dropout=0.12),
            ConvBlock(width * 4, width * 6, stride=2, dropout=0.15),
            ConvBlock(width * 6, width * 6, dropout=0.15),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 6, num_classes),
        )

    def get_classifier(self):
        return self.classifier

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(model_key, num_classes=len(CLASSES), pretrained=True):
    if model_key == "custom_cnn_scratch":
        return SortSmartCNN(num_classes=num_classes)
    return timm.create_model(
        resolve_model_name(model_key),
        pretrained=pretrained,
        num_classes=num_classes,
    )


def build_transform(image_size=224, split="train", augment="light"):
    if split == "train":
        steps = [
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        if augment in {"light", "strong"}:
            steps.append(transforms.ColorJitter(0.25, 0.25, 0.25, 0.08))
        if augment == "strong":
            steps.append(transforms.RandAugment(num_ops=2, magnitude=9))
        steps.extend([transforms.ToTensor()])
        if augment == "strong":
            steps.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.18)))
        steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        return transforms.Compose(steps)

    resize_to = int(round(image_size / 0.875))
    return transforms.Compose([
        transforms.Resize(resize_to),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_checkpoint(checkpoint_path, device=None):
    checkpoint_path = Path(checkpoint_path)
    device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        checkpoint["model_key"],
        num_classes=len(checkpoint["classes"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, device


class DeepImageClassifier:
    def __init__(self, checkpoint_path, device=None):
        self.model, self.checkpoint, self.device = load_checkpoint(checkpoint_path, device=device)
        self.classes = self.checkpoint["classes"]
        self.transform = build_transform(
            image_size=self.checkpoint["image_size"],
            split="eval",
            augment=self.checkpoint["augment"],
        )

    def predict_path(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.predict_image(image)

    def predict_image(self, image):
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu()
        idx = int(probabilities.argmax())
        return {
            "prediction": self.classes[idx],
            "confidence": float(probabilities[idx]),
            "probabilities": {
                cls: float(probabilities[i]) for i, cls in enumerate(self.classes)
            },
        }


def predict_image(checkpoint_path, image_path, device=None):
    return DeepImageClassifier(checkpoint_path, device=device).predict_path(image_path)
