"""MobileNetV3 pooled-feature identity candidates.

The public embedding contract is deliberately small: RGB PIL input is converted by
``get_transform`` and the model returns row-wise L2-normalized pooled features.
Small is 576-D; the unfinetuned ImageNet Large reference is 960-D.
"""
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torchvision.models import (MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights,
                                mobilenet_v3_small, mobilenet_v3_large)
from torchvision.transforms import functional as TF
from PIL import Image, ImageOps

BACKBONES = {
    "mobilenet_v3_small": (mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V1),
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LetterboxTransform:
    def __init__(self, size: int = 160, augment: bool = False):
        self.size = int(size)
        self.augment = bool(augment)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = ImageOps.exif_transpose(image).convert("RGB")
        side = max(image.width, image.height)
        canvas = Image.new("RGB", (side, side), (128, 128, 128))
        canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        image = canvas.resize((self.size, self.size), Image.Resampling.BILINEAR)
        # ``augment`` is intentionally conservative and deterministic in the
        # default path; horizontal flip is the only training augmentation.
        if self.augment and torch.rand(()) < 0.5:
            image = ImageOps.mirror(image)
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def get_transform(size: int = 160, augment: bool = False):
    """Return the fixed PIL->tensor transform used by training and inference."""
    return LetterboxTransform(size=size, augment=augment)


class IdentityModel(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int | None = None):
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = backbone.classifier[0].in_features
        # The torchvision classifier is not part of this candidate. Keeping it
        # would add unused parameters and make checkpoint contracts ambiguous.
        self.backbone.classifier = nn.Identity()
        self.num_classes = num_classes
        self.head = nn.Linear(self.embedding_dim, num_classes) if num_classes else None

    def _pooled(self, x: torch.Tensor) -> torch.Tensor:
        # torchvision MobileNetV3 forward is features -> avgpool -> flatten -> classifier.
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self._pooled(x), p=2, dim=1, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.head is None:
            raise RuntimeError("this model has no classification head")
        # CE trains on the natural pooled representation; retrieval always
        # uses the normalized embedding returned by forward().
        return self.head(self._pooled(x))


def create_model(pretrained: bool = True, num_classes: int | None = None,
                 architecture: str = "mobilenet_v3_small") -> IdentityModel:
    if architecture not in BACKBONES:
        raise ValueError(f"Unsupported backbone: {architecture}")
    builder, weights = BACKBONES[architecture]
    return IdentityModel(builder(weights=weights if pretrained else None), num_classes=num_classes)


def load_model(checkpoint: Path | str | None, device: str = "cpu",
               architecture: str = "mobilenet_v3_small") -> IdentityModel:
    """Load a candidate checkpoint, or create pretrained model when checkpoint is None."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    target = torch.device(device)
    if checkpoint is None:
        model = create_model(pretrained=True, architecture=architecture)
    else:
        if architecture != "mobilenet_v3_small":
            raise ValueError("Training checkpoints currently support mobilenet_v3_small only")
        payload = torch.load(Path(checkpoint), map_location=target, weights_only=True)
        state = payload.get("model_state", payload) if isinstance(payload, Mapping) else payload
        num_classes = payload.get("num_classes") if isinstance(payload, Mapping) else None
        model = create_model(pretrained=False, num_classes=num_classes)
        model.load_state_dict(state, strict=True)
    return model.to(target).eval()
