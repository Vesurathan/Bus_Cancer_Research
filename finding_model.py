"""
finding_model.py -- the interpretation classifier: a pretrained ViT backbone with
a 28-way multi-label head, plus an inference wrapper the finding agent calls.

Shared by train_finding_agent.py (training) and finding_agent.py (inference), so
the architecture and preprocessing stay identical across both.
"""
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from vocab import TERMS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_model():
    return timm.create_model("vit_base_patch16_224", pretrained=True,
                             num_classes=len(TERMS))


def get_transform(train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class Predictor:
    """Loads a trained checkpoint and maps one image -> {term: probability}."""
    def __init__(self, ckpt_path, device):
        self.device = device
        self.model = build_model().to(device).eval()
        state = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(state)
        self.tf = get_transform(train=False)

    @torch.no_grad()
    def predict_proba(self, image_path):
        img = Image.open(image_path).convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.device)
        probs = torch.sigmoid(self.model(x)).squeeze(0).cpu().tolist()
        return {t: float(p) for t, p in zip(TERMS, probs)}
