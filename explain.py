"""
explain.py -- lesion localisation for the ViT malignancy model via gradient-based
attention (Grad-CAM adapted to a Vision Transformer). Answers "where in the image
did the model look?" so the PDF can overlay a heatmap on the suspicious region.

We hook the last transformer block, take the gradient of the malignancy logit
w.r.t. the patch tokens, weight the token activations by those gradients, reshape
the 196 patch tokens to a 14x14 map, ReLU + upsample to the image, and blend.
"""
import numpy as np
import torch
from PIL import Image

import config
from malignancy_model import build_model, get_transform


class GradCAMViT:
    def __init__(self, ckpt_path, device=None):
        self.device = device or config.DEVICE
        self.model = build_model().to(self.device).eval()
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.tf = get_transform(train=False)
        self._act = None
        self._grad = None
        # Hook the SECOND-to-last block: this ViT is CLS-token pooled, so the last
        # block's output patch tokens are unused (zero gradient). One block earlier
        # the patch tokens still feed the CLS via the final block's attention, so
        # their gradients are informative for localisation.
        block = self.model.blocks[-2]
        block.register_forward_hook(self._fwd)
        block.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self._act = o.detach()

    def _bwd(self, m, gi, go):
        self._grad = go[0].detach()

    def heatmap(self, image_path):
        """Return P(malignant) and a HxW heatmap in [0,1] aligned to the input."""
        img = Image.open(image_path).convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.device)
        x.requires_grad_(True)
        self.model.zero_grad()
        logit = self.model(x)                     # (1,1)
        prob = torch.sigmoid(logit).item()
        logit.backward()

        act = self._act[0]                        # (tokens, dim)
        grad = self._grad[0]
        # drop the CLS token -> 196 patch tokens
        if act.shape[0] == 197:
            act, grad = act[1:], grad[1:]
        # per-token importance = sum over channels of (activation * gradient),
        # robust for ViTs; fall back to magnitude if the signed map is empty
        cam = (act * grad).sum(-1)                 # (196,)
        cam = torch.relu(cam)
        if float(cam.max()) <= 0:
            cam = (act * grad).sum(-1).abs()
        n = int(cam.shape[0] ** 0.5)               # 14
        cam = cam.reshape(1, 1, n, n)
        cam = torch.nn.functional.interpolate(cam, size=(224, 224),
                                              mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return prob, cam


def _jet(v):
    """Minimal jet colormap: v in [0,1] -> (r,g,b) in [0,255]."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1) * 255


def save_overlay(image_path, cam, out_path, alpha=0.5, sigma=8):
    """Smooth the CAM into a blob, colour it (jet), and blend over the US image."""
    try:
        from scipy.ndimage import gaussian_filter
        cam = gaussian_filter(cam, sigma=sigma)
    except Exception:
        pass
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    base = np.asarray(img).astype("float32")
    heat = _jet(cam)
    a = (alpha * cam)[..., None]                  # transparent where cold
    blended = base * (1 - a) + heat * a
    Image.fromarray(blended.clip(0, 255).astype("uint8")).save(out_path)
    return out_path
