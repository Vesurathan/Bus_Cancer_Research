"""
generator_model.py -- the report generator: a pretrained ViT-B/16 patch encoder
feeding a 3-layer Transformer decoder over the local word-level report
vocabulary (gen_vocab.py), with input/output embeddings tied. Greedy decoding
produces the report string end-to-end from the image.

Shared by train_generator.py (training) and draft_agent.py (inference), so the
architecture and preprocessing stay identical across both.
"""
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

from gen_vocab import VOCAB_SIZE, PAD_ID, BOS_ID, EOS_ID, detokenize
import config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

D_MODEL = 256
N_HEAD = 8
N_LAYERS = 3
FF_DIM = 512
DROPOUT = 0.1


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ReportGenerator(nn.Module):
    """ViT-B/16 patch encoder -> 3-layer Transformer decoder, weight-tied."""

    def __init__(self, vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_len=None):
        super().__init__()
        self.max_len = max_len or config.MAX_LEN
        self.encoder = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.enc_proj = nn.Linear(self.encoder.num_features, d_model)

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(self.max_len, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model, N_HEAD, FF_DIM, DROPOUT, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, N_LAYERS)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.out_proj.weight = self.tok_emb.weight  # weight tying

    def encode_image(self, images):
        feats = self.encoder.forward_features(images)  # (B, 197, enc_dim)
        return self.enc_proj(feats)                     # (B, 197, d_model)

    def _decode_step(self, mem, ys):
        L = ys.size(1)
        pos = torch.arange(L, device=ys.device).unsqueeze(0)
        x = self.tok_emb(ys) + self.pos_emb(pos)
        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=ys.device), diagonal=1)
        pad_mask = (ys == PAD_ID)
        h = self.decoder(x, mem, tgt_mask=causal_mask, tgt_key_padding_mask=pad_mask)
        return self.out_proj(h)

    def forward(self, images, tgt_in):
        """tgt_in: (B, L) ids, teacher-forced (BOS + content, no trailing EOS needed)."""
        mem = self.encode_image(images)
        return self._decode_step(mem, tgt_in)

    @torch.no_grad()
    def greedy_generate(self, images, device):
        self.eval()
        mem = self.encode_image(images)
        B = images.size(0)
        ys = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(self.max_len - 1):
            logits = self._decode_step(mem, ys)[:, -1]
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD_ID), nxt)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            done = done | (nxt == EOS_ID)
            if done.all():
                break
        return ys


class Generator:
    """Loads a trained checkpoint and maps one image -> report string."""

    def __init__(self, ckpt_path, device):
        self.device = device
        self.model = ReportGenerator(max_len=config.MAX_LEN).to(device).eval()
        state = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(state)
        self.tf = get_transform()

    @torch.no_grad()
    def generate(self, image_path):
        img = Image.open(image_path).convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.device)
        ids = self.model.greedy_generate(x, self.device)[0].tolist()
        return detokenize(ids)
