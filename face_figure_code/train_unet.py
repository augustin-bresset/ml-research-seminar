"""
Train a bias-free UNet Flow Matching model on CelebA grayscale 64x64.
Saves weights to unet_fm_celeba_gray64.pt (compatible with the analysis notebook).

Usage:
    python train_unet.py
    python train_unet.py --epochs 30 --batch-size 256 --n-train 160000 --lr 1e-4
    python train_unet.py --checkpoint ckpt_unet_fm_celeba_gray64.pt  # resume
"""

import argparse, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
import torchvision.transforms as T
from model import UNetFM

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epochs",     type=int,   default=30)
parser.add_argument("--batch-size", type=int,   default=256)
parser.add_argument("--n-train",    type=int,   default=160_000)
parser.add_argument("--lr",         type=float, default=1e-4)
parser.add_argument("--checkpoint", type=str,   default=None,
                    help="path to a full checkpoint dict to resume from")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE  = 64
CHANNELS  = 1
BASE_CH   = 64
CH_MULTS  = (1, 2, 4, 8)   # → [64, 128, 256, 512], ~13.3M params
TIME_DIM  = 256
GRAD_CLIP = 1.0
CKPT_FILE = "ckpt_unet_fm_celeba_gray64.pt"
OUT_FILE  = "unet_fm_celeba_gray64.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


# ── Model ─────────────────────────────────────────────────────────────────────
model = UNetFM(CHANNELS, BASE_CH, CH_MULTS, TIME_DIM).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"UNet: {n_params/1e6:.1f}M params | bias-free: {all('bias' not in n for n, _ in model.named_parameters())}")

# ── Dataset ───────────────────────────────────────────────────────────────────
transform = T.Compose([
    T.Grayscale(),
    T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize([0.5], [0.5]),
])

print(f"loading CelebA ({args.n_train} images)...")
hf_ds   = load_dataset("tglcourse/CelebA-faces-cropped-128", split="train")
tensors = [transform(hf_ds[i]["image"]) for i in tqdm(range(args.n_train), desc="preprocess")]
X_gpu   = torch.stack(tensors).to(device)
print(f"dataset on GPU: {X_gpu.shape}")

# ── Training setup ────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
scaler    = torch.amp.GradScaler("cuda")

start_epoch  = 0
train_losses = []

if args.checkpoint and os.path.exists(args.checkpoint):
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch  = ckpt["epoch"] + 1
    train_losses = ckpt.get("train_losses", [])
    print(f"resumed from epoch {start_epoch} | last loss={train_losses[-1]:.4f}")

model_c = torch.compile(model)
model_c.train()

# ── Training loop ─────────────────────────────────────────────────────────────
for epoch in range(start_epoch, args.epochs):
    t0 = time.time()
    epoch_loss, n_batches = 0.0, 0
    perm = torch.randperm(args.n_train, device=device)

    for start in range(0, args.n_train, args.batch_size):
        idx = perm[start : start + args.batch_size]
        x1  = X_gpu[idx]
        B   = x1.shape[0]
        x0  = torch.randn_like(x1)
        t   = torch.rand(B, device=device)
        xt  = t[:, None, None, None] * x1 + (1 - t[:, None, None, None]) * x0

        with torch.amp.autocast("cuda"):
            loss = F.mse_loss(model_c(t, xt), x1 - x0)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        n_batches  += 1

    scheduler.step()
    avg = epoch_loss / n_batches
    train_losses.append(avg)
    elapsed = time.time() - t0
    print(f"epoch {epoch+1:3d}/{args.epochs} | loss={avg:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

    # full checkpoint every epoch (resume-safe)
    torch.save({
        "epoch": epoch, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "train_losses": train_losses,
    }, CKPT_FILE)

# weights-only file for inference
torch.save(model.state_dict(), OUT_FILE)
print(f"done. weights saved to {OUT_FILE}")

