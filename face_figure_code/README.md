# Flow Matching UNet — Jacobian Analysis

Reproduction of Figure 3 from *"Generalization in Diffusion Models Arises from Geometry-Adaptive Harmonic Representations"* (ICLR 2024) using a flow matching UNet trained from scratch on CelebA grayscale 64×64.

## Files

| File | Role |
|---|---|
| `model.py` | UNet architecture (shared by notebook and training script) |
| `train_unet.py` | Train the model from scratch |
| `notebook_compare.ipynb` | Jacobian analysis and figures |
| `requirements.txt` | Dependencies (pip) |
| `pyproject.toml` | Dependencies (uv) |

## Setup

**With uv (recommended):**
```bash
uv sync
uv run jupyter lab notebook_compare.ipynb
```

**With pip:**
```bash
pip install -r requirements.txt
jupyter lab notebook_compare.ipynb
```

## Weights

The notebook needs `unet_fm_celeba_gray64.pt`. Two options:

**Option A — Download from Google Drive:**
Set `GDRIVE_FILE_ID` in cell 3 of the notebook with your file ID.
To get the ID: right-click the `.pt` file on drive.google.com → "Get link"
The link looks like: `https://drive.google.com/file/d/`**`THIS_IS_THE_ID`**`/view`

**Option B — Train from scratch:**
```bash
uv run python train_unet.py
# resumes automatically if ckpt_unet_fm_celeba_gray64.pt exists
```
