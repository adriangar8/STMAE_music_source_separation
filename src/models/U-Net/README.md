# Lightweight U‑Net
## Key Idea
Inspired by TFC‑TDF‑UNet v3, this model uses a standard U‑Net encoder–bottleneck–decoder backbone with reduced depth and channel counts, making it trainable on a single GPU (e.g., GTX 1650).

## Architecture
- Encoder: 3 blocks of Conv2D → BatchNorm → ReLU → MaxPool, with filter sizes 8 → 16 → 32
- Bottleneck: Conv2D with 64 channels + Dropout(0.3)
- Decoder: Conv2DTranspose layers with skip connections + Dropout(0.2)
- Output: 1×1 convolution to predict the magnitude spectrogram, then combine with the mixture phase for iSTFT reconstruction

## References
- TFC‑TDF‑UNet v3 paper: https://arxiv.org/abs/2211.08553
- Example implementation: https://github.com/jianchang512/vocal-separate
- Fundamentals of Music Source Separation & Data Augmentation: https://source-separation.github.io/tutorial/intro/src_sep_101.html
