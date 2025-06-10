# Introduction

This project focuses on Music Source Separation (MSS), which decomposes a mixed music signal into independent components such as vocals, drums, bass, and accompaniment—crucial for remixing, automatic transcription, and music education. We enhance the BandSplitRNN (BSRNN) model by integrating Squeeze‑and‑Excitation (SE) attention layers to improve its ability to focus on key time–frequency features. Additionally, we implement a lightweight U‑Net inspired by TFC‑TDF‑UNet v3 to evaluate performance in resource‑constrained environments. These complementary approaches explore the trade‑off between model complexity and separation quality.



# Audio Augmentation Pipelines for MUSDB18

Go check the data folder's README.md file for more details.

# Models

## 1. SE‑Enhanced BandSplitRNN
### Key Idea
Builds on the original BandSplitRNN (BSRNN) by adding Squeeze‑and‑Excitation (SE) layers after each residual block. These channel‑wise attention modules automatically reweight feature channels, improving the model’s focus on critical time–frequency information.

### Architecture
- Sub‑Band Split: Partition the input STFT spectrogram into several non‑overlapping sub‑bands and project each into a shared feature space.
- Dual‑Path BLSTM: Apply bidirectional LSTMs alternately over the time and frequency dimensions, with residual connections across blocks.
- Mask Estimation: Use sub‑band–specific MLPs followed by a GLU to predict complex masks and reconstruct the full spectrogram.

### Parameter Count
Approximately 32.5 M parameters; the added SE layers incur a negligible increase while delivering significant performance gains.

### References
- paper: https://arxiv.org/abs/2209.15174
- pytorch implementation: https://github.com/amanteur/BandSplitRNN-Pytorch

## 2. Lightweight U‑Net
### Key Idea
Inspired by TFC‑TDF‑UNet v3, this model uses a standard U‑Net encoder–bottleneck–decoder backbone with reduced depth and channel counts, making it trainable on a single GPU (e.g., GTX 1650).

### Architecture
- Encoder: 3 blocks of Conv2D → BatchNorm → ReLU → MaxPool, with filter sizes 8 → 16 → 32
- Bottleneck: Conv2D with 64 channels + Dropout(0.3)
- Decoder: Conv2DTranspose layers with skip connections + Dropout(0.2)
- Output: 1×1 convolution to predict the magnitude spectrogram, then combine with the mixture phase for iSTFT reconstruction

### References
- TFC‑TDF‑UNet v3 paper: https://arxiv.org/abs/2211.08553
- Example implementation: https://github.com/jianchang512/vocal-separate







