# Introduction

This project focuses on Music Source Separation (MSS), which decomposes a mixed music signal into independent components such as vocals, drums, bass, and accompaniment—crucial for remixing, automatic transcription, and music education. We enhance the BandSplitRNN (BSRNN) model by integrating Squeeze‑and‑Excitation (SE) attention layers to improve its ability to focus on key time–frequency features. Additionally, we implement a lightweight U‑Net inspired by TFC‑TDF‑UNet v3 to evaluate performance in resource‑constrained environments. These complementary approaches explore the trade‑off between model complexity and separation quality.



# Audio Augmentation Pipelines for MUSDB18

In order to increase the training material, we implemented an augmentation pipeline for the audio files present in the dataset.

## Core Functionality

- Dataset Integration: Seamless loading and processing of MUSDB18 dataset
- Mono Conversion: Automatic stereo-to-mono conversion for simplified processing
- Multi-stem Processing: Individual handling of vocals, drums, bass, and other instrument tracks
- Audio Quality Assessment: Automatic validation of audio segments for energy and activity levels
- Energy Threshold: Minimum RMS energy validation (0.005)
- Silent Segment Detection: Activity ratio analysis (30% minimum)

## Audio Augmentation Techniques
- Pitch Shifting: Transpose audio by specified semitones (-2 to +2 semitones)
- Time Stretching: Modify playback speed while preserving pitch (0.8x to 1.2x)
- Dynamic Range Compression: Apply audio compression with configurable threshold and ratio
- Reverb Effects: Add spatial ambience with adjustable room size and damping parameters

Go check the data folder's README.md file for more details.


# Model Architectures for MSS
Below is a parallel, pipeline‑style overview of our two separation models:

## 1. SE‑Enhanced BandSplitRNN
### Core Functionality
- Sub‑Band Decomposition: Split input STFT into K non‑overlapping frequency bands
- Dual‑Path Modeling: Interleave time‑axis and frequency‑axis BLSTMs with residual connections
- Channel Attention: Squeeze‑and‑Excitation layers after each BLSTM to reweight feature channels
- Mask Estimation: Sub‑band MLPs + GLU to predict complex masks and reconstruct the full spectrogram
  
### Key Features
- Preserves original BSRNN flow while adding lightweight SE blocks
- Minimal parameter increase (~32.4 M → 32.5 M)
- Faster convergence and higher SDR on MUSDB18

### Training Details
- Loss: Weighted L1 on magnitude + phase (α = 0.5)
- Optimizer: Adam, lr=1e-3 (decay 0.98 every 2 epochs), batch size=2
- Regularization: Gradient clipping (max norm=5), early stopping after 10 stagnant epochs

### Usage
```bash
# train SE-Enhanced BSRNN

```

## 2. Lightweight U‑Net
### Core Functionality
- Encoder–Decoder Backbone: Standard U‑Net with reduced depth/channels for single‑GPU training
- Magnitude‑Only Prediction: 1×1 conv to estimate magnitude spectrogram, reuse mixture phase for reconstruction
- Skip Connections: Preserve fine‑grained detail via concatenation between encoder and decoder

### Key Features
- Inspired by TFC‑TDF‑UNet v3 but scaled down (filters 8→16→32 in encoder)
- Encoder: Conv2D→BatchNorm→ReLU→MaxPool ×3
- Bottleneck: Conv2D(64) + Dropout(0.3)
- Decoder: Conv2DTranspose + Skip + Dropout(0.2)

### Training Details
- Preprocessing: 16 kHz resampling, Hann‑window STFT (nfft=1024, hop=512)
- Optimizer: Adam, lr=1e-3, batch size=4, L2 weight decay=1e-5
- Padding: Zero‑pad to multiple of 8 in time/freq, crop output to original shape

### Usage
```bash
# train U‑Net

```






