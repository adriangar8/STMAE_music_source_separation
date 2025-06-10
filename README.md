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


# Models


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
- Fundamentals of Music Source Separation & Data Augmentation: https://source-separation.github.io/tutorial/intro/src_sep_101.html







