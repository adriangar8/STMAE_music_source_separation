# Introduction

This project focuses on Music Source Separation (MSS), which decomposes a mixed music signal into independent components such as vocals, drums, bass, and accompaniment—crucial for remixing, automatic transcription.This repository hosts all code, data, and experiment logs for our STMAE course project. The goal is to compare and improve two deep-learning architectures (NMF and U-Net) on multi-track music source separation.

# Repository Structure
```bash
.
├── README.md                           # Project overview and instructions
├── requirements.txt                    # Python dependencies
├── imgs                                # Image assets               
└── src                                 # Source code directory
    ├── data                            # Data preparation and augmentation
    │   ├── MUSDB18_Audio_Data_Augmentation.ipynb   
    │   ├── MUSDB18_Data_Crop2.ipynb                
    │   └── README.md                   # Details on dataset structure and usage
    └── models                          # Model implementations
        ├── NMF                         # Non-negative Matrix Factorization baseline
        │   └── README.md               # Usage and training instructions for NMF
        └── U‑Net                       # Lightweight U‑Net model
            └── README.md               # Usage and training instructions for U‑Net
```


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

--> See the data folder's README.md file for more details.


# Model Architectures for MSS
We implement two complementary MSS models:

## 1. Lightweight U‑Net
- What it is: Scaled‑down U‑Net (8→16→32 filters) inspired by TFC‑TDF‑UNet v3
- How it works: Encoder (Conv→BN→ReLU→Pool) → Bottleneck (64 + Dropout) → Decoder (TransposeConv + Skip)
- Why use it: Trainable on a single GPU (e.g. GTX 1650) as a low‑resource baseline

--> See the models folder's README.md file for more details.

## 2. Non-negative Matrix Factorization

- What it is: Split a spectrogram into two non‑negative parts:W = sound “shapes” (like tiny snippets of timbre), H = when each shape happens over time

- How it works: Start with random W and H.Update them again and again with a simple multiply‑and‑divide rule until W H looks like the original spectrogram.Use W H to build masks and convert back to audio.

- Why use it： Runs on a laptop CPU.Easy to see and edit what each component means.Good, classic baseline before deep‑learning models.

--> See the models folder's README.md file for more details.


#  Models at a Glance 
    
- **U‑Net (baseline)**  
  - Classic symmetric convolutional encoder–decoder  
  - Path: `src/models/U-Net/`
    
- **NMF (baseline)**  
  - Non‑negative Matrix Factorization baseline  
  - Path: `src/models/NMF/`

  
# Data Pipeline
For full details on data download, cropping and augmentation, please refer to src/README.md.




