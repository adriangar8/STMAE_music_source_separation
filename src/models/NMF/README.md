# NMF-based Audio Source Separation

This script performs audio source separation using Non-negative Matrix Factorization (NMF) and K-means clustering on the MUSDB18HQ dataset. It separates mixed audio into four components: vocals, drums, bass, and other.

## Features

- Loads and processes audio files from MUSDB18HQ dataset
- Performs STFT to obtain magnitude and phase spectra
- Applies NMF for component decomposition
- Uses K-means clustering to group components
- Reconstructs separated sources
- Evaluates separation quality using cSDR and uSDR metrics
- Generates spectrogram visualizations
- Saves separated audio tracks

## Requirements

- Python 3.6+
- Libraries listed in requirements.txt

## Output

For each song in the test set, the script will:
- Save separated audio tracks as WAV files
- Generate spectrogram plots for vocals (GT and predicted)
- Create a metrics.txt file with average cSDR and uSDR scores

## Metrics

The script computes two evaluation metrics:
- cSDR (conventional Signal-to-Distortion Ratio)
- uSDR (unmasked Signal-to-Distortion Ratio)

## Note

The performance depends on the NMF parameters (number of components) and may need tuning for optimal results.
