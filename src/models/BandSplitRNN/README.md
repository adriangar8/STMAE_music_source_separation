## 1. SE‑Enhanced BandSplitRNN
### Key Idea
Builds on the original BandSplitRNN (BSRNN) by adding Squeeze‑and‑Excitation (SE) layers after each residual block. These channel‑wise attention modules automatically reweight feature channels, improving the model’s focus on critical time–frequency information.

### Architecture
- Sub‑Band Split: Partition the input STFT spectrogram into several non‑overlapping sub‑bands and project each into a shared feature space.
- Dual‑Path BLSTM: Apply bidirectional LSTMs alternately over the time and frequency dimensions, with residual connections across blocks.
- Mask Estimation: Use sub‑band–specific MLPs followed by a GLU to predict complex masks and reconstruct the full spectrogram.

## Architecture
<img src="https://raw.githubusercontent.com/adriangar8/STMAE_music_source_separation/tree/main/imgs/BSRNN_original.png" 
     alt="BSRNN Architecture Diagram" 
     style="width:80%; max-width:600px; display: block; margin: 0 auto;"/>


*Figure 1: Complete architecture overview showing sub-band processing paths*

*Figure 1: Complete architecture overview showing sub-band processing paths*

### Parameter Count
Approximately 32.5 M parameters; the added SE layers incur a negligible increase while delivering significant performance gains.

### References
- paper: https://arxiv.org/abs/2209.15174
- pytorch implementation: https://github.com/amanteur/BandSplitRNN-Pytorch
