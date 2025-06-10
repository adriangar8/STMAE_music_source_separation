## 1. SE‑Enhanced BandSplitRNN
### Key Idea
Builds on the original BandSplitRNN (BSRNN) by adding Squeeze‑and‑Excitation (SE) layers after each residual block. These channel‑wise attention modules automatically reweight feature channels, improving the model’s focus on critical time–frequency information.

### Architecture
- Sub‑Band Split: Partition the input STFT spectrogram into several non‑overlapping sub‑bands and project each into a shared feature space.
- Dual‑Path BLSTM: Apply bidirectional LSTMs alternately over the time and frequency dimensions, with residual connections across blocks.
- Mask Estimation: Use sub‑band–specific MLPs followed by a GLU to predict complex masks and reconstruct the full spectrogram.

![Alt text](https://www.google.com/url?sa=i&url=https%3A%2F%2Fpaperswithcode.com%2Fpaper%2Fmusic-source-separation-with-band-split-rnn&psig=AOvVaw0E4kJQqNeRYTpc08tkyfNG&ust=1749662306197000&source=images&opi=89978449)

*Figure 1: Complete architecture overview showing sub-band processing paths*

### Parameter Count
Approximately 32.5 M parameters; the added SE layers incur a negligible increase while delivering significant performance gains.

### References
- paper: https://arxiv.org/abs/2209.15174
- pytorch implementation: https://github.com/amanteur/BandSplitRNN-Pytorch
