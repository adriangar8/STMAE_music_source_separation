import os
import librosa
import librosa.display
import numpy as np
import soundfile as sf
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans

DATASET_PATH = "/home/agarcias/datasets/musdb18hq"
SCRIPT_DIR = "/home/agarcias/traditional_method/results/NMF"
SR = 44100
N_COMPONENTS = 8
N_FFT = 2048
HOP_LENGTH = 512
TARGETS = ['vocals', 'drums', 'bass', 'other']

def load_mixture_and_sources(song_dir):
    y_mix, _ = librosa.load(os.path.join(song_dir, "mixture.wav"), sr=SR, mono=True)
    sources = {
        target: librosa.load(os.path.join(song_dir, f"{target}.wav"), sr=SR, mono=True)[0]
        for target in TARGETS
    }
    return y_mix, sources


def stft_audio(y):
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.abs(S), np.angle(S)


def istft_audio(mag, phase):
    S = mag * np.exp(1j * phase)
    return librosa.istft(S, hop_length=HOP_LENGTH)


def apply_nmf(V):
    model = NMF(n_components=N_COMPONENTS, init='random', solver='mu', beta_loss='kullback-leibler',
                max_iter=200, random_state=0)
    W = model.fit_transform(V)
    H = model.components_
    return W, H


def cluster_components(W, n_clusters=4):
    return KMeans(n_clusters=n_clusters, random_state=0).fit(W.T).labels_


def reconstruct_by_cluster(W, H, phase, labels, cluster_id):
    idxs = np.where(labels == cluster_id)[0]
    if len(idxs) == 0:
        return np.zeros(W.shape[0])
    V_hat = np.dot(W[:, idxs], H[idxs])
    return istft_audio(V_hat, phase)


def compute_sdr_metrics(estimated, reference):
    min_len = min(len(estimated), len(reference))
    estimated = estimated[:min_len]
    reference = reference[:min_len]
    noise = estimated - reference
    signal_power = np.sum(reference ** 2)
    noise_power = np.sum(noise ** 2)
    csdr = 10 * np.log10(signal_power / (1e-10 + noise_power))
    threshold = 0.01 * np.max(np.abs(reference))
    masked_noise = np.where(np.abs(noise) > threshold, noise, 0)
    masked_noise_power = np.sum(masked_noise ** 2)
    usdr = 10 * np.log10(signal_power / (1e-10 + masked_noise_power))
    return csdr, usdr


def plot_spectrogram(mag, title, path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(mag, ref=np.max),
                             sr=SR, hop_length=HOP_LENGTH,
                             y_axis='log', x_axis='time')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def evaluate_on_test_set():
    test_songs = sorted(glob(os.path.join(DATASET_PATH, "test", "*")))
    results = {target: [] for target in TARGETS}

    for song_dir in tqdm(test_songs, desc="Evaluating songs"):
        song_name = os.path.basename(song_dir)
        output_prefix = os.path.join(SCRIPT_DIR, song_name)

        y_mix, sources = load_mixture_and_sources(song_dir)
        mag, phase = stft_audio(y_mix)
        W, H = apply_nmf(mag)
        labels = cluster_components(W)

        cluster_signals = [
            reconstruct_by_cluster(W, H, phase, labels, i)
            for i in range(4)
        ]

        used = set()
        assignments = {}
        for target in TARGETS:
            best_score = -np.inf
            best_idx = None
            for i, signal in enumerate(cluster_signals):
                if i in used:
                    continue
                csdr, _ = compute_sdr_metrics(signal, sources[target])
                if csdr > best_score:
                    best_score = csdr
                    best_idx = i
            assignments[target] = best_idx
            used.add(best_idx)

        for target in TARGETS:
            cluster_idx = assignments[target]
            estimated = cluster_signals[cluster_idx]
            reference = sources[target]
            csdr, usdr = compute_sdr_metrics(estimated, reference)
            results[target].append((csdr, usdr))

            sf.write(f"{output_prefix}_{target}_PRED.wav", estimated, SR)

            if target == "vocals":
                est_mag, _ = stft_audio(estimated)
                ref_mag, _ = stft_audio(reference)
                plot_spectrogram(ref_mag, f"{song_name} GT Vocals",
                                 f"{output_prefix}_vocals_GT.png")
                plot_spectrogram(est_mag, f"{song_name} Pred Vocals",
                                 f"{output_prefix}_vocals_PRED.png")

    with open(os.path.join(SCRIPT_DIR, "metrics.txt"), "w") as f:
        for target in TARGETS:
            csdrs = [x[0] for x in results[target]]
            usdrs = [x[1] for x in results[target]]
            line = f"{target.upper()}: cSDR = {np.mean(csdrs):.2f} dB, uSDR = {np.mean(usdrs):.2f} dB\n"
            print(line.strip())
            f.write(line)


if __name__ == "__main__":
    evaluate_on_test_set()
