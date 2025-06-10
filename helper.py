import numpy as np
import torch
import os
from scipy.stats import skew, kurtosis
from scipy import signal
fs = 250

def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to build a softmax thresholded graph from a distance matrix
def build_softmax_thresholded_graph(dist, beta=10.0, keep_ratio=0.9):
    N = dist.shape[0]
    edge_index = []
    edge_attr = []
    added = set()

    for i in range(N):
        scores = -beta * (dist[i] ** 2)
        scores[i] = -np.inf
        weights = np.exp(scores)
        weights /= weights.sum()

        if keep_ratio == 1.0:
            keep = [j for j in range(N) if j != i]
        else:
            sorted_idx = np.argsort(-weights)
            cumulative = np.cumsum(weights[sorted_idx])
            cutoff = np.searchsorted(cumulative, keep_ratio) + 1
            keep = sorted_idx[:cutoff]

        for j in keep:
            if i != j and (i, j) not in added:
                edge_index.append([i, j])
                edge_attr.append(weights[j])
                # Add reverse edge (j, i) with same weight to make the graph undirected
                edge_index.append([j, i])
                edge_attr.append(weights[j])
                added.add((i, j))
                added.add((j, i))

    edge_index = torch.tensor(edge_index, dtype=torch.long).T  # shape [2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)   # shape [num_edges]
    return edge_index, edge_attr


def fft_filtering(x: np.ndarray) -> np.ndarray:
    """Compute FFT and only keep"""
    x = np.abs(np.fft.fft(x, axis=0))
    x = np.log(np.where(x > 1e-8, x, 1e-8))

    win_len = x.shape[0]
    # Only frequencies b/w 0.5 and 30Hz
    return x[int(0.5 * win_len // fs) : 30 * win_len // fs]

# Function to compute features from EEG data, optionally including FFT features
def handcrafted_features(x: np.ndarray, with_fft: bool) -> np.ndarray:
    mean_abs_diff1 = np.mean(np.abs(x[1:] - x[:-1]), axis=0)
    mean_abs_diff2 = np.mean(np.abs(x[2:] - x[:-2]), axis=0)
    std = np.std(x, axis=0)
    skewness = np.nan_to_num(skew(x, axis=0), nan=0.0)
    kurt = np.nan_to_num(kurtosis(x, axis=0), nan=0.0)

    fft_vals = np.abs(np.fft.rfft(x, axis=0))
    freqs = np.fft.rfftfreq(x.shape[0], d=1 / fs)

    def band_power(fmin, fmax):
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        return np.mean(fft_vals[idx], axis=0)

    delta = band_power(0.5, 4)
    theta = band_power(4, 8)
    alpha = band_power(8, 13)
    beta = band_power(13, 30)

    total_power = np.sum(fft_vals, axis=0)

    # Avoid division by zero
    relative_delta = np.divide(delta, total_power, out=np.zeros_like(delta), where=total_power != 0)
    relative_theta = np.divide(theta, total_power, out=np.zeros_like(theta), where=total_power != 0)
    relative_alpha = np.divide(alpha, total_power, out=np.zeros_like(alpha), where=total_power != 0)
    relative_beta = np.divide(beta, total_power, out=np.zeros_like(beta), where=total_power != 0)

    def hjorth_params(x_channel):
        dx = np.diff(x_channel)
        ddx = np.diff(dx)
        var_x = np.var(x_channel)
        var_dx = np.var(dx)
        var_ddx = np.var(ddx)
        mobility = np.sqrt(var_dx / var_x) if var_x != 0 else 0
        complexity = np.sqrt(var_ddx / var_dx) / mobility if var_dx != 0 and mobility != 0 else 0
        return mobility, complexity

    mobility = np.zeros(x.shape[1])
    complexity = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        mobility[i], complexity[i] = hjorth_params(x[:, i])

    zero_crossings = np.sum(np.abs(np.diff(np.signbit(x), axis=0)), axis=0)

    def shannon_entropy(signal):
        hist, _ = np.histogram(signal, bins=50, density=True)
        hist += 1e-12
        return -np.sum(hist * np.log2(hist))

    entropy = np.array([shannon_entropy(x[:, i]) for i in range(x.shape[1])])

    psd = fft_vals / (np.sum(fft_vals, axis=0, keepdims=True) + 1e-12)
    spectral_entropy = -np.sum(psd * np.log2(psd + 1e-12), axis=0)

    features = np.concatenate([
        mean_abs_diff1[:, None],
        mean_abs_diff2[:, None],
        std[:, None],
        skewness[:, None],
        kurt[:, None],
        delta[:, None], theta[:, None], alpha[:, None], beta[:, None],
        relative_delta[:, None], relative_theta[:, None], relative_alpha[:, None], relative_beta[:, None],
        mobility[:, None], complexity[:, None],
        zero_crossings[:, None],
        entropy[:, None],
        spectral_entropy[:, None],
    ], axis=1).T

    # Add FFT features (optional)
    if with_fft:
        fft_result = fft_filtering(x)
        features = np.concatenate([features, fft_result], axis=0)

    return features

# Function to create spectrograms from EEG data to feed into a CNN
def create_spectrogram(x: np.ndarray):
    spectrograms = []
    for i in range(x.shape[1]):
        f, t, Sxx = signal.spectrogram(
            x[:, i], fs=fs,
            nperseg=128, noverlap=64
        )
        mask = (f >= 0.5) & (f <= 30)
        Sxx = Sxx[mask, :]
        Sxx = np.log(Sxx + 1e-8)

        spectrograms.append(Sxx)

    spec_array = np.stack(spectrograms, axis=0)  # Shape: [channels, freq, time]
    return torch.tensor(spec_array, dtype=torch.float32)