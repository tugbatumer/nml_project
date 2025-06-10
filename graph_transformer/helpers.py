import random 
import numpy as np
import torch
import os

from scipy import signal
from scipy.stats import skew, kurtosis
import numpy as np


def seed_everything(seed: int):
    # Python random module
    random.seed(seed)
    # Numpy random module
    np.random.seed(seed)
    # Torch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Set PYTHONHASHSEED environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior in cudnn (may slow down your training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_softmax_thresholded_graph(dist, beta=10.0, keep_ratio=0.9):
    N = dist.shape[0]
    edge_index = []
    edge_attr = []

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
            if i != j:
                edge_index.append([i, j])
                edge_attr.append(weights[j])

    edge_index = torch.tensor(edge_index).T
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    return edge_index, edge_attr


fs = 250 # Sampling frequency

bp_filter = signal.butter(4, (0.5, 30), btype="bandpass", output="sos", fs=fs)

def time_filtering(x: np.ndarray) -> np.ndarray:
    """Filter signal in the time domain"""
    return signal.sosfiltfilt(bp_filter, x, axis=0).copy()


def fft_filtering(x: np.ndarray) -> np.ndarray:
    """Compute FFT and only keep"""
    x = np.abs(np.fft.fft(x, axis=0))
    x = np.log(np.where(x > 1e-8, x, 1e-8))

    win_len = x.shape[0]
    # Only frequencies b/w 0.5 and 30Hz
    return x[int(0.5 * win_len // 250) : 30 * win_len // 250]

def fft_filtering_features(x: np.ndarray) -> np.ndarray:
    filtered = fft_filtering(x)
    return filtered.T

def handcrafted_features(x: np.ndarray) -> np.ndarray:

    mean_abs_diff1 = np.mean(np.abs(x[1:] - x[:-1]), axis=0)
    mean_abs_diff2 = np.mean(np.abs(x[2:] - x[:-2]), axis=0)
    std      = np.std(x, axis=0)
    skewness = np.nan_to_num(skew(x, axis=0), nan=0.0)
    kurt     = np.nan_to_num(kurtosis(x, axis=0), nan=0.0)

    fft_vals = np.abs(np.fft.rfft(x, axis=0))
    freqs = np.fft.rfftfreq(x.shape[0], d=1/fs)

    def band_power(fmin, fmax):
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        return np.mean(fft_vals[idx], axis=0)


    delta = band_power(0.5, 4)
    theta = band_power(4, 8)
    alpha = band_power(8, 13)
    beta  = band_power(13, 30)

    total_power = np.sum(fft_vals, axis=0)

    # Avoid division by zero
    relative_delta = np.divide(delta, total_power, out=np.zeros_like(delta), where=total_power != 0)
    relative_theta = np.divide(theta, total_power, out=np.zeros_like(theta), where=total_power != 0)
    relative_alpha = np.divide(alpha, total_power, out=np.zeros_like(alpha), where=total_power != 0)
    relative_beta  = np.divide(beta,  total_power, out=np.zeros_like(beta),  where=total_power != 0)


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
    ], axis=1)
    
    features = (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-8)
    return features
    
def handcrafted_features_combined(x: np.ndarray) -> np.ndarray:

    mean_abs_diff1 = np.mean(np.abs(x[1:] - x[:-1]), axis=0)
    mean_abs_diff2 = np.mean(np.abs(x[2:] - x[:-2]), axis=0)
    std      = np.std(x, axis=0)
    skewness = np.nan_to_num(skew(x, axis=0), nan=0.0)
    kurt     = np.nan_to_num(kurtosis(x, axis=0), nan=0.0)

    fft_vals = np.abs(np.fft.rfft(x, axis=0))
    freqs = np.fft.rfftfreq(x.shape[0], d=1/fs)

    def band_power(fmin, fmax):
        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        return np.mean(fft_vals[idx], axis=0)


    delta = band_power(0.5, 4)
    theta = band_power(4, 8)
    alpha = band_power(8, 13)
    beta  = band_power(13, 30)

    total_power = np.sum(fft_vals, axis=0)

    # Avoid division by zero
    relative_delta = np.divide(delta, total_power, out=np.zeros_like(delta), where=total_power != 0)
    relative_theta = np.divide(theta, total_power, out=np.zeros_like(theta), where=total_power != 0)
    relative_alpha = np.divide(alpha, total_power, out=np.zeros_like(alpha), where=total_power != 0)
    relative_beta  = np.divide(beta,  total_power, out=np.zeros_like(beta),  where=total_power != 0)


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

    fft_result = fft_filtering(x)


    features = np.concatenate([
        fft_result.T,  # shape (19, 354)
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
    ], axis=1)
    
    features = (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-8)
    return features


def batch_to_dense_E(edge_index, edge_attr, batch_size, num_nodes):
    """
    Given one (edge_index, edge_attr) for a single 19‐node graph,
    produce a dense tensor of shape (batch_size, num_nodes, num_nodes, 1).
    """
    device = edge_attr.device
    E_dense = torch.zeros((batch_size, num_nodes, num_nodes, 1), device=device)
    rows = edge_index[0]
    cols = edge_index[1]
    weights = edge_attr.view(-1)
    for idx in range(rows.size(0)):
        i = rows[idx].item()
        j = cols[idx].item()
        w = weights[idx]
        E_dense[:, i, j, 0] = w
        E_dense[:, j, i, 0] = w
    return E_dense