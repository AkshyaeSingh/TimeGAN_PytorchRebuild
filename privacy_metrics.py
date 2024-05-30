# privacy_metrics.py
import numpy as np
from sklearn.neighbors import NearestNeighbors

def nearest_neighbor_distance_ratio(real_data, synthetic_data):
    real_data = np.asarray(real_data).reshape(-1, real_data[0].shape[-1])
    synthetic_data = np.asarray(synthetic_data).reshape(-1, synthetic_data[0].shape[-1])

    nbrs_real = NearestNeighbors(n_neighbors=1).fit(real_data)
    nbrs_synthetic = NearestNeighbors(n_neighbors=1).fit(synthetic_data)

    distances_real, _ = nbrs_real.kneighbors(real_data)
    distances_synthetic, _ = nbrs_synthetic.kneighbors(synthetic_data)

    nndr = np.mean(distances_real) / np.mean(distances_synthetic)
    return nndr

def k_anonymity(synthetic_data, k=5):
    synthetic_data = np.asarray(synthetic_data).reshape(-1, synthetic_data[0].shape[-1])
    nbrs = NearestNeighbors(n_neighbors=k).fit(synthetic_data)
    _, indices = nbrs.kneighbors(synthetic_data)
    return np.mean(np.unique(indices, axis=1).shape[1] >= k)

def l_diversity(synthetic_data, sensitive_attribute_idx=0, l=2):
    synthetic_data = np.asarray(synthetic_data).reshape(-1, synthetic_data[0].shape[-1])
    sensitive_values = synthetic_data[:, sensitive_attribute_idx]
    unique_counts = np.unique(sensitive_values, return_counts=True)[1]
    return np.mean(unique_counts >= l)
