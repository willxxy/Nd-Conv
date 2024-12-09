import numpy as np

def get_data(N, n_N, M, n_M, seed=42, prefetching=False):
    np.random.seed(seed)
    dtype = np.float32
    if prefetching:
        A = np.random.rand(n_N, N, N).astype(dtype)
        B = np.random.rand(n_M, M, M).astype(dtype)
        C = np.zeros((n_N, n_M, N, N), dtype=dtype)
    else:
        A = np.random.rand(N, N, n_N).astype(dtype)
        B = np.random.rand(M, M, n_M).astype(dtype)
        C = np.zeros((N, N, n_N, n_M), dtype=dtype)
    return A, B, C