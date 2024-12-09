import argparse
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from src.utils.data_utils import get_data
from src.utils.timer_utils import timeit
from src.utils.viz_utils import plot_performance_comparison
from src.nd_conv.np_conv import test_numpy_fft
from src.nd_conv.mlx_conv import test_mlx_fft
from src.nd_conv.torch_conv import test_torch_fft

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_min_power', type=int, default=5)  # 2^5 = 32
    parser.add_argument('--N_max_power', type=int, default=12)  # 2^11 = 2048
    parser.add_argument('--n_N', type=int, default=2)
    parser.add_argument('--M', type=int, default=16)
    parser.add_argument('--n_M', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prefetching', action='store_true')
    parser.add_argument('--loops', type=int, default=10)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()

def benchmark_size(N, args):
    A, B, C = get_data(N, args.n_N, args.M, args.n_M, args.seed, args.prefetching)
    C_mlx = np.zeros_like(C, dtype=np.float32)
    C_torch = np.zeros_like(C, dtype=np.float32)
    print('Size of A:', A.shape, 'Size of B:', B.shape, 'Size of C:', C.shape)

    if args.device == 'cpu':
        C_np = np.zeros_like(C, dtype=np.float32)
        np_time = timeit(partial(test_numpy_fft, A, B, C_np, args.prefetching), number=args.loops, repeat=args.repeat).mean
    else:
        np_time = None
    mlx_time = timeit(partial(test_mlx_fft, A, B, C_mlx, args.device, args.prefetching), number=args.loops, repeat=args.repeat).mean
    torch_time = timeit(partial(test_torch_fft, A, B, C_torch, args.device, args.prefetching), number=args.loops, repeat=args.repeat).mean
    
    return np_time, mlx_time, torch_time

def main(args):
    powers = range(args.N_min_power, args.N_max_power + 1)
    N_values = [2**p for p in powers]
    np_times, mlx_times, torch_times = [], [], []

    for N in N_values:
        print(f"\nBenchmarking size N={N} (2^{int(np.log2(N))})")
        np_time, mlx_time, torch_time = benchmark_size(N, args)
        np_times.append(np_time)
        mlx_times.append(mlx_time)
        torch_times.append(torch_time)

    if args.device == 'cpu':
        np_times = np.array(np_times) / 1000
    else:
        np_times = None
    mlx_times = np.array(mlx_times) / 1000
    torch_times = np.array(torch_times) / 1000
    
    plot_performance_comparison(args, N_values, np_times, mlx_times, torch_times)
    print('\nBenchmarking complete.')
    
if __name__ == '__main__':
    main(get_args())