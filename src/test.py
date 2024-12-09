import argparse
from functools import partial
import numpy as np

from src.utils.data_utils import get_data
from src.utils.timer_utils import timeit
from src.utils.viz_utils import plot_histogram
from src.nd_conv.np_conv import test_numpy_fft
from src.nd_conv.mlx_conv import test_mlx_fft
from src.nd_conv.torch_conv import test_torch_fft

def get_args():
    parser = argparse.ArgumentParser(description="Test NumPy and MLX FFT Convolution Implementations")
    parser.add_argument('--N', type=int, default=1024, help="Size of the first dimension for A")
    parser.add_argument('--n_N', type=int, default=2, help="Size of the second dimension for A")
    parser.add_argument('--M', type=int, default=16, help="Size of the first dimension for B")
    parser.add_argument('--n_M', type=int, default=3, help="Size of the second dimension for B")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--prefetching', action='store_true', default=False, help="Enable prefetching mode")
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()

def main(args):

    A, B, C = get_data(args.N, args.n_N, args.M, args.n_M, args.seed, args.prefetching)
    
    C_mlx = np.zeros_like(C, dtype=np.float32)
    C_torch = np.zeros_like(C, dtype=np.float32)
    C_np = np.zeros_like(C, dtype=np.float32)
    
    print("Running np backend...")
    timeit(partial(test_numpy_fft, A, B, C_np, args.prefetching))

    print("Running mlx backend...")
    timeit(partial(test_mlx_fft, A, B, C_mlx, args.device, args.prefetching))
    
    print("Running torch backend...")
    timeit(partial(test_torch_fft, A, B, C_torch, args.device, args.prefetching))

    if np.allclose(C_np, C_mlx, atol=1e-5) and np.allclose(C_np, C_torch, atol=1e-5):
        print("Test Passed: NumPy, MLX, and torch outputs are equivalent within tolerance.")
    else:
        print('NumPy and MLX outputs are not equivalent.')
        difference = np.abs(C_np - C_mlx)
        max_diff = difference.max()
        mean_diff = difference.mean()
        std_diff = difference.std()
        print(f"Test Failed: Maximum difference between outputs is {max_diff}")
        print(f"Mean difference: {mean_diff}, Std deviation: {std_diff}")
        plot_histogram(difference, 'mlx')
        
        print('torch and MLX outputs are not equivalent.')
        difference = np.abs(C_np - C_torch)
        max_diff = difference.max()
        mean_diff = difference.mean()
        std_diff = difference.std()
        print(f"Test Failed: Maximum difference between outputs is {max_diff}")
        print(f"Mean difference: {mean_diff}, Std deviation: {std_diff}")
        plot_histogram(difference, 'torch')

if __name__ == '__main__':
    args = get_args()
    main(args)
