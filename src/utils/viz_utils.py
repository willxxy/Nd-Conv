import matplotlib.pyplot as plt

def plot_histogram(difference, backend):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(difference.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of Differences')
    plt.xlabel('Absolute Difference')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    max_diff_per_axis = difference.max(axis=(0, 1))
    plt.plot(max_diff_per_axis, marker='o', linestyle='-')
    plt.title('Maximum Difference per Slice')
    plt.xlabel('Slice Index')
    plt.ylabel('Maximum Difference')

    plt.tight_layout()
    plt.savefig(f'./pngs/histogram_{backend}.png')
    plt.close()
    
def plot_performance_comparison(N_values, np_times, mlx_times, torch_times, args):
    plt.figure(figsize=(10, 6))
    if args.device == 'cpu':
        plt.plot(N_values, np_times, 'o-', label='NumPy')
    plt.plot(N_values, mlx_times, 's-', label='MLX')
    plt.plot(N_values, torch_times, '^-', label='PyTorch')
    
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xlabel('Input Size (N)')
    plt.ylabel('Time (ms)')
    plt.title(f'{args.device.upper()} FFT Convolution Performance Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.xticks(N_values, [f'2^{int(np.log2(N))}' for N in N_values])
    plt.savefig(f'./pngs/performance_comparison_{args.device}.png')
    plt.close()