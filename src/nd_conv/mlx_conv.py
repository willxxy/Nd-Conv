import mlx.core as mx
import numpy as np

def mlx_fftconvolve(A, B):
    return mx.real(
        mx.fft.ifft2(
            mx.fft.fft2(A, axes=[-2, -1]) *
            mx.fft.fft2(B, s=A.shape[-2:], axes=[-2, -1])
        )
    )

def test_mlx_fft(A, B, C, device, prefetching=False):
    if device == 'cpu':
        mx.set_default_device(mx.cpu)
    else:
        mx.set_default_device(mx.gpu)
    A_mlx = mx.array(A)
    B_mlx = mx.array(B)
    C_mlx = mx.array(C)
    
    if prefetching:
        B_shape_0 = B_mlx.shape[0]
        A_shape_0 = A_mlx.shape[0]
        A_shape_last = A_mlx.shape[-2:]
        
        for i_N in range(A_shape_0):
            f_A_buffer = mx.fft.fft2(A_mlx[i_N], s=A_shape_last, axes=[-2, -1])
            for i_M in range(B_shape_0):
                f_B = mx.fft.fft2(B_mlx[i_M], s=A_shape_last, axes=[-2, -1])
                fft_product = f_A_buffer * f_B
                C_mlx[i_N, i_M] = mx.real(mx.fft.ifft2(fft_product, axes=[-2, -1]))
    else:
        A_shape_0, A_shape_1 = A_mlx.shape[:2]
        A_shape_last = A_mlx.shape[-1]
        B_shape_last = B_mlx.shape[-1]
        
        for i_N in range(A_shape_last):
            f_A_buffer = mx.fft.fft2(A_mlx[:, :, i_N], s=(A_shape_0, A_shape_1), axes=[-2, -1])
            for i_M in range(B_shape_last):
                f_B = mx.fft.fft2(B_mlx[:, :, i_M], s=(A_shape_0, A_shape_1), axes=[-2, -1])
                fft_product = f_A_buffer * f_B
                C_mlx[:, :, i_N, i_M] = mx.real(mx.fft.ifft2(fft_product, axes=[-2, -1]))
    
    C[:] = np.array(C_mlx).astype(np.float32)