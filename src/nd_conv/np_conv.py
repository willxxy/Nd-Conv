import numpy as np
from numpy.fft import fft2, ifft2

### Optimized from https://laurentperrinet.github.io/sciblog/posts/2017-09-20-the-fastest-2d-convolution-in-the-world.html

def np_fftconvolve(A, B):
    return np.real(ifft2(fft2(A) * fft2(B, s=A.shape)))

def test_numpy_fft(A, B, C, prefetching=False):
    if prefetching:
        B_shape_0 = B.shape[0]
        A_shape_0 = A.shape[0]
        A_shape_last = A.shape[-2:]
        
        f_B = np.empty((B_shape_0, A_shape_last[0], A_shape_last[1]), dtype=np.complex64)
        f_A_buffer = np.empty(A_shape_last, dtype=np.complex64)
        fft_product = np.empty(A_shape_last, dtype=np.complex64)
        
        for i_M in range(B_shape_0):
            f_B[i_M] = fft2(B[i_M], s=A_shape_last)
        
        for i_N in range(A_shape_0):
            fft2(A[i_N], out=f_A_buffer)
            for i_M in range(B_shape_0):
                np.multiply(f_A_buffer, f_B[i_M], out=fft_product)
                C[i_N, i_M] = np.real(ifft2(fft_product))
    else:
        A_shape_0, A_shape_1 = A.shape[:2]
        A_shape_last = A.shape[-1]
        B_shape_last = B.shape[-1]
        
        f_B = np.empty((A_shape_0, A_shape_1, B_shape_last), dtype=np.complex64)
        f_A_buffer = np.empty((A_shape_0, A_shape_1), dtype=np.complex64)
        fft_product = np.empty((A_shape_0, A_shape_1), dtype=np.complex64)
        
        for i_M in range(B_shape_last):
            f_B[:, :, i_M] = fft2(B[:, :, i_M], s=(A_shape_0, A_shape_1))
        
        for i_N in range(A_shape_last):
            fft2(A[:, :, i_N], out=f_A_buffer)
            for i_M in range(B_shape_last):
                np.multiply(f_A_buffer, f_B[:, :, i_M], out=fft_product)
                C[:, :, i_N, i_M] = np.real(ifft2(fft_product))