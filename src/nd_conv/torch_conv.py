import torch
from torch.fft import fft2, ifft2

def torch_fftconvolve(A, B):
    return torch.real(
        ifft2(
            fft2(A) * fft2(B, s=A.shape)
        )
    )

def test_torch_fft(A, B, C, device, prefetching=False):
    if device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('mps')
    A_torch = torch.from_numpy(A).to(torch.float32).to(device)
    B_torch = torch.from_numpy(B).to(torch.float32).to(device)
    C_torch = torch.from_numpy(C).to(torch.float32).to(device)
    
    if prefetching:
        B_shape_0 = B_torch.shape[0]
        A_shape_0 = A_torch.shape[0]
        A_shape_last = A_torch.shape[-2:]
        
        f_B = torch.empty((B_shape_0, A_shape_last[0], A_shape_last[1]), 
                         dtype=torch.complex64, device=A_torch.device)
        f_A_buffer = torch.empty(A_shape_last, 
                               dtype=torch.complex64, device=A_torch.device)
        fft_product = torch.empty(A_shape_last, 
                                dtype=torch.complex64, device=A_torch.device)
        
        for i_M in range(B_shape_0):
            f_B[i_M] = fft2(B_torch[i_M], s=A_shape_last)
        
        for i_N in range(A_shape_0):
            f_A_buffer = fft2(A_torch[i_N])
            for i_M in range(B_shape_0):
                torch.multiply(f_A_buffer, f_B[i_M], out=fft_product)
                C_torch[i_N, i_M] = torch.real(ifft2(fft_product))
                
    else:
        A_shape_0, A_shape_1 = A_torch.shape[:2]
        A_shape_last = A_torch.shape[-1]
        B_shape_last = B_torch.shape[-1]
        
        f_B = torch.empty((A_shape_0, A_shape_1, B_shape_last), 
                         dtype=torch.complex64, device=A_torch.device)
        f_A_buffer = torch.empty((A_shape_0, A_shape_1), 
                               dtype=torch.complex64, device=A_torch.device)
        fft_product = torch.empty((A_shape_0, A_shape_1), 
                                dtype=torch.complex64, device=A_torch.device)
        
        for i_M in range(B_shape_last):
            f_B[:, :, i_M] = fft2(B_torch[:, :, i_M], s=(A_shape_0, A_shape_1))
        
        for i_N in range(A_shape_last):
            f_A_buffer = fft2(A_torch[:, :, i_N])
            for i_M in range(B_shape_last):
                torch.multiply(f_A_buffer, f_B[:, :, i_M], out=fft_product)
                C_torch[:, :, i_N, i_M] = torch.real(ifft2(fft_product))
    
    C[:] = C_torch.cpu().numpy()