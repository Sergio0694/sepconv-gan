#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>

__global__ void SepconvKernel(
    const int ntasks,
    const float* input,
    const float* kv,
    const float* kh,
    const int h, 
    const int w,
    const int kchannels,
    float* output)
{
    // Get offset, abort if over the threshold
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= ntasks) return;

    // Retrieve the current position
    int in = (idx / (h * w * 3));
    int iy = (idx / (w * 3))    % h;
    int ix = (idx / 3)          % w;
    int ic = (idx)              % 3;

    // Derived pitches
    int _k_offset = ((in * h + iy) * w + ix) * kchannels;
    int _n_offset = in * h * w * 3;
    int kc_2 = kchannels / 2;

    // Limits
    int iv_low = iy >= kc_2 ? 0 : kc_2 - iy;
    int iv_high = iy + kc_2 < h ? kchannels : kchannels - (iy + kc_2 - h) - 1;
    int ih_low = ix >= kc_2 ? 0 : kc_2 - ix;
    int ih_high = ix + kc_2 < w ? kchannels : kchannels - (ix + kc_2 - w) - 1;
    
    // Perform the separable convolution
    float result = 0.0;
    for (int iv = iv_low; iv < iv_high; iv++)
    {
        for (int ih = ih_low; ih < ih_high; ih++)
        {
            result +=
                input[_n_offset + ((iy - kc_2 + iv) * w + (ix - kc_2 + ih)) * 3 + ic]
                * kv[_k_offset + iv]
                * kh[_k_offset + ih];
        }
    }
    output[idx] = result;
}

#define THREADS_PER_BLOCK_FORWARD 512

 void SepconvKernelLauncher(
    const float* input, 
    const float* kv,
    const float* kh,
    const int n, 
    const int h, 
    const int w,
    const int kchannels,
    float* output)
{
    int ntasks = n * h * w * 3;
    SepconvKernel<<<(ntasks + THREADS_PER_BLOCK_FORWARD - 1) / THREADS_PER_BLOCK_FORWARD, THREADS_PER_BLOCK_FORWARD>>>(
        ntasks, input, kv, kh, h, w, kchannels, output);
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        printf("SepConv launch failed with error \"%s\".\n", cudaGetErrorString(error));
}
