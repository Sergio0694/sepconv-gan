#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>

/* ==============
 * Forward op
 * =========== */

__global__ void SepConvKernel(
    const int ntasks,
    const float* inputs,
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
    
    // Perform the separable convolution
    float result = 0.0;
    for (int iv = 0; iv < kchannels; iv++)
    {
        for (int ih = 0; ih < kchannels; ih++)
        {
            int y_t = iy - kc_2 + iv;
            int x_t = ix - kc_2 + ih;
            if (y_t < 0 || y_t >= h || x_t < 0 || x_t >= w)
                continue;
            result +=
                inputs[_n_offset + (y_t * w + x_t) * 3 + ic]
                * kv[_k_offset + iv]
                * kh[_k_offset + ih];
        }
    }
    output[idx] = result;
}

#define THREADS_PER_BLOCK_FORWARD 512

 void SepConvKernelLauncher(
    const float* inputs, 
    const float* kv,
    const float* kh,
    const int n, 
    const int h, 
    const int w,
    const int kchannels,
    float* output)
{
    int ntasks = n * h * w * 3;
    SepConvKernel<<<(ntasks + THREADS_PER_BLOCK_FORWARD - 1) / THREADS_PER_BLOCK_FORWARD, THREADS_PER_BLOCK_FORWARD>>>(
        ntasks, inputs, kv, kh, h, w, kchannels, output);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("SepConv launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}
