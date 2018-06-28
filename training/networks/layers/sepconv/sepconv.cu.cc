#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 512

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
    const int c,
    const int kchannels,
    float* output)
{
    // Get offset, abort if over the threshold
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= ntasks) return;

    // Calculate reusable pitches
    int _2d_resolution = h * w;                 // 2D resolution for input, kernels and output
    int _3d_resolution = _2d_resolution * c;

    // Retrieve the current position
    int in = (idx / _3d_resolution);
    int ic = (idx / _2d_resolution) % c;
    int iy = (idx / w)              % h;
    int ix = (idx)                  % w;

    // Derived pitches
    int _k_offset = (in * _2d_resolution + iy * w + ix) * kchannels;
    int _n_offset = in * _3d_resolution;
    int kc_2 = kchannels / 2;
    
    // Perform the separable convolution
    float result = 0.0;
    for (int iv = 0; iv < kchannels; iv++)
    {
        //int _ih_koffset = iv * _2d_resolution; // Reusable in inner loop
        for (int ih = 0; ih < kchannels; ih++)
        {
            int y_t = iy - kc_2 + iv;
            int x_t = ix - kc_2 + ih;
            if (y_t < 0 || y_t >= h || x_t < 0 || x_t >= w)
                continue;
            result +=
                inputs[_n_offset + (y_t * w + x_t) * c + ic]
                * kv[_k_offset + iv]
                * kh[_k_offset + ih];
        }
    }
    output[idx] = result;
}

 void SepConvKernelLauncher(
    const float* inputs, 
    const float* kv,
    const float* kh,
    const int n, 
    const int h, 
    const int w,
    const int c,
    const int kchannels,
    float* output)
{
    int ntasks = n * h * w * c;
    printf("Launching %d tasks\n", ntasks);
    SepConvKernel<<<(ntasks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
        ntasks, inputs, kv, kh, h, w, c, kchannels, output);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("SepConv launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}
