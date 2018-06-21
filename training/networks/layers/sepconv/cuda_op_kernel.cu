#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 512

/* ==============
 * Backwards op
 * =========== */

 void SepConvKernelLauncher(
    const float* inputs, 
    const float* kh,
    const float* kv,
    const int n, 
    const int h, 
    const int w,
    const int kchannels,
    float* output);
{
    int ntasks = n * h * w * 3;
    SepConvKernel<<<(tasks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
        ntasks, inputs, kh, kv, h, w, kchannels, output);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("SepConv launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}

__global__ void SepConvKernel(
    const int ntasks,
    const float* inputs,
    const float* kh,
    const float* kv,
    const int h, 
    const int w,
    const int kchannels 
    float* output)
{
    // Get offset, abort if over the threshold
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= ntasks) return;

    // Calculate reusable pitches
    int _2d_resolution = h * w;                         // 2D resolution for input, kernels and output
    int _3d_resolution = _2d_resolution * 3;            // RGB image, 3 channels

    // Retrieve the current position
    int in = (idx / _3d_resolution);
    int ic = (idx / _2d_resolution) % 3;
    int iy = (idx / w)              % h;
    int ix = (idx)                  % w;

    // Derived pitches
    int _nc_ioffset = in * _3d_resolution + ic * _2d_resolution;
    int _n_koffset = in *  _2d_resolution * kchannels;  // Batch offset for the kernels
    int _xy_koffset = iy * w + ix;                      // 2D coordinate for the kernels
    
    // Perform the separable convolution
    float result = 0.0;
    for (int ih = 0; ih < kchannels; ih++)
    {
        int _ih_koffset = ih * _2d_resolution; // Reusable in inner loop
        for (int iv = 0; iv < kchannels; iv++)
            result +=
                inputs[_nc_ioffset + iy * w + ix]
                * kh[_n_koffset + _ih_koffset + _xy_koffset]
                * kv[_n_koffset + iv * _2d_resolution + _xy_koffset];
    }
    output[idx] = result;
}

/* ==============
 * Backwards op
 * =========== */

 void SepConvGradKernelLauncher(
    const float* grad, 
    const float* kh,
    const float* kv,
    const float* source,
    const int n, 
    const int h, 
    const int w,
    const int kchannels,
    float* backprop_kh,
    float* backprop_kv);
{
    //InputKernel<<<,THREADS_PER_BLOCK>>>(grad, kh, kv, source, h, w, kchannels, backprop_kh, backprop_kv); TODO
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("SepConvGrad launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}

__global__ void SepConvGradKernel(
    const float* grad, 
    const float* kh,
    const float* kv,
    const float* source,
    const int h, 
    const int w,
    const int kchannels,
    float* backprop_kh,
    float* backprop_kv)
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO
}
