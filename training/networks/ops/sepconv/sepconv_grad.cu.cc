#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>

__global__ void SepconvGradKernel(
    const int ntasks,
    const float* grad,
    const float* input,
    const float* kv,
    const float* kh,
    const int n,
    const int h, 
    const int w,
    const int kchannels,
    float* kv_grad,
    float* kh_grad)
{
    // Get offset, abort if over the threshold
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= ntasks) return;

    // Retrieve the current position
    int in = (idx / (h * w));
    int iy = (idx / w)      % h;
    int ix = (idx)          % w;

    // Derived pitches
    int _3d_resolution = h * w * 3;
    int kc_2 = kchannels / 2;
    int _k_offset = ((in * h + iy) * w + ix) * kchannels;
    int _grad_offset = in * _3d_resolution + (iy * w + ix) * 3;

    // Limits
    int iv_low = iy >= kc_2 ? 0 : kc_2 - iy;
    int iv_high = iy + kc_2 < h ? kchannels : kchannels - (iy + kc_2 - h) - 1;
    int ih_low = ix >= kc_2 ? 0 : kc_2 - ix;
    int ih_high = ix + kc_2 < w ? kchannels : kchannels - (ix + kc_2 - w) - 1;
    
    // Calculate the gradients
    for (int iv = iv_low; iv < iv_high; iv++)
    {
        for (int ih = ih_low; ih < ih_high; ih++)
        {
            // Aggregate the output gradient across the batch
            int _c_offset = in * _3d_resolution + ((iy - kc_2 + iv) * w + (ix - kc_2 + ih)) * 3;
            float result = 
                input[_c_offset] * grad[_grad_offset]
                + input[_c_offset + 1] * grad[_grad_offset + 1]
                + input[_c_offset + 2] * grad[_grad_offset + 2];

            // Add the partial gradient to the target row and column
            kv_grad[_k_offset + iv] += result * kh[_k_offset + ih];
            kh_grad[_k_offset + ih] += result * kv[_k_offset + iv];
        }
    }
}

#define THREADS_PER_BLOCK_BACKWARD 512

void SepconvGradKernelLauncher(
    const float* grad,
    const float* input,
    const float* kv,
    const float* kh,
    const int n, 
    const int h, 
    const int w,
    const int kchannels,
    float* kv_grad,
    float* kh_grad)
{
    // Clear the gradient tensors
    int bytes = n * h * w * kchannels * 4;
    cudaError_t error = cudaMemset(kv_grad, 0, bytes);
    if (error != cudaSuccess)
        printf("Failed to set kv_grad to 0 with error \"%s\".\n", cudaGetErrorString(error));
    error = cudaMemset(kh_grad, 0, bytes);
    if (error != cudaSuccess)
        printf("Failed to set kh_grad to 0 with error \"%s\".\n", cudaGetErrorString(error));

    // Start the gradient kernel
    int ntasks = n * h * w;
    SepconvGradKernel<<<(ntasks + THREADS_PER_BLOCK_BACKWARD - 1) / THREADS_PER_BLOCK_BACKWARD, THREADS_PER_BLOCK_BACKWARD>>>(
        ntasks, grad, input, kv, kh, n, h, w, kchannels, kv_grad, kh_grad);
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        printf("SepConv launch failed with error \"%s\".\n", cudaGetErrorString(error));
}
