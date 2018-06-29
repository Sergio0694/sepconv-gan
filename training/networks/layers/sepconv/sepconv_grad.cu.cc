#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>

__global__ void SepconvGradKernel(
    const int ntasks,
    const float* grad,
    const float* input,
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

    // Clear the gradients
    for (int ivh = 0; ivh < kchannels; ivh++)
    {
        kv_grad[_k_offset + ivh] = 0.0;
        kh_grad[_k_offset + ivh] = 0.0;
    }
    
    // Calculate the gradients
    for (int iv = 0; iv < kchannels; iv++)
    {
        for (int ih = 0; ih < kchannels; ih++)
        {
            // Get the coordinates from the image patch for (iv, ih)
            int y_t = iy - kc_2 + iv;
            int x_t = ix - kc_2 + ih;
            if (y_t < 0 || y_t >= h || x_t < 0 || x_t >= w)
                continue;

            // Aggregate the output gradient across the batch
            float result = 0.0;
            for (int ic = 0; ic < 3; ic++)
                result += 
                    input[in * _3d_resolution + (y_t * w + x_t) * 3 + ic]
                    * grad[_grad_offset + ic];

            // Add the partial gradient to the target row and column
            kv_grad[_k_offset + iv] += result;
            kh_grad[_k_offset + ih] += result;
        }
    }
}

#define THREADS_PER_BLOCK_BACKWARD 256

void SepconvGradKernelLauncher(
    const float* grad,
    const float* input,
    const int n, 
    const int h, 
    const int w,
    const int kchannels,
    float* kv_grad,
    float* kh_grad)
{
    int ntasks = n * h * w;
    SepconvGradKernel<<<(ntasks + THREADS_PER_BLOCK_BACKWARD - 1) / THREADS_PER_BLOCK_BACKWARD, THREADS_PER_BLOCK_BACKWARD>>>(
        ntasks, grad, input, n, h, w, kchannels, kv_grad, kh_grad);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("SepConv launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}
