#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>

__global__ void SepconvGradKernel(
    const int ntasks,
    const float* grad,
    const float* input,
    const float* kv,
    const float* kh,
    const int h, 
    const int w,
    const int kchannels,
    float* kv_grad,
    float* kh_grad)
{
    // Get offset, abort if over the threshold
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= ntasks) return;

    /*
    // Retrieve the current position
    int iy = (idx / (w * 2));
    int ix = (idx / 2)      % w;
    int im = (idx)          % 2;

    // Derived pitches
    int _3d_resolution = h * w * 3;
    int _k_offset = (iy * w + ix) * kchannels;
    int kc_2 = kchannels / 2;
    
    // Calculate the gradients
    //float grad_xy = grad[_in_offset + (iy * w + ix) * 3 + ic];
    if (im)
    {
        // Vertical axis
        for (int iv = 0; iv < kchannels; iv++)
        {
            for (int ih = 0; ih < kchannels; ih++)
            {
                float result = 0.0;
                int y_t = iy - kc_2 + iv;
                int x_t = ix - kc_2 + ih;
                if (y_t < 0 || y_t >= h || x_t < 0 || x_t >= w)
                    continue;
                for (int in = 0; in < n; in++)
                    for (int ic = 0; ic < 3; ic++)
                        result += 
                            inputs[in * _3d_resolution + (y_t * w + x_t) * 3 + ic]
                            * grad_xy;
                
            }
            kv_grad[_k_offset + iv] = result;
        }
    }
    else
    {
        // Horizontal axis
        for (int ih = 0; ih < kchannels; ih++)
        {
            float result = 0.0;
            for (int iv = 0; iv < kchannels; iv++)
            {
                int y_t = iy - kc_2 + iv;
                int x_t = ix - kc_2 + ih;
                if (y_t < 0 || y_t >= h || x_t < 0 || x_t >= w)
                    continue;
                for (int in = 0; in < n; in++)
                    for (int ic = 0; ic < 3; ic++)
                        result += 
                            inputs[in * _3d_resolution + (y_t * w + x_t) * 3 + ic]
                            * grad_xy;
            }
            kv_grad[_k_offset + ih] = result;
        }
    } */
}

#define THREADS_PER_BLOCK_BACKWARD 256

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
    int ntasks = h * w * 2;
    SepconvGradKernel<<<(ntasks + THREADS_PER_BLOCK_BACKWARD - 1) / THREADS_PER_BLOCK_BACKWARD, THREADS_PER_BLOCK_BACKWARD>>>(
        ntasks, grad, input, kv, kh, h, w, kchannels, kv_grad, kh_grad);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("SepConv launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}
