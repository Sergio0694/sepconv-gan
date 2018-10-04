#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>

#define PIXEL_THRESHOLD 0.011764706

__global__ void NearestShaderKernel(
    const int ntasks,
    const float* input,
    const float* frame_0,
    const float* frame_1,
    const int h, 
    const int w,
    float* output)
{
    // Get offset, abort if over the threshold
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= ntasks) return;

    // Retrieve the current position
    int in = (idx / h);
    int iy = (idx)      % h;

    // Derived pitches
    int _y_offset = (in * h * w * 3) + iy * w * 3;
    int end = w * 3;
    
    // Apply the pixel shader
    for (int x = 0; x < end; x += 3)
    {
        if (abs(frame_0[_y_offset + x] - frame_1[_y_offset + x]) < PIXEL_THRESHOLD &&
            abs(frame_0[_y_offset + x + 1] - frame_1[_y_offset + x + 1]) < PIXEL_THRESHOLD &&
            abs(frame_0[_y_offset + x + 2] - frame_1[_y_offset + x + 2]) < PIXEL_THRESHOLD)
        {
            output[_y_offset + x] = (frame_0[_y_offset + x] + frame_1[_y_offset + x]) / 2;
            output[_y_offset + x + 1] = (frame_0[_y_offset + x + 1] + frame_1[_y_offset + x + 1]) / 2;
            output[_y_offset + x + 2] = (frame_0[_y_offset + x + 2] + frame_1[_y_offset + x + 2]) / 2;
        }
        else
        {
            output[_y_offset + x] = input[_y_offset + x];
            output[_y_offset + x + 1] = input[_y_offset + x + 1];
            output[_y_offset + x + 2] = input[_y_offset + x + 2];
        }
    }
}

#define THREADS_PER_BLOCK_FORWARD 512

void NearestShaderKernelLauncher(
    const float* input, 
    const float* frame_0,
    const float* frame_1,
    const int n, 
    const int h,
    const int w,
    float* output)
{
    int ntasks = n * h;
    NearestShaderKernel<<<(ntasks + THREADS_PER_BLOCK_FORWARD - 1) / THREADS_PER_BLOCK_FORWARD, THREADS_PER_BLOCK_FORWARD>>>(
        ntasks, input, frame_0, frame_1, h, w, output);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("SepConv launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}
