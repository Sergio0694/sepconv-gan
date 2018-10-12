#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>

#define B(tensor, i) tensor[i]
#define G(tensor, i) tensor[i + 1]
#define R(tensor, i) tensor[i + 2]
#define CLIP(value, min, max) fmaxf(min, fminf(max, value))
#define INVERSE_LERP(value, min, max) (value - min) / (max - min)
#define LERP(factor, a, b) a + (b - a) * factor

#define THRESHOLD 3.1
#define MIN 2
#define MAX 8.1

__global__ void NearestShaderKernel(
    const int ntasks,
    const float* input,
    const float* frame_0,
    const float* frame_1,
    const float* delta_map,
    const int h,
    const int w,
    float* output)
{
    // Get offset, abort if over the threshold
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= ntasks) return;

    // Retrieve the current position
    const int in = (idx / h);
    const int iy = (idx)      % h;

    // Derived pitches
    const int _y_offset = (in * h * w * 3) + iy * w * 3;
    const int _delta_offset = (in * h * w) + iy * w;
    const int end = w * 3;
    
    // Apply the pixel shader
    for (int x = 0; x < end; x += 3)
    {
        const int xy = _y_offset + x;
        const int delta_xy = _delta_offset + x / 3;
        if (delta_map[delta_xy] < MAX)
        {
            const float b = (B(frame_0, xy) + B(frame_1, xy)) / 2;
            const float g = (G(frame_0, xy) + G(frame_1, xy)) / 2;
            const float r = (R(frame_0, xy) + R(frame_1, xy)) / 2;
            const float d = CLIP(delta_map[delta_xy], MIN, MAX);
            const float f = INVERSE_LERP(d, MIN, MAX);
            B(output, xy) = LERP(f, b, B(input, xy));
            G(output, xy) = LERP(f, g, G(input, xy));
            R(output, xy) = LERP(f, r, R(input, xy));
        }
        else
        {
            // Fallback
            B(output, xy) = B(input, xy);
            G(output, xy) = G(input, xy);
            R(output, xy) = R(input, xy);
        }
    }
}

#define THREADS_PER_BLOCK_FORWARD 512

void NearestShaderKernelLauncher(
    const float* input, 
    const float* frame_0,
    const float* frame_1,
    const float* delta_map,
    const int n, 
    const int h,
    const int w,
    float* output)
{
    int ntasks = n * h;
    NearestShaderKernel<<<(ntasks + THREADS_PER_BLOCK_FORWARD - 1) / THREADS_PER_BLOCK_FORWARD, THREADS_PER_BLOCK_FORWARD>>>(
        ntasks, input, frame_0, frame_1, delta_map, h, w, output);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("Shader launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}
