#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>

#define OFFSET(y, x, c, w) (y * w + x) * 3
#define SET(y, x, h, w, target, value)      \
    if (y >= 0 && y < h && x >= 0 && x < w) \
        target += value

__global__ void DilatedSepconvKernel(
    const int ntasks,
    const float* input,
    const float* kv,
    const float* kh,
    const int h, 
    const int w,
    const int kchannels,
    const int d,
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

    // Locals
    int dilated = kchannels - d;        // Weights for the dilated convolution
    int side = dense + dilated * 2;     // Size of the effective image patch
    int dense_2 = d / 2;                // Max dense offset from center
    int y0 = iy - dense_2 - dilated;
    int x0 = ix - dense_2 - dilated;    // Top left starting point for the dilated convolution

    // Derived pitches
    int _k_offset = ((in * h + iy) * w + ix) * kchannels;
    int _n_offset = in * h * w * 3;

    // Dilated convolution
    float result = 0.0;
    bool angle = dilated % 4 != 0;      // The innermost pattern is the angled one
    int kchannels_left = kchannels;     // Counter for the leftover usable kernel area size
    for (int i = 0; i < d_2; i++)
    {
        // Corners
        SET(y0, x0, h, w, result, input[_n_offset + OFFSET(y0, x0, ic)]
            * kv[_k_offset] 
            * kh[_k_offset]);
        SET(y0 + side - 1, x0, h, w, result, input[_n_offset + OFFSET(y0  + side - 1, x0, ic)]
            * kv[_k_offset + kchannels - 1]
            * kh[_k_offset]);
        SET(y0, x0 + side - 1, h, w, result, input[_n_offset + OFFSET(y0, x0 + side - 1, ic)]
            * kv[_k_offset]
            * kh[_k_offset + kchannels - 1]);
        SET(y0 + side - 1, x0 + side - 1, h, w, result, input[_n_offset + OFFSET(y0 + side - 1, x0 + side - 1, ic)]
            * kv[_k_offset + kchannels - 1]
            * kh[_k_offset + kchannels - 1]);

        if (angle)
        {
            // Middle points
            SET(y0 + side / 2, x0, h, w, result, input[_n_offset + OFFSET(y0 + side / 2, x0, ic)]
                * kv[_k_offset + kchannels / 2] 
                * kh[_k_offset]);
            SET(y0, x0 + side / 2, h, w, result, input[_n_offset + OFFSET(y0, x0 + side / 2, ic)]
                * kv[_k_offset]
                * kh[_k_offset + kchannels / 2]);
            SET(y0 + side / 2, x0 + side - 1, h, w, result, input[_n_offset + OFFSET(y0 + side / 2, x0 + side - 1, ic)]
                * kv[_k_offset + kchannels / 2]
                * kh[_k_offset + kchannels - 1]);
            SET(y0 + side - 1, x0 + side / 2, h, w, result, input[_n_offset + OFFSET(y0 + side - 1, x0 + side / 2, ic)]
                * kv[_k_offset + kchannels - 1]
                * kh[_k_offset + kchannels / 2]);

            // Locals
            int step = ((edge - 1) / 2) - 1
            int diff = (side - edge) / 2
            int side_length = (step + diff + 1) * 2
            int middle_offset = step + diff + 1
            int second_half_offset = middle_offset + diff
            int kernel_second_half_offset = 0; // TODO

            // Patterns
            for (int j = 0; j < step; j++)
            {
                // First half
                SET(y0 + j + 1, x0, h, w, result, input[_n_offset + OFFSET(y0 + j + 1, x0, ic)]
                    * kv[_k_offset + j + 1]
                    * kh[_k_offset]);
                SET(y0, x0 + j + 1, h, w, result, input[_n_offset + OFFSET(y0, x0 + j + 1, ic)]
                    * kv[_k_offset]
                    * kh[_k_offset + j + 1]);
                SET(y0 + j + 1, x0 + side_length, h, w, result, input[_n_offset + OFFSET(y0 + j + 1, x0 + side_length, ic)]
                    * kv[_k_offset + j + 1]
                    * kh[_k_offset + kchannels_left]);
                SET(y0 + side_length, x0 + j + 1, h, w, result, input[_n_offset + OFFSET(y0 + side_length, x0 + j + 1, ic)]
                    * kv[_k_offset + kchannels_left]
                    * kh[_k_offset + j + 1]);

                // Second half
                SET(y0 + j + 1 + second_half_offset, x0, h, w, result, input[_n_offset + OFFSET(y0 + j + 1 + second_half_offset, x0, ic)]
                    * kv[_k_offset + j + 1 + kernel_second_half_offset]
                    * kh[_k_offset]);
                SET(y0, x0 + j + 1 + second_half_offset, h, w, result, input[_n_offset + OFFSET(y0, x0 + j + 1 + second_half_offset, ic)]
                    * kv[_k_offset]
                    * kh[_k_offset + j + 1 + kernel_second_half_offset]);
                SET(y0 + j + 1 + second_half_offset, x0 + side_length, h, w, result, input[_n_offset + OFFSET(y0 + j + 1 + second_half_offset, x0 + side_length, ic)]
                    * kv[_k_offset + j + 1 + kernel_second_half_offset]
                    * kh[_k_offset + kchannels_left]);
                SET(y0 + side_length, x0 + j + 1 + second_half_offset, h, w, result, input[_n_offset + OFFSET(y0 + side_length, x0 + j + 1 + second_half_offset, ic)]
                    * kv[_k_offset + kchannels_left]
                    * kh[_k_offset + j + 1 + kernel_second_half_offset]);
            }
        }
        else
        {
            int step = 0;
            int diff = 0;
            int side_length = 0; // TODO
            int kernel_side_size = 0;

            for (int j = 0; j < step; j++)
            {
                SET(y0 + j + 1 + diff, x0, h, w, result, input[_n_offset + OFFSET(y0 + j + 1 + diff, x0, ic)]
                    * kv[_k_offset + j + 1]
                    * kh[_k_offset]);
                SET(y0, x0 + j + 1 + diff, h, w, result, input[_n_offset + OFFSET(y0, x0 + j + 1 + diff, ic)]
                    * kv[_k_offset]
                    * kh[_k_offset + j + 1]);
                SET(y0 + j + 1 + diff, x0 + side_length, h, w, result, input[_n_offset + OFFSET(y0 + j + 1 + diff, x0 + side_length, ic)]
                    * kv[_k_offset + j + 1]
                    * kh[_k_offset + kernel_side_size]);
                SET(y0 + side_length, x0 + j + 1 + diff, h, w, result, input[_n_offset + OFFSET(y0 + side_length, x0 + j + 1 + diff, ic)]
                    * kv[_k_offset + kernel_side_size]
                    * kh[_k_offset + j + 1]);
            }
        }
    }

    // Dense convolution
    int d_2 = d / 2;                                    // Max offset from the center for the dense convolution
    int _k_dense_offset = _k_offset + dilated / 2;      // Offset for the dense kernel inside the two linear vectors

    // Limits
    int iv_low = iy >= d_2 ? 0 : d_2 - iy;
    int iv_high = iy + d_2 < h ? d : d - (iy + d_2 - h) - 1;
    int ih_low = ix >= d_2 ? 0 : d_2 - ix;
    int ih_high = ix + d_2 < w ? d : d - (ix + d_2 - w) - 1;
    
    // Perform the separable convolution
    for (int iv = iv_low; iv < iv_high; iv++)
    {
        for (int ih = ih_low; ih < ih_high; ih++)
        {
            result +=
                input[_n_offset + ((iy - kc_2 + iv) * w + (ix - kc_2 + ih)) * 3 + ic]
                * kv[_k_dense_offset + iv]
                * kh[_k_dense_offset + ih];
        }
    }
    output[idx] = result;
}

#define THREADS_PER_BLOCK_FORWARD 512

/* =============
 * Parameters
 * =============
 * kchannels: the total depth of each convolution vector. The
 *   total area for the convolution operation is kchannels^2.
 * d: the side of the convolution kernel that is reserved for
 * dense convolution. The total image area that will be covered
 * by the operation will be (d + 2*(kchannels - d))^2. */
void DilatedSepconvKernelLauncher(
    const float* input, 
    const float* kv,
    const float* kh,
    const int n, 
    const int h, 
    const int w,
    const int kchannels,
    const int d,
    float* output)
{
    int ntasks = n * h * w * 3;
    DilatedSepconvKernel<<<(ntasks + THREADS_PER_BLOCK_FORWARD - 1) / THREADS_PER_BLOCK_FORWARD, THREADS_PER_BLOCK_FORWARD>>>(
        ntasks, input, kv, kh, h, w, kchannels, output);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("SepConv launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
}
