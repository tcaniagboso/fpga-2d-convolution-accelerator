#include "conv_kernels.h"

void conv_linebuffer(
    uint8_t image[],
    int kernel[KERNEL_SIZE * KERNEL_SIZE],
    int output[],
    int height,
    int width,
    int norm_shift)
{
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem

#pragma HLS INTERFACE s_axilite port=kernel bundle=control
#pragma HLS INTERFACE s_axilite port=height bundle=control
#pragma HLS INTERFACE s_axilite port=width bundle=control
#pragma HLS INTERFACE s_axilite port=norm_shift bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS ARRAY_PARTITION variable=kernel complete

    const int K = KERNEL_SIZE;
    const int KM1 = K - 1;
    const int AREA = KERNEL_AREA;

    if (height < K || width < K)
        return;

    const int output_width = width - K + 1;

    // Stores the previous K-1 rows
    uint8_t linebuffer[KM1][MAX_WIDTH];
#pragma HLS ARRAY_PARTITION variable=linebuffer complete dim=1

    // Temporary register copy of the current column values from linebuffer
    uint8_t prev[KM1];
#pragma HLS ARRAY_PARTITION variable=prev complete

    for (int i = 0; i < height; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_HEIGHT

        for (int j = width - 1; j >= 0; j--) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_WIDTH
#pragma HLS PIPELINE II=1

            const uint8_t pixel = image[i * width + j];

            // Read the current column from the linebuffer
            for (int r = 0; r < KM1; r++) {
#pragma HLS LOOP_TRIPCOUNT min=(KERNEL_SIZE - 1) max=(KERNEL_SIZE - 1)
#pragma HLS UNROLL
                prev[r] = linebuffer[r][j];
            }

            // Compute convolution always
            
            int sum = 0;
            const int end = AREA - K;

            // Top K-1 rows come from linebuffer
            for (int k = 0; k < end; k++) {
#pragma HLS LOOP_TRIPCOUNT min=(KERNEL_AREA - KERNEL_SIZE) max=(KERNEL_AREA - KERNEL_SIZE)
#pragma HLS UNROLL
                const int ki = k / K;
                const int kj = k % K;
                sum += linebuffer[ki][j - KM1 + kj] * kernel[k];
            }

            // Bottom row comes from current image row
            for (int k = end; k < AREA; k++) {
#pragma HLS LOOP_TRIPCOUNT min=KERNEL_SIZE max=KERNEL_SIZE
#pragma HLS UNROLL
                const int kj = k % K;
                sum += image[i * width + (j - KM1 + kj)] * kernel[k];
            }

            const int bias = (norm_shift > 0) ? (1 << (norm_shift - 1)) : 0;

            bool valid = (i >= KM1 && j >= KM1);
            if (valid) {
                output[(i - KM1) * output_width + (j - KM1)] = (sum + bias) >> norm_shift;
            }

            // Shift linebuffer downward and insert current pixel
            for (int r = 0; r < KM1 - 1; r++) {
#pragma HLS LOOP_TRIPCOUNT min=(KERNEL_SIZE - 2) max=(KERNEL_SIZE - 2)
#pragma HLS UNROLL
                linebuffer[r][j] = prev[r + 1];
            }

            linebuffer[KM1 - 1][j] = pixel;
        }
    }
}