#include "conv_kernels.h"

void conv_dataflow(
    uint8_t image[],
    int kernel[KERNEL_AREA],
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
    const int AREA = KERNEL_AREA;
    const int KM1 = K - 1;

    if (height < K || width < K)
        return;

    uint8_t prev[KM1];
    uint8_t linebuffer[KM1][MAX_WIDTH];
    uint8_t window[K][K];

#pragma HLS ARRAY_PARTITION variable=prev complete
#pragma HLS ARRAY_PARTITION variable=window complete dim=0
#pragma HLS ARRAY_PARTITION variable=linebuffer complete dim=1

    const int output_width = width - K + 1;

    for (int i = 0; i < height; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_HEIGHT

        for (int j = 0; j < width; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_WIDTH
#pragma HLS PIPELINE II=1

            const uint8_t pixel = image[i * width + j];

            // Read current column from linebuffer
            for (int r = 0; r < KM1; r++) {
#pragma HLS LOOP_TRIPCOUNT min=(KERNEL_SIZE - 1) max=(KERNEL_SIZE - 1)
#pragma HLS UNROLL
                prev[r] = linebuffer[r][j];
            }

            // Shift linebuffer downward and insert current pixel
            for (int r = 0; r < KM1 - 1; r++) {
#pragma HLS LOOP_TRIPCOUNT min=(KERNEL_SIZE - 2) max=(KERNEL_SIZE - 2)
#pragma HLS UNROLL
                linebuffer[r][j] = prev[r + 1];
            }
            linebuffer[KM1 - 1][j] = pixel;

            // Shift window left
            for (int r = 0; r < K; r++) {
#pragma HLS LOOP_TRIPCOUNT min=KERNEL_SIZE max=KERNEL_SIZE
#pragma HLS UNROLL
                for (int c = 0; c < KM1; c++) {
#pragma HLS LOOP_TRIPCOUNT min=(KERNEL_SIZE - 1) max=(KERNEL_SIZE - 1)
#pragma HLS UNROLL
                    window[r][c] = window[r][c + 1];
                }
            }

            // Insert new rightmost column
            for (int r = 0; r < KM1; r++) {
#pragma HLS LOOP_TRIPCOUNT min=(KERNEL_SIZE - 1) max=(KERNEL_SIZE - 1)
#pragma HLS UNROLL
                window[r][KM1] = prev[r];
            }
            window[KM1][KM1] = pixel;

            // Compute always
            int sum = 0;
            for (int k = 0; k < AREA; k++) {
#pragma HLS LOOP_TRIPCOUNT min=KERNEL_AREA max=KERNEL_AREA
#pragma HLS UNROLL
                const int ki = k / K;
                const int kj = k % K;
                sum += window[ki][kj] * kernel[k];
            }

            const int bias = (norm_shift > 0) ? (1 << (norm_shift - 1)) : 0;

            // Write if valid
            bool valid = (i >= KM1 && j >= KM1);
            if (valid) {
                output[(i - KM1) * output_width + (j - KM1)] = (sum + bias) >> norm_shift;
            }
        }
    }
}