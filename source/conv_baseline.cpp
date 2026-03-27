#include "conv_kernels.h"

void conv_baseline(
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

    if (height < KERNEL_SIZE || width < KERNEL_SIZE)
        return;

    const int output_height = height - KERNEL_SIZE + 1;
    const int output_width  = width - KERNEL_SIZE + 1;

    for (int i = 0; i < output_height; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=(MAX_HEIGHT - KERNEL_SIZE + 1)

        for (int j = 0; j < output_width; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=(MAX_WIDTH - KERNEL_SIZE + 1)

            int sum = 0;

            for (int k = 0; k < KERNEL_AREA; k++) {
#pragma HLS LOOP_TRIPCOUNT min=KERNEL_AREA max=KERNEL_AREA

                const int ki = k / KERNEL_SIZE;
                const int kj = k % KERNEL_SIZE;

                sum += image[(i + ki) * width + (j + kj)] * kernel[k];
            }

            const int bias = (norm_shift > 0) ? (1 << (norm_shift - 1)) : 0; 
            output[i * output_width + j] = (sum + bias) >> norm_shift;
        }
    }
}