#include "conv_kernels.h"

void conv_pipeline(
    uint8_t image[],
    int kernel[KERNEL_SIZE * KERNEL_SIZE],
    int output[],
    int height,
    int width)
{

#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem

#pragma HLS INTERFACE s_axilite port=kernel bundle=control
#pragma HLS INTERFACE s_axilite port=height bundle=control
#pragma HLS INTERFACE s_axilite port=width bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS ARRAY_PARTITION variable=kernel complete

    if (height < KERNEL_SIZE || width < KERNEL_SIZE)
        return;

    int output_height = height - KERNEL_SIZE + 1;
    int output_width  = width - KERNEL_SIZE + 1;

    for (int i = 0; i < output_height; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512

        for (int j = 0; j < output_width; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
#pragma HLS PIPELINE II=1

            int sum = 0;

            for (int k = 0; k < KERNEL_SIZE * KERNEL_SIZE; k++) {
#pragma HLS LOOP_TRIPCOUNT min=9 max=9
#pragma HLS UNROLL

                int ki = k / KERNEL_SIZE;
                int kj = k % KERNEL_SIZE;

                sum += image[(i + ki) * width + (j + kj)] * kernel[k];
            }

            output[i * output_width + j] = sum;
        }
    }
}