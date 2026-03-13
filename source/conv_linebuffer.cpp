#include "conv_kernels.h"

void conv_linebuffer(
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

    // store previous two rows
    uint8_t linebuffer[2][MAX_WIDTH];

#pragma HLS ARRAY_PARTITION variable=linebuffer complete dim=1

    for (int i = 0; i < height; i++) {

#pragma HLS LOOP_TRIPCOUNT min=1 max=512

        for (int j = width - 1; j >= 0; j--) {

#pragma HLS LOOP_TRIPCOUNT min=1 max=512
#pragma HLS PIPELINE II=1

            uint8_t pixel = image[i * width + j];

            // read line buffer
            uint8_t prev2 = linebuffer[0][j];
            uint8_t prev1 = linebuffer[1][j];

            // compute convolution only when kernel fits
            if (i >= 2 && j >= 2) {

                int sum = 0;

                // row i-2
                sum += linebuffer[0][j-2] * kernel[0];
                sum += linebuffer[0][j-1] * kernel[1];
                sum += prev2 * kernel[2];

                // row i-1
                sum += linebuffer[1][j-2] * kernel[3];
                sum += linebuffer[1][j-1] * kernel[4];
                sum += prev1 * kernel[5];

                // row i
                sum += image[i * width + (j-2)] * kernel[6];
                sum += image[i * width + (j-1)] * kernel[7];
                sum += pixel * kernel[8];

                output[(i-2) * output_width + (j-2)] = sum;
            }

            // shift line buffer
            linebuffer[0][j] = prev1;
            linebuffer[1][j] = pixel;
        }
    }
}