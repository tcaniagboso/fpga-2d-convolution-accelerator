#include "conv_kernels.h"

void conv_dataflow(
    uint8_t image[],
    int kernel[9],
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

    uint8_t linebuffer[2][MAX_WIDTH];
    uint8_t window[3][3];

#pragma HLS ARRAY_PARTITION variable=window complete
#pragma HLS ARRAY_PARTITION variable=linebuffer complete dim=1

    int output_width = width - 2;

    for(int i = 0; i < height; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512

        for(int j = 0; j < width; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
#pragma HLS PIPELINE II=1

            uint8_t pixel = image[i * width + j];

            uint8_t prev2 = linebuffer[0][j];
            uint8_t prev1 = linebuffer[1][j];

            linebuffer[0][j] = prev1;
            linebuffer[1][j] = pixel;

            // shift window
            for (int r = 0; r < 3; r++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3                
                window[r][0] = window[r][1];
                window[r][1] = window[r][2];
            }

            // insert new column
            window[0][2] = prev2;
            window[1][2] = prev1;
            window[2][2] = pixel;

            if (i >= 2 && j >= 2) {

                int sum = 0;

                sum += window[0][0] * kernel[0];
                sum += window[0][1] * kernel[1];
                sum += window[0][2] * kernel[2];

                sum += window[1][0] * kernel[3];
                sum += window[1][1] * kernel[4];
                sum += window[1][2] * kernel[5];

                sum += window[2][0] * kernel[6];
                sum += window[2][1] * kernel[7];
                sum += window[2][2] * kernel[8];

                output[(i-2) * output_width + (j-2)] = sum;
            }
        }
    }
}