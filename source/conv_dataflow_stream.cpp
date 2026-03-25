#include "conv_kernels.h"

void conv_dataflow_stream(
    hls::stream<axis_pixel_t>& in_stream,
    hls::stream<axis_pixel_t>& out_stream,
    int kernel[KERNEL_SIZE * KERNEL_SIZE],
    int height,
    int width,
    int norm_shift)
{

#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream

#pragma HLS INTERFACE s_axilite port=kernel bundle=control
#pragma HLS INTERFACE s_axilite port=height bundle=control
#pragma HLS INTERFACE s_axilite port=width bundle=control
#pragma HLS INTERFACE s_axilite port=norm_shift bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS ARRAY_PARTITION variable=kernel complete

    const int K = KERNEL_SIZE;
    const int KM1 = K - 1;
    const int AREA = KERNEL_AREA;
    uint32_t in_word = 0;
    int in_count = 0;
    uint32_t out_word = 0;
    int out_count = 0;
    int total_pixels = (height - K + 1) * (width - K + 1);
    int pixel_counter = 0;

    if (height < K || width < K)
        return;

    uint8_t linebuffer[KM1][MAX_WIDTH];
#pragma HLS ARRAY_PARTITION variable=linebuffer complete dim=1

    uint8_t window[K][K];
#pragma HLS ARRAY_PARTITION variable=window complete dim=0

    uint8_t prev[KM1];
#pragma HLS ARRAY_PARTITION variable=prev complete

    for (int i = 0; i < height; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_HEIGHT

        for (int j = 0; j < width; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_WIDTH
#pragma HLS PIPELINE II=1

            // =========================
            // 1. Read 4 uint8_t pixels
            //    from 32 bits wide stream
            // =========================

            if (in_count == 0) {
                axis_pixel_t in_pkt = in_stream.read();
                in_word = in_pkt.data;
            }

            uint8_t pixel = (in_word >> (8 * in_count)) & 0xFF;

            in_count++;
            if (in_count == 4) in_count = 0;

            // =========================
            // 2. Read linebuffer -> prev
            // =========================
            for (int r = 0; r < KM1; r++) {
#pragma HLS LOOP_TRIPCOUNT min=(KERNEL_SIZE - 1) max=(KERNEL_SIZE - 1)
#pragma HLS UNROLL
                prev[r] = linebuffer[r][j];
            }

            // =========================
            // 3. Update linebuffer
            // =========================
            for (int r = 0; r < KM1 - 1; r++) {
#pragma HLS LOOP_TRIPCOUNT min=(KERNEL_SIZE - 2) max=(KERNEL_SIZE - 2)               
#pragma HLS UNROLL
                linebuffer[r][j] = prev[r + 1];
            }
            linebuffer[KM1 - 1][j] = pixel;

            // =========================
            // 4. Shift window LEFT
            // =========================
            for (int r = 0; r < K; r++) {
#pragma HLS UNROLL
#pragma HLS LOOP_TRIPCOUNT min=KERNEL_SIZE max=KERNEL_SIZE
                for (int c = 0; c < KM1; c++) {
#pragma HLS LOOP_TRIPCOUNT min=(KERNEL_SIZE - 1) max=(KERNEL_SIZE - 1)
#pragma HLS UNROLL
                    window[r][c] = window[r][c + 1];
                }
            }

            // =========================
            // 5. Insert new column
            // =========================
            for (int r = 0; r < KM1; r++) {
#pragma HLS LOOP_TRIPCOUNT min=(KERNEL_SIZE - 1) max=(KERNEL_SIZE - 1)
#pragma HLS UNROLL
                window[r][KM1] = prev[r];
            }
            window[KM1][KM1] = pixel;

            // =========================
            // 6. Compute
            // =========================
            
            int sum = 0;

            for (int k = 0; k < AREA; k++) {
#pragma HLS UNROLL
                const int ki = k / K;
                const int kj = k % K;
                sum += window[ki][kj] * kernel[k];
            }

            const int bias = (norm_shift > 0) ? (1 << (norm_shift - 1)) : 0;
            const int result = (sum + bias) >> norm_shift;

            // =========================
            // 7. Write output
            // =========================
            bool valid = (i >= KM1 && j >= KM1);
            if (valid) {
                int clamped = (result < 0)? 0 : result;
                clamped = (clamped > 255)? 255 : clamped;

                out_word |= (clamped & 0xFF) << (8 * out_count);
                out_count++;
                pixel_counter++;

                if (out_count == 4) {
                    axis_pixel_t out_pkt;
                    out_pkt.data = out_word;
                    out_pkt.keep = -1;

                    bool is_last_pixel = (pixel_counter == total_pixels);
                    out_pkt.last = is_last_pixel;

                    out_stream.write(out_pkt);

                    out_word = 0;
                    out_count = 0;
                }
            }
        }
    }

    if (out_count != 0) {
        axis_pixel_t out_pkt;
        out_pkt.data = out_word;
        out_pkt.keep = -1;
        out_pkt.last = 1;
        out_stream.write(out_pkt);
    }
    
}

void conv_dataflow_stream_int(
    hls::stream<axis_pixel_t>& in_stream,
    hls::stream<axis_int_t>& out_stream,
    int kernel[KERNEL_SIZE * KERNEL_SIZE],
    int height,
    int width,
    int norm_shift)
{

#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream

#pragma HLS INTERFACE s_axilite port=kernel bundle=control
#pragma HLS INTERFACE s_axilite port=height bundle=control
#pragma HLS INTERFACE s_axilite port=width bundle=control
#pragma HLS INTERFACE s_axilite port=norm_shift bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS ARRAY_PARTITION variable=kernel complete

    const int K = KERNEL_SIZE;
    const int KM1 = K - 1;
    const int AREA = KERNEL_AREA;
    uint32_t in_word = 0;
    int in_count = 0;

    if (height < K || width < K)
        return;

    uint8_t linebuffer[KM1][MAX_WIDTH];
#pragma HLS ARRAY_PARTITION variable=linebuffer complete dim=1

    uint8_t window[K][K];
#pragma HLS ARRAY_PARTITION variable=window complete dim=0

    uint8_t prev[KM1];
#pragma HLS ARRAY_PARTITION variable=prev complete

    for (int i = 0; i < height; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_HEIGHT

        for (int j = 0; j < width; j++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_WIDTH
#pragma HLS PIPELINE II=1

            // =========================
            // 1. Read 4 uint8_t pixels
            //    from 32 bits wide stream
            // =========================

            if (in_count == 0) {
                axis_pixel_t in_pkt = in_stream.read();
                in_word = in_pkt.data;
            }

            uint8_t pixel = (in_word >> (8 * in_count)) & 0xFF;

            in_count++;
            if (in_count == 4) in_count = 0;


            // =========================
            // 2. Read linebuffer -> prev
            // =========================
            for (int r = 0; r < KM1; r++) {
#pragma HLS LOOP_TRIPCOUNT min=(KERNEL_SIZE - 1) max=(KERNEL_SIZE - 1)
#pragma HLS UNROLL
                prev[r] = linebuffer[r][j];
            }

            // =========================
            // 3. Update linebuffer
            // =========================
            for (int r = 0; r < KM1 - 1; r++) {
#pragma HLS LOOP_TRIPCOUNT min=(KERNEL_SIZE - 2) max=(KERNEL_SIZE - 2)               
#pragma HLS UNROLL
                linebuffer[r][j] = prev[r + 1];
            }
            linebuffer[KM1 - 1][j] = pixel;

            // =========================
            // 4. Shift window LEFT
            // =========================
            for (int r = 0; r < K; r++) {
#pragma HLS UNROLL
#pragma HLS LOOP_TRIPCOUNT min=KERNEL_SIZE max=KERNEL_SIZE
                for (int c = 0; c < KM1; c++) {
#pragma HLS LOOP_TRIPCOUNT min=(KERNEL_SIZE - 1) max=(KERNEL_SIZE - 1)
#pragma HLS UNROLL
                    window[r][c] = window[r][c + 1];
                }
            }

            // =========================
            // 5. Insert new column
            // =========================
            for (int r = 0; r < KM1; r++) {
#pragma HLS LOOP_TRIPCOUNT min=(KERNEL_SIZE - 1) max=(KERNEL_SIZE - 1)
#pragma HLS UNROLL
                window[r][KM1] = prev[r];
            }
            window[KM1][KM1] = pixel;

            // =========================
            // 6. Compute
            // =========================
            
            int sum = 0;

            for (int k = 0; k < AREA; k++) {
#pragma HLS UNROLL
                const int ki = k / K;
                const int kj = k % K;
                sum += window[ki][kj] * kernel[k];
            }

            const int bias = (norm_shift > 0) ? (1 << (norm_shift - 1)) : 0;
            const int result = (sum + bias) >> norm_shift;

            // =========================
            // 7. Write output
            // =========================
            bool valid = (i >= KM1 && j >= KM1);
            if (valid) {
                axis_int_t out_pkt;
                out_pkt.data = result;
                out_pkt.keep = -1;

                const int out_i = i - KM1;
                const int out_j = j - KM1;

                const int out_h = height - K + 1;
                const int out_w = width  - K + 1;

                out_pkt.last = (out_i == out_h - 1 && out_j == out_w - 1);

                out_stream.write(out_pkt);
            }
        }
    }
}