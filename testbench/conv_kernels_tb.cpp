#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "conv_kernels.h"

#define HEIGHT 512
#define WIDTH  512

typedef void (*conv_fn)(
    uint8_t[],
    int[],
    int[],
    int,
    int,
    int
);

struct Design {
    const char* design_name_;
    conv_fn fn_;

    Design(const char* name, conv_fn f)
        : design_name_{name}, fn_{f} {}
};

struct Kernel {
    const char* kernel_name_;
    int* kernel_;
    int norm_shift_;

    Kernel(const char* kernel_name, int* kernel, int norm_shift)
        : kernel_name_{kernel_name}, kernel_{kernel}, norm_shift_{norm_shift} {}
};

void initialize_image(uint8_t image[]) {
    for (int i = 0; i < HEIGHT * WIDTH; i++) {
        image[i] = rand() % 256;
    }
}

void conv_golden(
    uint8_t image[],
    int kernel[],
    int output[],
    int height,
    int width,
    int norm_shift)
{
    int output_height = height - KERNEL_SIZE + 1;
    int output_width  = width - KERNEL_SIZE + 1;

    for(int i = 0; i < output_height; i++)
    {
        for(int j = 0; j < output_width; j++)
        {
            int sum = 0;

            for(int ki = 0; ki < KERNEL_SIZE; ki++)
            {
                for(int kj = 0; kj < KERNEL_SIZE; kj++)
                {
                    sum += image[(i + ki)*width + (j + kj)]
                           * kernel[ki*KERNEL_SIZE + kj];
                }
            }

            int bias = (norm_shift > 0) ? (1 << (norm_shift - 1)) : 0; 
            output[i * output_width + j] = (sum + bias) >> norm_shift;
        }
    }
}

void convert_to_uint8(int input[], uint8_t output[], int size) {
    for (int i = 0; i < size; i++) {
        int v = input[i];
        v = (v < 0) ? 0 : v;
        v = (v > 255) ? 255 : v;
        output[i] = (uint8_t)v;
    }
}

template<typename T>
bool compare_to_golden(const T output_golden[], const T output_design[], const char* design_name, int size) {
    for (int i = 0; i < size; i++) {
        if (output_golden[i] != output_design[i]) {
            std::printf("%s mismatch at index %d\n", design_name, i);
            return false;
        }
    }

    return true;
}

int compute_norm_shift(int* kernel, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) sum += kernel[i];

    int shift = 0;
    while ((1 << shift) < sum) shift++;
    return shift;
}

void run_common_test_setup(
    uint8_t image[],
    int output_golden_int[],
    uint8_t output_golden_uint8[],
    const Kernel& kernel,
    int size
) {
    initialize_image(image);

    memset(output_golden_int, 0, size * sizeof(int));
    memset(output_golden_uint8, 0, size * sizeof(uint8_t));

    conv_golden(image, kernel.kernel_, output_golden_int, HEIGHT, WIDTH, kernel.norm_shift_);
    convert_to_uint8(output_golden_int, output_golden_uint8, size);
}

void run_memory_tests(
    uint8_t image[],
    const int output_golden[],
    const Design designs[],
    int num_designs,
    int size,
    const Kernel& kernel
) {
    int output_design[size];

    for (int i = 0; i < num_designs; i++) {
        memset(output_design, 0, sizeof(output_design));

        designs[i].fn_(image, kernel.kernel_, output_design, HEIGHT, WIDTH, kernel.norm_shift_);

        bool passed = compare_to_golden<int>(
            output_golden,
            output_design,
            designs[i].design_name_,
            size
        );

        std::printf("%s design %s\n",
            designs[i].design_name_,
            passed ? "PASSED" : "FAILED");
    }
}

template<typename T, typename Out,
         void (*StreamFn)(
             hls::stream<axis_pixel_t>&,
             hls::stream<Out>&,
             int[],
             int,
             int,
             int)>
void run_stream_test(
    uint8_t image[],
    const T output_golden[],
    const Kernel& kernel,
    int size,
    const char* design_name
) {
    hls::stream<axis_pixel_t> in_stream;
    hls::stream<Out> out_stream;

    // write input
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            axis_pixel_t pkt;
            pkt.data = image[i * WIDTH + j];
            pkt.keep = -1;
            pkt.last = (i == HEIGHT - 1 && j == WIDTH - 1);
            in_stream.write(pkt);
        }
    }

    // run kernel
    StreamFn(in_stream, out_stream, kernel.kernel_, HEIGHT, WIDTH, kernel.norm_shift_);

    // read output
    T output_stream[size];

    for (int i = 0; i < size; i++) {
        Out pkt = out_stream.read();
        output_stream[i] = pkt.data;
    }

    bool passed = compare_to_golden<T>(output_golden, output_stream, design_name, size);

    std::printf("%s design %s\n", design_name, passed ? "PASSED" : "FAILED");
}

int main() {
    // Kernel
    int sobel_x[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };

    int sobel_y[9] = {
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1
    };

    int gaussian[9] = {
        1, 2, 1,
        2, 4, 2,
        1, 2, 1        
    };

    Kernel kernels[] = {
        {"Sobel X", sobel_x, compute_norm_shift(sobel_x, KERNEL_AREA)},
        {"Sobel Y", sobel_y, compute_norm_shift(sobel_y, KERNEL_AREA)},
        {"Gaussian Blur", gaussian, compute_norm_shift(gaussian, KERNEL_AREA)},
    };

    int num_kernels = sizeof(kernels) / sizeof(Kernel);

    // image
    uint8_t image[HEIGHT * WIDTH];
    
    // Output dimentsions
    int output_height = HEIGHT - KERNEL_SIZE + 1;
    int output_width = WIDTH - KERNEL_SIZE + 1;
    int size = output_height * output_width;
    
    // Golden output
    int output_golden_int[size];
    uint8_t output_golden_uint8[size];
    
    // Create and test designs;
    Design designs[] = {
        {"Baseline", conv_baseline},
        {"Pipeline", conv_pipeline},
        {"Linebuffer", conv_linebuffer},
        {"Dataflow", conv_dataflow}
    };

    int num_designs = sizeof(designs) / sizeof(Design);
    const char* stream_int = "Stream (int)";
    const char* stream_uint8 = "Stream (uint8_t)";    

    srand(0);
    for (const auto& kernel : kernels) {
        std::printf("%s Kernel Test:\n", kernel.kernel_name_);

        for (int t = 1; t <= 10; t++) {
            std::printf("TEST #%d:\n", t);

            run_common_test_setup(image, output_golden_int, output_golden_uint8, kernel, size);

            run_memory_tests(image, output_golden_int, designs, num_designs, size, kernel);

            run_stream_test<int, axis_int_t, conv_dataflow_stream_int>(image, output_golden_int, kernel, size, stream_int);

            run_stream_test<uint8_t, axis_pixel_t, conv_dataflow_stream>(image, output_golden_uint8, kernel, size, stream_uint8);

            std::printf("\n");
        }

        std::printf("\n");
    }

    return 0;
}

// BIGGER KERNELS
// // 7 X 7
// int sobel_x[49] = {
//     -1, 0, 1, -1, 0, 1, -1,
//     -2, 0, 2, -2, 0, 2, -2,
//     -1, 0, 1, -1, 0, 1, -1,
//     -2, 0, 2, -2, 0, 2, -2,
//     -1, 0, 1, -1, 0, 1, -1,
//     -2, 0, 2, -2, 0, 2, -2,
//     -1, 0, 1, -1, 0, 1, -1
// };

// int sobel_y[49] = {
//     -1, -2, -1, -2, -1, -2, -1,
//     0, 0, 0, 0, 0, 0, 0,
//     1, 2, 1, 2, 1, 2, 1,
//     0, 0, 0, 0, 0, 0, 0,
//     1, 2, 1, 2, 1, 2, 1,
//     0, 0, 0, 0, 0, 0, 0,
//     1, 2, 1, 2, 1, 2, 1
// };

// int gaussian[49] = {
//     1, 2, 1, 2, 1, 2, 1,
//     2, 4, 2, 4, 2, 4, 2,
//     1, 2, 1, 2, 1, 2, 1,
//     2, 4, 2, 4, 2, 4, 2,
//     1, 2, 1, 2, 1, 2, 1,
//     2, 4, 2, 4, 2, 4, 2,
//     1, 2, 1, 2, 1, 2, 1
// };

// 5 X 5
// int sobel_x[25] = {
//     -1, 0, 1, -1, 0,
//     -2, 0, 2, -2, 0,
//     -1, 0, 1, -1, 0,
//     -2, 0, 2, -2, 0,
//     -1, 0, 1, -1, 0
// };

// int sobel_y[25] = {
//     -1, -2, -1, -2, -1,
//     0, 0, 0, 0, 0,
//     1, 2, 1, 2, 1,
//     0, 0, 0, 0, 0,
//     1, 2, 1, 2, 1
// };

// int gaussian[25] = {
//     1, 2, 1, 2, 1,
//     2, 4, 2, 4, 2,
//     1, 2, 1, 2, 1,
//     2, 4, 2, 4, 2,
//     1, 2, 1, 2, 1
// };





