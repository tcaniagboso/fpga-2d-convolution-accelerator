#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include "conv_kernels.h"

#define HEIGHT 512
#define WIDTH  512

typedef void (*conv_fn)(
    uint8_t[],
    int[],
    int[],
    int,
    int
);

struct Design {
    const char* design_name;
    conv_fn fn;

    Design(const char* name, conv_fn f)
        : design_name(name), fn(f) {}
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
    int width)
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

            output[i*output_width + j] = sum;
        }
    }
}

bool compare_to_golden(int output_golden[], int output_design[], const char* design_name, int size) {
    for (int i = 0; i < size; i++) {
        if (output_golden[i] != output_design[i]) {
            std::printf("%s mismatch at index %d\n", design_name, i);
            return false;
        }
    }

    return true;
}

int main() {
    // Kernel
    int kernel[9] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };

    // image
    uint8_t image[HEIGHT * WIDTH];
    
    // Output dimentsions
    int output_height = HEIGHT - KERNEL_SIZE + 1;
    int output_width = WIDTH - KERNEL_SIZE + 1;
    int size = output_height * output_width;
    
    // Golden output
    int output_golden[size];
    
    // Create and test designs;
    Design designs[] = {
        {"Baseline", conv_baseline},
        {"Pipeline", conv_pipeline},
        {"Linebuffer", conv_linebuffer},
        {"Dataflow", conv_dataflow}
    };

    int num_designs = sizeof(designs) / sizeof(Design);
    int output_design[size];

    srand(0);
    for (int t = 1; t <= 10; t++) {
        // Clear and generate image
        memset(image, 0, sizeof(image));
        initialize_image(image);

        // Clear and comput golden reference
        memset(output_golden, 0, sizeof(output_golden));
        conv_golden(image, kernel, output_golden, HEIGHT, WIDTH);

        // Test designs
        std::printf("TEST #%d:\n", t);
        for (int i = 0; i < num_designs; i++) {
            memset(output_design, 0, sizeof(output_design));
            Design& cur_design = designs[i];
            cur_design.fn(image, kernel, output_design, HEIGHT, WIDTH);
            bool passed = compare_to_golden(output_golden, output_design, cur_design.design_name, size);
            std::printf("%s design %s\n",
               cur_design.design_name,
               passed ? "PASSED" : "FAILED");
        }

        std::printf("\n"); 
    }

    return 0;
}




