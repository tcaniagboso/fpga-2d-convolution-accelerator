#ifndef CONV_KERNELS_H
#define CONV_KERNELS_H

#define MAX_HEIGHT 512
#define MAX_WIDTH 512
#define KERNEL_SIZE 3

#include <cstdint>

void conv_baseline(
    uint8_t image[],
    int kernel[KERNEL_SIZE * KERNEL_SIZE],
    int output[],
    int height,
    int width);

void conv_pipeline(
    uint8_t image[],
    int kernel[KERNEL_SIZE * KERNEL_SIZE],
    int output[],
    int height,
    int width);

void conv_linebuffer(
    uint8_t image[],
    int kernel[KERNEL_SIZE * KERNEL_SIZE],
    int output[],
    int height,
    int width);

void conv_dataflow(
    uint8_t image[],
    int kernel[KERNEL_SIZE * KERNEL_SIZE],
    int output[],
    int height,
    int width);


#endif