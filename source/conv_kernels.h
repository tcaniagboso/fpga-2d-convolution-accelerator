#ifndef CONV_KERNELS_H
#define CONV_KERNELS_H

#define MAX_HEIGHT 512
#define MAX_WIDTH 512
#define KERNEL_SIZE 3
#define KERNEL_AREA (KERNEL_SIZE * KERNEL_SIZE)

#include <ap_axi_sdata.h>
#include <cstdint>
#include <hls_stream.h>

typedef ap_axiu<32, 0, 0, 0> axis_pixel_t;   // holds 8-bit 4 pixels
typedef ap_axiu<32, 0, 0, 0> axis_int_t;   // int representations

void conv_baseline(
    uint8_t image[],
    int kernel[KERNEL_SIZE * KERNEL_SIZE],
    int output[],
    int height,
    int width,
    int norm_shift);

void conv_pipeline(
    uint8_t image[],
    int kernel[KERNEL_SIZE * KERNEL_SIZE],
    int output[],
    int height,
    int width,
    int norm_shift);

void conv_linebuffer(
    uint8_t image[],
    int kernel[KERNEL_SIZE * KERNEL_SIZE],
    int output[],
    int height,
    int width,
    int norm_shift);

void conv_dataflow(
    uint8_t image[],
    int kernel[KERNEL_SIZE * KERNEL_SIZE],
    int output[],
    int height,
    int width,
    int norm_shift);

void conv_dataflow_stream(
    hls::stream<axis_pixel_t>& in_stream,
    hls::stream<axis_pixel_t>& out_stream,
    int kernel[KERNEL_SIZE * KERNEL_SIZE],
    int height,
    int width,
    int norm_shift);

void conv_dataflow_stream_int(
    hls::stream<axis_pixel_t>& in_stream,
    hls::stream<axis_int_t>& out_stream,
    int kernel[KERNEL_SIZE * KERNEL_SIZE],
    int height,
    int width,
    int norm_shift);

#endif