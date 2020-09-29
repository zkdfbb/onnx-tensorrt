/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "ResizeBilinear.hpp"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

#include <cuda_fp16.h>
#include <cassert>

// TODO: Move this to a common header
inline bool is_CHW(nvinfer1::Dims const& dims) {
    return (dims.nbDims == 3 &&
            dims.type[0] == nvinfer1::DimensionType::kCHANNEL &&
            dims.type[1] == nvinfer1::DimensionType::kSPATIAL &&
            dims.type[2] == nvinfer1::DimensionType::kSPATIAL);
}

nvinfer1::Dims ResizeBilinearPlugin::getOutputDimensions(int index,
        const nvinfer1::Dims *inputDims,
        int nbInputs) {
    assert(nbInputs == 1);
    nvinfer1::Dims const& input = inputDims[0];
    assert(is_CHW(input));
    assert(_ndims == 2);
    assert(index == 0);
    nvinfer1::Dims output;
    output.nbDims = input.nbDims;
    int s = 0;
    for( int d=0; d<input.nbDims; ++d ) {
        output.type[d] = input.type[d];
        if( input.type[d] == nvinfer1::DimensionType::kSPATIAL ) {
            output.d[d] = int(input.d[d] * _scale[s++]);
        } else {
            output.d[d] = input.d[d];
        }
    }
    return output;
}

int ResizeBilinearPlugin::initialize() {
    _output_dims = this->getOutputDimensions(0, &this->getInputDims(0), 1);
    assert(is_CHW(this->getInputDims(0)));
    assert(is_CHW(_output_dims));
    assert(_ndims == 2);
    return 0;
}

template <typename Data>
__host__ __forceinline__ static Data area_pixel_compute_scale(
        int input_size,
        int output_size,
        bool align_corners) {
    if (output_size > 1) {
        return align_corners ? (Data)(input_size - 1) / (output_size - 1)
            : (Data)input_size / output_size;
    } else {
        return static_cast<Data>(0);
    }
}

template <typename Data>
__device__ __forceinline__ static Data area_pixel_compute_source_index(
        Data scale,
        int dst_index,
        bool align_corners,
        bool cubic) {
    if (align_corners) {
        return scale * dst_index;
    } else {
        Data src_idx = scale * (dst_index + static_cast<Data>(0.5)) -
            static_cast<Data>(0.5);
        // See Note[Follow Opencv resize logic]
        return (!cubic && src_idx < static_cast<Data>(0))
            ? static_cast<Data>(0)
            : src_idx;
    }
}

template <typename Data>
__global__ void upsample_bilinear2d_out_frame(const int n, const Data rheight,
        const Data rwidth, const bool align_corners,
        const int batchsize, const int channels,
        const int height1, const int width1, const int step1,
        const int height2, const int width2, const int step2,
        const Data* idata, Data* odata) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
        const int w2 = index % width2; // 0:width2-1
        const int h2 = index / width2; // 0:height2-1
        // special case: just copy
        if (height1 == height2 && width1 == width2) {
            int idx = index;
            for (int b = 0; b < batchsize; b++) {
                for (int c = 0; c < channels; ++c) {
                    odata[idx] = idata[idx];
                    idx += step1;
                }
            }
            return;
        }

        const Data h1r = area_pixel_compute_source_index<Data>(
                rheight, h2, align_corners, /*cubic=*/false);
        const int h1 = h1r;
        const int h1p = (h1 < height1 - 1) ? 1 : 0;
        const Data h1lambda = h1r - h1;
        const Data h0lambda = static_cast<Data>(1) - h1lambda;

        const Data w1r = area_pixel_compute_source_index<Data>(
                rwidth, w2, align_corners, /*cubic=*/false);
        const int w1 = w1r;
        const int w1p = (w1 < width1 - 1) ? 1 : 0;
        const Data w1lambda = w1r - w1;
        const Data w0lambda = static_cast<Data>(1) - w1lambda;

        int idx2 = h2 * width2 + w2;
        int idx1 = h1 * width1 + w1;
        for (int b = 0; b < batchsize; b++) {
            for (int c = 0; c < channels; ++c) {
                int idx00 = idx1;
                int idx01 = idx1 + w1p;
                int idx10 = idx1 + h1p * width1;
                int idx11 = idx10 + w1p;
                const Data val = h0lambda * (w0lambda * idata[idx00] + w1lambda * idata[idx01]) +
                    h1lambda * (w0lambda * idata[idx10] + w1lambda * idata[idx11]);
                odata[idx2] = val;
                idx1 += step1;
                idx2 += step2;
            }
        }
    }
}


int ResizeBilinearPlugin::enqueue(int batchSize,
        const void *const *inputs, void **outputs,
        void *workspace, cudaStream_t stream) {
    auto const& input_dims = this->getInputDims(0);
    int channels = input_dims.d[0];
    // cout << "ResizeBilinearPlugin batchSize: " << batchSize;
    // cout << " input_dims: " << input_dims.d[0] << " " << input_dims.d[1] << " " << input_dims.d[2];
    // cout << " output_dims: " << _output_dims.d[0] << " " << _output_dims.d[1] << " " << _output_dims.d[2] << endl;
    switch( _ndims ) {
        case 2: {
            const int input_height = input_dims.d[1];
            const int input_width = input_dims.d[2];
            const int output_height = _output_dims.d[1];
            const int output_width = _output_dims.d[2];
            const int input_step = input_height * input_width;
            const int output_step = output_height * output_width;
            const bool align_corners = false;
            const int num_kernels = output_height * output_width;
            const int num_threads = 512;
            const int num_blocks = (num_kernels + num_threads - 1) / num_threads;

            if (getDataType()==nvinfer1::DataType::kFLOAT) {
                const float rheight = area_pixel_compute_scale<float>(input_height, output_height, align_corners);
                const float rwidth = area_pixel_compute_scale<float>(input_width, output_width, align_corners);
                upsample_bilinear2d_out_frame<<<num_blocks, num_threads>>>(num_kernels, rheight, rwidth, align_corners,
                    batchSize, channels, input_height, input_width, input_step,
                    output_height, output_width, output_step,
                    static_cast<float const*>(inputs[0]),
                    static_cast<float*>(outputs[0]));
                return cudaGetLastError() != cudaSuccess;
            } else {
                return -1;
            }
        }
        default: return -1;
    }
}
