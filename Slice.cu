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

#include "Slice.hpp"
#include <cuda_fp16.h>
#include <cassert>

nvinfer1::Dims SlicePlugin::getOutputDimensions(int index,
                                                const nvinfer1::Dims *inputDims,
                                                int nbInputs) {
  assert(nbInputs == 1);
  assert(index < this->getNbOutputs());
  assert(_axis == 0);
  nvinfer1::Dims const& input_dims = inputDims[0];
  nvinfer1::Dims output_dims = input_dims;
  output_dims.d[_axis] = _end - _start;
  return output_dims;
}

int SlicePlugin::initialize() {
  _output_dims = this->getOutputDimensions(0, &this->getInputDims(0), 1);
  return 0;
}

template <typename Data>
__global__ void slice_channels(const int n,
        const int batchsize, const int channels,
        const int height, const int width,
        const int start, const int end,
        const Data* idata, Data* odata) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
      for (int b = 0; b < batchsize; b++) {
        for (int c = 0; c < channels; ++c) {
            if( c >= start && c < end){
              odata[idx - start * n - b * (channels + start - end) * n] = idata[idx];
            }
            idx += n;
        }
      }
    }
  }

int SlicePlugin::enqueue(int batchSize,
                                 const void *const *inputs, void **outputs,
                                 void *workspace, cudaStream_t stream) {
  auto const& input_dims = this->getInputDims(0);
  const int channels = input_dims.d[0];
  // cout << "SlicePlugin batchSize: " << batchSize;
  // cout << " input_dims: " << input_dims.d[0] << " " << input_dims.d[1] << " " << input_dims.d[2];
  // cout << " output_dims: " << _output_dims.d[0] << " " << _output_dims.d[1] << " " << _output_dims.d[2] << endl;
  const int input_height = input_dims.d[1];
  const int input_width = input_dims.d[2];
  const int num_kernels = input_height * input_width;
  const int num_threads = 512;
  const int num_blocks = (num_kernels + num_threads - 1) / num_threads;
  if (getDataType() == nvinfer1::DataType::kFLOAT) {
    slice_channels<<<num_blocks, num_threads>>>(num_kernels, batchSize, channels, 
      input_height, input_width, _start, _end,
      static_cast<float const*>(inputs[0]),
      static_cast<float*>(outputs[0]));
  } else {
    slice_channels<<<num_blocks, num_threads>>>(num_kernels, batchSize, channels, 
      input_height, input_width, _start, _end,
      static_cast<half const*>(inputs[0]),
      static_cast<half*>(outputs[0]));
  }
  return cudaGetLastError() != cudaSuccess;
}
