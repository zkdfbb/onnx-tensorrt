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


#include "ReduceSum.hpp"
#include <cuda_fp16.h>
#include <cassert>

nvinfer1::Dims ReduceSumPlugin::getOutputDimensions(int index,
                                                const nvinfer1::Dims *inputDims,
                                                int nbInputs) {
  assert(nbInputs == 1);
  assert(index < this->getNbOutputs());
  nvinfer1::Dims output_dims = inputDims[0];
  output_dims.d[_axis] = 1;
  return output_dims;
}

int ReduceSumPlugin::initialize() {
  _output_dims = this->getOutputDimensions(0, &this->getInputDims(0), 1);
  return 0;
}

template <typename Data>
__global__ void reduce_sum(const int n,
      const int batchsize, const int channels,
      const int height, const int width,
      const Data *idata, Data* odata) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    for (int b = 0; b < batchsize; ++b) {
      Data sum = 0.;
      auto odx = idx - b * (channels - 1) * n;
      for (int c = 0; c < channels; ++c) {
        sum += idata[idx];
        idx += n;
      }
      odata[odx] = sum;
    }
  }
}

int ReduceSumPlugin::enqueue(int batchSize,
                                  const void *const *inputs, void **outputs,
                                  void *workspace, cudaStream_t stream) {
  auto const& input_dims = this->getInputDims(0);
  const int channels = input_dims.d[0];
  const int input_height = input_dims.d[1];
  const int input_width = input_dims.d[2];
  const int num_kernels = input_height * input_width;
  const int num_threads = 512;
  const int num_blocks = (num_kernels + num_threads - 1) / num_threads;
  if (getDataType() == nvinfer1::DataType::kFLOAT) {
    reduce_sum<<<num_blocks, num_threads>>>(num_kernels, batchSize, channels,
      input_height, input_width,
      static_cast<float const*>(inputs[0]),
      static_cast<float*>(outputs[0]));
  } else {
    return -1;
  }
  return cudaGetLastError() != cudaSuccess;
}
