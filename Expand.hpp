#pragma once

#include "plugin.hpp"
#include "serialize.hpp"

#include <cassert>

class ExpandPlugin final : public onnx2trt::Plugin {
  int _output_height;
  int _output_width;
  nvinfer1::Dims _output_dims;
protected:
  void deserialize(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    deserialize_value(&serialData, &serialLength, &_output_height);
    deserialize_value(&serialData, &serialLength, &_output_width);
  }
  size_t getSerializationSize() override {
    return getBaseSerializationSize() + serialized_size(_output_height) + serialized_size(_output_width);
  }
  void serialize(void *buffer) override {
    serializeBase(buffer);
    serialize_value(&buffer, _output_height);
    serialize_value(&buffer, _output_width);
  }
public:
  ExpandPlugin(int const& output_height, int const& output_width)
  :_output_height(output_height), _output_width(output_width)
  {}
  ExpandPlugin(void const* serialData, size_t serialLength) {
    this->deserialize(serialData, serialLength);
  }
  virtual const char* getPluginType() const override { return "Expand"; }
  virtual int getNbOutputs() const override { return 1; }
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims *inputs, int nbInputDims) override;
  virtual int initialize() override;
  
  int enqueue(int batchSize,
              const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;
};
