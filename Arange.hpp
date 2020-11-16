
#pragma once

#include "plugin.hpp"
#include "serialize.hpp"

#include <cassert>

class ArangePlugin final : public onnx2trt::Plugin {
protected:
  void deserialize(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
  }
  size_t getSerializationSize() override {
    return getBaseSerializationSize();
  }
  void serialize(void *buffer) override {
    serializeBase(buffer);
  }
public:
  ArangePlugin(){}
  ArangePlugin(void const* serialData, size_t serialLength) {
    this->deserialize(serialData, serialLength);
  }
  virtual const char* getPluginType() const override { return "Arange"; }
  virtual int getNbOutputs() const override { return 1; }
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims *inputs, int nbInputDims) override;
  virtual int initialize() override;
  int enqueue(int batchSize,
              const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;
};
