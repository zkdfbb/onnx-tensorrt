#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <cassert>
#include <opencv2/opencv.hpp>
#include "NvOnnxParserRuntime.h"
#include "NvInfer.h"
#include "common.h"


using namespace std;
static Logger gLogger;

struct Dim
{
    int B_;
    int C_;
    int H_;
    int W_;

    Dim() : B_(0), C_(0), H_(0), W_(0) {}
    Dim(int B, int C, int H, int W) : B_(B), C_(C), H_(H), W_(W) {}
    Dim(int C, int H, int W) : B_(1), C_(C), H_(H), W_(W) {}
    int size() { return B_ * C_ * H_ * W_; }
};

class BaseNet
{
public:
    BaseNet();
    virtual ~BaseNet();
    int init(vector<string> input_names, vector<Dim> input_dims,
             vector<string> output_names, vector<Dim> output_dims);
    int build(const string &trt_name);
    void infer(vector<float *> &inputs);
    void get_buffer(vector<float *> &outputs);
    void debug(int size);

private:
    int batch_size_;
    vector<string> input_names_;
    vector<string> output_names_;
    vector<Dim> input_dims_;
    vector<Dim> output_dims_;
    vector<int> input_indices_;
    vector<int> output_indices_;
    vector<int> input_sizes_;
    vector<int> output_sizes_;
    void **buffers_;

    nvinfer1::IRuntime *runtime_{nullptr};
    nvinfer1::ICudaEngine *engine_{nullptr};
    nvinfer1::IExecutionContext *context_{nullptr};
    cudaStream_t stream_;
};
