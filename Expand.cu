#include "Expand.hpp"
#include <cuda_fp16.h>
#include <cassert>
// it is worthless, just for virtual
nvinfer1::Dims ExpandPlugin::getOutputDimensions(int index,
    const nvinfer1::Dims *inputDims, int nbInputs)
{
    assert(nbInputs == 1);
    assert(index < this->getNbOutputs());
    nvinfer1::Dims const& input_dims = inputDims[0];
    nvinfer1::Dims output_dims = input_dims;
    return output_dims;
}
int ExpandPlugin::initialize()
{
    return 0;
}
__global__ void tile(const float *input,float *output,int n, int dim,int w,int h,int c)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int batch_idx = index / dim;
    int channel_idx = (index - batch_idx*dim) /(w * h); 
    if(index < n)
    {
        output[index] = input[batch_idx * c + channel_idx];
    } 
}
int ExpandPlugin::enqueue(int batchSize, 
                        const void *const *inputs, void **outputs,
                        void *workspace, cudaStream_t stream)
{
    auto const& input_dims = this->getInputDims(0);
    const int channels = input_dims.d[0];
    const int num_threads = 256;
    const int n = _output_width * _output_height * channels * batchSize;
    const int dim = _output_width * _output_height * channels;
    const int num_blocks = (n + num_threads - 1 ) / num_threads;
    const float *input = static_cast<float const*>(inputs[0]);
    float *output = static_cast<float *>(outputs[0]);
    // cout << "output hhhhhhhh " << output_height << output_width << channels <<  endl;
    if (getDataType() == nvinfer1::DataType::kFLOAT) 
    {
        tile<<<num_blocks,num_threads>>>(input,output,n,dim,_output_width,_output_height,channels);
    } 
    else
    {
        return -1;
    }
    
    return cudaGetLastError() != cudaSuccess;
}