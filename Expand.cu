#include "Expand.hpp"
#include <cuda_fp16.h>
#include <cassert>
// it is worthless, just for virtual
nvinfer1::Dims ExpandPlugin::getOutputDimensions(int index,
    const nvinfer1::Dims *inputDims, int nbInputs)
{
    assert(nbInputs == 1);
    assert(index < this->getNbOutputs());
    nvinfer1::Dims const& input = inputDims[0];
    
    nvinfer1::Dims output;
    output.nbDims = input.nbDims;
    for( int d=0; d<input.nbDims; ++d )
    {
        output.type[d] = input.type[d];
    }
    output.d[0] = input.d[0];
    output.d[1] = _output_height;
    output.d[2] = _output_width;
    return output;
}
int ExpandPlugin::initialize()
{
    _output_dims = this->getOutputDimensions(0, &this->getInputDims(0), 1);
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
    const int output_height = _output_dims.d[1];
    const int output_width = _output_dims.d[2];
    const int n = output_width * output_height * channels * batchSize;
    const int dim = output_width * output_height * channels;
    const int num_blocks = (n + num_threads - 1 ) / num_threads;
    const float *input = static_cast<float const*>(inputs[0]);
    float *output = static_cast<float *>(outputs[0]);
    // cout << "output height " << output_height << endl;
    // cout << "output width" << output_width << endl;
    // cout << "chanensls " << channels << endl;
    if (getDataType() == nvinfer1::DataType::kFLOAT) 
    {
        tile<<<num_blocks,num_threads>>>(input,output,n,dim,output_width,output_height,channels);
    } 
    else
    {
        return -1;
    }
    
    return cudaGetLastError() != cudaSuccess;
}
