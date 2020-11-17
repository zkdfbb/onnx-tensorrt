#include "GridSampler.hpp"
#include <cuda_fp16.h>
#include <cassert>
#include <iostream>
#include <fstream>

//设置output 的dim
nvinfer1::Dims GridSamplerPlugin::getOutputDimensions(int index,
    const nvinfer1::Dims *inputDims,
    int nbInputs) {
    //input [0] : feature input[1] : grid
    assert(nbInputs == 2);
    assert(index < this->getNbOutputs());
    nvinfer1::Dims output_dims = inputDims[1];  //output size  = input_c * grid_h * gird_w;
    output_dims.d[0] = inputDims[0].d[0];
    output_dims.d[1] = inputDims[1].d[0];
    output_dims.d[2] = inputDims[1].d[1];
    return output_dims;
}


int GridSamplerPlugin::initialize() {
    return 0;
}


using namespace std;

#define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < W && y >= 0 && y < H)

#undef MIN
#define MIN(a,b) ( ((a)<(b)) ? (a) : (b) )
#undef MAX
#define MAX(a,b) ( ((a)>(b)) ? (a) : (b) )
#define CLIP_COORDINATES(in, out, clip_limit) out = MIN((clip_limit-1), MAX(in, 0))

#define INT_MAX 2147483647
#define INT_MIN (-INT_MAX - 1)


#define CUDA_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename In, typename Out>
struct ScalarConvert {
    static __host__ __device__ Out to(const In v) { return (Out) v; }
};

// DEPRECATED: use static_cast in kernels instead of scalar_cast
template <typename T, typename U>
__host__ __device__ T scalar_cast(U u) {
    return ScalarConvert<U, T>::to(u);
}

__forceinline__ __device__ float safe_downgrade_to_int_range(float x){
    // -100.0 does not have special meaning. This is just to make sure 
    // it's not within_bounds_2d or within_bounds_3d, and does not cause 
    // undefined behavior. See #35506.  
    if (x > INT_MAX-1 || x < INT_MIN || !::isfinite(static_cast<double>(x))) 
      return static_cast<float>(-100.0); 
    return x;
  }
  
  
  
  __forceinline__ __device__ float clip_coordinates(float in, int64_t clip_limit) {
      return ::min(static_cast<float>(clip_limit - 1), ::max(in, static_cast<float>(0)));
  }
  
  __forceinline__ __device__ float compute_coordinates(float coord, int64_t size) {
    //coord = clip_coordinates(coord, size);
    coord = safe_downgrade_to_int_range(coord);
    return coord;
  }
  
  
  
  __forceinline__ __device__ float grid_sampler_unnormalize(float coord, int64_t size) {
      return ((coord + 1) * size - 1) / 2;
  }
  
  
  
  __forceinline__ __device__ float grid_sampler_compute_source_index(
      float coord,
      int64_t size) {
    coord = grid_sampler_unnormalize(coord, size);
    coord = compute_coordinates(coord, size);
    return coord;
  }
  



template <typename Dtype>
__launch_bounds__(1024)
__global__ void SpatialGridSamplerBilinear(
    const int grid_h,
    const int grid_w,
    const int IH,
    const int IW,
    const int C,
    const int nthreads,
    const Dtype* input,
    const Dtype* grid,
    Dtype* output){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_dim = grid_w * grid_h;
    const int spatial_dim = IH * IW;
   

    CUDA_KERNEL_LOOP(idx, nthreads){
        const int n = idx / (grid_dim*2);
        const int w = (idx / 2) % grid_w;
        const int h = (idx / 2) / grid_w;
        int c;
        Dtype ix = grid[n * grid_dim * 2 + grid_w * h *2 + w*2];
        Dtype iy = grid[n * grid_dim * 2 + grid_w * h *2 + w*2 + 1];

          
        ix = grid_sampler_compute_source_index(ix,IW);
        iy = grid_sampler_compute_source_index(iy,IH);
        int ix_nw = static_cast<int>(::floor(ix));
        int iy_nw = static_cast<int>(::floor(iy));

        // ix = ScalarConvert<float,Dtype>::to(((ix + 1) / 2) * (IW-1));
        // iy = ScalarConvert<float,Dtype>::to(((iy + 1) / 2) * (IH-1));
        // int ix_nw = floor(ScalarConvert<Dtype,float>::to(ix));
        // int iy_nw = floor(ScalarConvert<Dtype,float>::to(iy));
        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;
        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;
        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;

        // get surfaces to each neighbor:
        Dtype nw = (ix_se - ix)    * (iy_se - iy);
        Dtype ne = (ix    - ix_sw) * (iy_sw - iy);
        Dtype sw = (ix_ne - ix)    * (iy    - iy_ne);
        Dtype se = (ix    - ix_nw) * (iy    - iy_nw);

        Dtype out_val = 0;
        #pragma unroll
        for(c = 0;c<C;++c){
            out_val = static_cast<Dtype>(0);
            if (WITHIN_BOUNDS(ix_nw, iy_nw, IH, IW)) {
                out_val += input[n * C * spatial_dim + c * spatial_dim + iy_nw * IW + ix_nw] * nw; 
            }
            if (WITHIN_BOUNDS(ix_ne, iy_ne, IH, IW)) {
                out_val += input[n * C * spatial_dim + c * spatial_dim + iy_ne * IW + ix_ne] * ne;
            }
            if (WITHIN_BOUNDS(ix_sw, iy_sw, IH, IW)) {
                out_val += input[n * C * spatial_dim + c* spatial_dim + iy_sw * IW + ix_sw] * sw;
            }
            if (WITHIN_BOUNDS(ix_se, iy_se, IH, IW)) {
                out_val += input[n * C * spatial_dim + c* spatial_dim + iy_se * IW + ix_se] * se;
            }
            output[n * C * grid_dim + c * grid_dim + h * grid_w + w] = out_val;
        }   
    }
}

int GridSamplerPlugin::enqueue(int batchSize,
    const void *const *inputs, void **outputs,
    void *workspace, cudaStream_t stream){
    
    auto const& input_dims_f = this->getInputDims(0);
    auto const& input_dims_g = this->getInputDims(1);
    const int grid_h = input_dims_g.d[0];
    const int grid_w = input_dims_g.d[1];
    const int idx_count = input_dims_g.d[2];
    const int IH = input_dims_f.d[1];
    const int IW = input_dims_f.d[2];
    const int C =input_dims_f.d[0];
    int threadPerBlock = 128;
    int blockPerGrid = 256;
    const int nthreads = batchSize  * grid_h * grid_w * idx_count;
    const float *input = static_cast<float const*>(inputs[0]);
    const float *grid  = static_cast<float const*>(inputs[1]);
    float *output = static_cast<float *>(outputs[0]);
    if (getDataType() == nvinfer1::DataType::kFLOAT) {
        SpatialGridSamplerBilinear<float><<<blockPerGrid,threadPerBlock>>>(grid_h,grid_w,IH,IW,C,nthreads,input,grid,output);
    }
    else{
        return -1;
    }
    return cudaGetLastError() != cudaSuccess; 

}
