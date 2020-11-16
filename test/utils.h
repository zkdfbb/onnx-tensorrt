#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#include "device_launch_parameters.h"
#include "malloc.h"
#include <stdio.h>
#include <cmath>
#include <float.h>
#include <string.h>

using namespace std;

enum init_type {ZERO, COMMON, GRID};

//For template instance
#define INSTANTIATE_CLASS(classname) \
   template class classname<float,int>;  \
   template class classname<double,int>


template<typename T1,typename T2>
class cell{
public:
  
    cell(int n,int c,int h,int w):n_(n),c_(c),h_(h),w_(w){
        spatial_dim_ = h_*w_;
        size_ = n_*c_*h_*w_;
        host_ptr_ = (T1*) malloc(sizeof(T1)*size_);
        memset(host_ptr_,0,sizeof(T1)*size_);
        cudaMalloc(&device_ptr_, sizeof(T1)*size_);
        cudaMemset(device_ptr_, 0x0, sizeof(T1)*size_);
    }
    ~cell(){
        cudaFree(device_ptr_);
        if (host_ptr_ != NULL)
            delete [] host_ptr_;
        host_ptr_ = NULL;
    }
  //type 0 : grid data
  void init_data(init_type type);
  void check_whole_data(init_type type);
  void check_part_data(int size);
  void sync_H2D();
  void sync_D2H();
  void read_data(string filename);
  void write_data(int size,string filename);
  T1* host_ptr();
  T1* device_ptr();

private:
  int n_;
  int c_;
  int h_;
  int w_;
  int spatial_dim_;
  int size_;
  T1* host_ptr_;
  void* device_ptr_;
};

#endif