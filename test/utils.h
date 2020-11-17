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



template<typename T1,typename T2>
class cell{
public:
  
    cell(int n,int c,int h,int w):n_(n),c_(c),h_(h),w_(w){
       this->init();
    }
    cell(Dim dim):dim_(dim){
        n_ = dim_.B_;
        c_ = dim_.C_;
        h_ = dim_.H_;
        w_ = dim_.W_;
        this->init();
    }
    ~cell(){
        cudaFree(device_ptr_);
        if (host_ptr_ != NULL)
            delete [] host_ptr_;
        host_ptr_ = NULL;
    }
  
  void init();
  void init_data(init_type type);
  void check_whole_data(init_type type);
  void check_part_data(int size);
  void sync_H2D();
  void sync_D2H();
  void read_data(string filename);
  void write_data(int size,string filename);
  void write_other_data(float*ptr,int size,string filename);
  T1* host_ptr();
  T1* device_ptr();

private:
  Dim dim_;
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