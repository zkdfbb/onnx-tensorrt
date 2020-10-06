#!/usr/bin/env bash
# -*- coding: utf-8 -*-

##############################################

# Tesla V100
# # ARCH= -gencode arch=compute_70,code=[sm_70,compute_70]
# 
# GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4
# # ARCH= -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61
# 
# GP100/Tesla P100 DGX-1
# # ARCH= -gencode arch=compute_60,code=sm_60
# 
# For Jetson Tx1 uncomment:
# # ARCH= -gencode arch=compute_51,code=[sm_51,compute_51]
# 
# For Jetson Tx2 or Drive-PX2 uncomment:
# # ARCH= -gencode arch=compute_62,code=[sm_62,compute_62]

# ModelImporter.cpp is Modified

# Custom TensorRT Plugin: ResizeBilinear Slice(on channel axis) Mish
# Slice only supported numpy operator : not support :: and onnx export opset=9

##############################################

set -e -x

rm -rf build
mkdir build && cd build

function cx2(){
    cmake-3.10.2 \
        -DTENSORRT_ROOT=/home/ais01/codes/wenbo/airbender-linux-devtools/tensorrt3 \
        -DGPU_ARCHS="62" \
        -DTX2=on ..
    make -j32
    rsync -avP onnx2trt 11:/usr/local/bin/onnx2trt
    rsync -avP lib* 11:/usr/local/lib/
}

function x86(){
    cmake .. -DTENSORRT_ROOT=/usr/local/tensorrt -DGPU_ARCHS="61"
    make -j32
    sudo make install
}

$@
