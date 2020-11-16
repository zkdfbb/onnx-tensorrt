cmake_minimum_required(VERSION 3.0)
# this is required
SET(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "-std=c++11 -g -Wno-write-strings")
SET(CMAKE_BUILD_TYPE "Release")
add_compile_options(-fPIC)

SET(PROJECT "trt_check")
OPTION(CMAKE_62 "Whether cmake on server 62" OFF)

if(NOT CMAKE_62)
    SET(AIRBENDER_DEV_ROOT /home/ais01/codes/wenbo/airbender-linux-devtools)
    SET(CMAKE_C_COMPILER   ${AIRBENDER_DEV_ROOT}/tegra-4.9-nv/usr/bin/aarch64-gnu-linux/aarch64-gnu-linux-gcc)
    SET(CMAKE_CXX_COMPILER  ${AIRBENDER_DEV_ROOT}/tegra-4.9-nv/usr/bin/aarch64-gnu-linux/aarch64-gnu-linux-g++)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
endif()

FIND_PACKAGE(CUDA REQUIRED)
if(NOT CMAKE_62)
    set(CUDA_NVCC_FLAGS
          -Xcompiler '-fPIC'
         -gencode arch=compute_62,code=sm_62;
         -ccbin ${CMAKE_C_COMPILER}
         -I ${AIRBENDER_DEV_ROOT}/cuda-8.0/include
         -L ${AIRBENDER_DEV_ROOT}/cuda-8.0/lib
         )
    SET(CUDA_LIBRARIES "${AIRBENDER_DEV_ROOT}/cuda-8.0/lib/libcudart.so")
    SET(CUDA_INCLUDE_DIRS "${AIRBENDER_DEV_ROOT}/cuda-8.0/include")
else()
    set(CUDA_NVCC_FLAGS
         -gencode arch=compute_62,code=sm_62;
         -gencode arch=compute_62,code=compute_62;
         -gencode arch=compute_61,code=sm_61;
         -ccbin ${CMAKE_C_COMPILER}
         )
endif()
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--expt-relaxed-constexpr;)

INCLUDE_DIRECTORIES("/usr/local/include")
LINK_DIRECTORIES("/usr/local/lib")
if(NOT CMAKE_62)
    INCLUDE_DIRECTORIES("${AIRBENDER_DEV_ROOT}/tensorrt3/include")
    INCLUDE_DIRECTORIES("${AIRBENDER_DEV_ROOT}/tegra-t18x/usr/include")
    INCLUDE_DIRECTORIES("${AIRBENDER_DEV_ROOT}/cuda-8.0/include")
    INCLUDE_DIRECTORIES("${AIRBENDER_DEV_ROOT}/tegra-t18x/usr/include/opencv")
    INCLUDE_DIRECTORIES("${AIRBENDER_DEV_ROOT}/tegra-t18x/usr/include/opencv2")
    INCLUDE_DIRECTORIES("${AIRBENDER_DEV_ROOT}/tegra-t18x/usr/include")

    LINK_DIRECTORIES("${AIRBENDER_DEV_ROOT}/tensorrt3/lib")
    LINK_DIRECTORIES("${AIRBENDER_DEV_ROOT}/cuda-8.0/lib")
    LINK_DIRECTORIES("${AIRBENDER_DEV_ROOT}/tegra-t18x/usr/lib")
    LINK_DIRECTORIES("${AIRBENDER_DEV_ROOT}/tegra-t18x/lib")
else()
    INCLUDE_DIRECTORIES("/usr/local/opencv/include")
    INCLUDE_DIRECTORIES("/usr/local/cuda-8.0/include")
    INCLUDE_DIRECTORIES("/usr/local/tensorrt/include")
    INCLUDE_DIRECTORIES("/usr/local/tensorrt/samples/common")
    LINK_DIRECTORIES("/usr/local/opencv/lib")
    LINK_DIRECTORIES("/usr/local/cuda-8.0/lib64")
    LINK_DIRECTORIES("/usr/local/tensorrt/lib")
endif()

SET(SOURCE_FILES main.cpp net.cpp utils.cpp)
SET(OPENCV_LIB opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs)
SET(TENSORRT_LIB cudnn cublas cudart nvinfer nvinfer_plugin nvonnxparser_runtime)
SET(LIBS  ${OPENCV_LIB} ${TENSORRT_LIB})

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
ADD_EXECUTABLE(${PROJECT} ${SOURCE_FILES})
TARGET_LINK_LIBRARIES(${PROJECT} ${LIBS})
