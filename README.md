# Excalibur

A light weighted Kernel C++ library for some **MATH**, **IMAGE** and **CNN** operations. This implementation, has been specifically optimized in response to **Intel CPU**, **Nvidia GPU** and **ARM** situation.

## Features

- Supports convolutional neural networks and most commonly used image processing operations.
- Supports multiple input and multi-branch structure, can calculate part of the branch.
- No third-party library dependencies in CPU-Only mode, does not rely on BLAS / NNPACK or any other computing framework.
- Pure C++ implementation, cross platform(Windows, x86-Linux, ARM-Linux, Android, iOS and MacOS) support.
- Multi-language support, except C++ native support, CSharp(.Net framework and .Net Core), Java/Koltin(Android), JavaScript(coming soon) and WebAssenbly(coming soon) interfaces are also included. 
- Sophisticated memory management and data structure design, very low cost in CPU and GPU interaction.
- Hard code models into executable file in binary to protect intellectual property.
- Can be registered with custom operations implementation and extended.
- Faster implementation on CPU for various types convolution operation.
- Support Convolution and Inner-product with sparse model.
- Half pricision(float16) support on x86(NVIDIA GPU only); fixed pricision(int8) support on x86 and ARM arch.

## Overview and Components

As we described above, the implementation is able to be divided into 2 parts: CPU and GPU. In order to accelerate the forward propagation, the following frameworks and libraries may be used(optional) in the project:
- [Intel® TBB](https://www.threadingbuildingblocks.org/)
- [Intel® Math Kernel Library](https://software.intel.com/en-us/intel-mkl)
- [Intel® MPI](https://software.intel.com/en-us/intel-mpi-library)
- [CUDA10.1](https://developer.nvidia.com/cuda-toolkit)
- [cuDNNv7.5](https://developer.nvidia.com/cudnn)
- [NCCL](https://github.com/NVIDIA/nccl)

### Excalibur

Core component of Excalibur. Mainly math, image operation, IO and CNN topology operations.

### Julius

A SIMD supported [BLAS](http://www.netlib.org/blas/) library. For more details, please ref [doc](docs/Julius).

### Primitives

The foundation of Excalibur. please ref [doc](docs/Primitives).

