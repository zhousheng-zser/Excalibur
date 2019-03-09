# Excalibur

A light weighted Kernel C++ library for some **MATH**, **IMAGE** and **CNN** operations. This implementation, has been specifically optimized in response to **Intel CPU**, **Nvidia GPU** and **ARM** situation.

## Overview and Components

As we described above, the implementation is able to be divided into 2 parts: CPU and GPU. In order to accelerate the forward propagation, the following frameworks and libraries may be used(optional) in the project:
- [Intel® TBB](https://www.threadingbuildingblocks.org/)
- [Intel® Math Kernel Library](https://software.intel.com/en-us/intel-mkl)
- [Intel® MPI](https://software.intel.com/en-us/intel-mpi-library)
- [CUDA10.1](https://developer.nvidia.com/cuda-toolkit)
- [cuDNNv7.5](https://developer.nvidia.com/cudnn)
- [NCCL](https://github.com/NVIDIA/nccl)
- [gemmlowp](https://github.com/inlmouse/gemmlowp)
- [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page)
- [OpenBLAS](https://www.openblas.net/)

### Excalibur

Core component of Excalibur. Mainly math, image operation, IO and CNN topology operations.

### Julius

A SIMD supported [BLAS](http://www.netlib.org/blas/) library. For more details, please ref [doc](docs/Julius).

### Cassius(Cassiunia)

A light weighted CNN implementation(with C++/CLI wrapper) of Unicorn Net, aim at face feature extraction. For more details, please ref [doc](docs/Cassius).

### Damocles

An Excalibur based [MTCNN](https://arxiv.org/abs/1604.02878) implementation with faster half O-Net.

### Longinus(Longinucia)

An extremely fast face detection and alignment library(with C++/CLI wrapper). The alignment part was implementated in [Damocles](README.md#Damocles) and [Romancia](README.md#Romancia).
For more details, please ref [doc](docs/Longinus).

### Irisvian

An Extremely Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph. For more details, please ref [doc](docs/Irisvian).

### PersonalReality

A tool of transfering prototxt and caffemodel into hpp and cpp. The last preparatory step to build a CNN with Excalibur. Planning to support [ONNX](https://github.com/onnx/onnx).

### Romancia

An Excalibur based head-pose estimation, landmark detection CNN.

## Features

- Supports convolutional neural networks and most commonly used image processing operations.
- Supports multiple input and multi-branch structure, can calculate part of the branch.
- No third-party library dependencies in CPU-Only mode, does not rely on BLAS / NNPACK or any other computing framework.
- Pure C++ implementation, easy to compile cross platform(Windows, Linux and MacOS).
- Sophisticated memory management and data structure design, very low cost in CPU and GPU interaction.
- Hard code models into executable file in binary to protect intellectual property.
- Can be registered with custom operations implementation and extended.
- Faster implementation on CPU for various types convolution operation.
- Support Convolution and Inner-product with sparse model.
- Half pricision(float16) support on x86(NVIDIA GPU only); fixed pricision(int8) support on x86 and ARM arch.
  

## Contributors

- Glasssix Research: [J. Hu](https://github.com/inlmouse)
- Glasssix Research: [Y. Zhang](https://github.com/zhangyifu2016)
- Glasssix Research: [J. Zhang](https://github.com/fengye2two)

## Copyright

Copyright © 2014 - 2019 Glasssix. All Rights Reserved. 

第六镜科技 版权所有


