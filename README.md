# MCL_Forward
A thread safe re-implementation of [MathCoreLibrary](https://github.com/CompileSense/Temporary_MathCoreLibrary) forward propagation part.

This implementation, has been specifically optimized in response to **Intel CPU** and **Nvidia GPU** situation. As  for embedded/ mobile terminal situation, you may need [MCLdroid]().

## Overview
As we described above, the implementation is able to be divided into 2 parts: CPU and GPU. In order to accelerate the forward propagation, the following frameworks and libraries will be used in the project:
- [Intel® TBB](https://www.threadingbuildingblocks.org/)
- [Intel® Math Kernel Library](https://software.intel.com/en-us/intel-mkl)
- [Intel® MKL-DNN](https://github.com/01org/mkl-dnn)
- [Intel® MPI](https://software.intel.com/en-us/intel-mpi-library)
- [CUDA8.0](https://developer.nvidia.com/cuda-toolkit)
- [cuDNNv5.1](https://developer.nvidia.com/cudnn)
- cuBLAS

## Features
  * Suppoert caffemodel directly without any change.
  * Faster implementation on CPU with MKL(specifically, with sgemm_batch() function).
  * Supported layers currently: `INPUT`, `CONVOLUTION`, `POOLING`, `DENSE`(or `INNER_PRODUCT`), `RELU`.

## How to use
  * Now, the project is under developing. No offical API has been provied.

## Todo list
  * Change some layer implementation into MKL and  and CUDA implementation.
  * Separate the net's weights and the images calculated to make it threadsafe.
  * Support multi-machine, multi-uint(CPU) and multi-card(GPU)
  * Add more layer support, such as PReLU, Eltwise, Softmax...
  * Compress the model file.
  * More secure model encryption.
