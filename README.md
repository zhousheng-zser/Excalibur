# MCL_Forward
A thread safe re-implementation of [MathCoreLibrary](https://github.com/CompileSense/Temporary_MathCoreLibrary) forward propagation part.

This implementation, has been specifically optimized in response to **Intel CPU**, **Nvidia GPU** and **ARM** situation.

## Overview
As we described above, the implementation is able to be divided into 2 parts: CPU and GPU. In order to accelerate the forward propagation, the following frameworks and libraries will be used in the project:
- [Intel® TBB](https://www.threadingbuildingblocks.org/)
- [Intel® Math Kernel Library](https://software.intel.com/en-us/intel-mkl)
- [Intel® MKL-DNN](https://github.com/01org/mkl-dnn)
- [Intel® MPI](https://software.intel.com/en-us/intel-mpi-library)
- [CUDA8.0](https://developer.nvidia.com/cuda-toolkit)
- [cuDNNv5.1](https://developer.nvidia.com/cudnn)
- [NCCL](https://github.com/NVIDIA/nccl)
- [gemmlowp](https://github.com/inlmouse/gemmlowp)
- [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page)

## Features
  * Suppoert caffemodel directly without any change.
  * Faster implementation on CPU with MKL(specifically, with sgemm_batch() function).
  * Faster implementation on Convolution and Inner-product with sparse model.
  * Half pricision(float16) support on x86(Intel only); fixed pricision(int8) support on x86 and ARM arch.
  * Supported layers currently: `INPUT`, `CONVOLUTION`, `POOLING`, `DENSE`(or `INNER_PRODUCT`), `RELU`.

## How to use
  * Now, the project is under developing. No offical API has been provied.

## Todo list
  * Change some layer implementation into MKL(MKLDNN) and CUDA(CUDNN) implementation.
  * Separate the net's weights and the images calculated to make it threadsafe.
  * Add ARM NEON support.
  * Support multi-machine, multi-uint(CPU) and multi-card(GPU)
  * Add more layer support, such as PReLU, Eltwise, Softmax...
  * Compress the model file.
  * More secure model encryption.
