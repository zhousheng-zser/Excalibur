# Excalibur
A thread safe implementation of **CNN** forward propagation part.

This implementation, has been specifically optimized in response to **Intel CPU**, **Nvidia GPU** and **ARM** situation.

## Overview
As we described above, the implementation is able to be divided into 2 parts: CPU and GPU. In order to accelerate the forward propagation, the following frameworks and libraries will be used in the project:
- [Intel® TBB](https://www.threadingbuildingblocks.org/)
- [Intel® Math Kernel Library](https://software.intel.com/en-us/intel-mkl)
- [Intel® MKL-DNN](https://github.com/01org/mkl-dnn)
- [Intel® MPI](https://software.intel.com/en-us/intel-mpi-library)
- [CUDA8.0](https://developer.nvidia.com/cuda-toolkit)
- [cuDNNv6](https://developer.nvidia.com/cudnn)
- [NCCL](https://github.com/NVIDIA/nccl)
- [gemmlowp](https://github.com/inlmouse/gemmlowp)
- [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page)

## Features
  * Suppoert caffemodel directly without any change.
  * Faster implementation on CPU with MKL(specifically, with sgemm_batch() function).
  * Faster implementation on Convolution and Inner-product with sparse model.
  * Half pricision(float16) support on x86(Intel only); fixed pricision(int8) support on x86 and ARM arch.
  
## Implementation log

|  Layers  |   BLAS  |   Native CUDA   | cuDNN | MKLDNN | NEON |
| :------: | :------:| :------: | :------: | :------: | :------: |
| Convolution | T |  T  |  T  |  F  |  F  |
| (P)ReLU | T |  T  |  F  |  F  |  F  |
| Pooling | T |  T  |  F  |  F  |  F  |
| Inner-Product | T |  T  |  F  |  F  |  F  |
| Softmax | T |  T  |  F  |  F  |  F  |
| Eltwise | T |  T  |  F  |  F  |  F  |
| Slice | T |  F  |  F  |  F  |  F  |

## How to use
  * Now, the project is under developing. No offical API has been provied.
  

## Optimization log
### The 1000 average forward time(ms) on CPU(Intel i7-7700k)

| Net(input size)     | Caffe |  mini-Caffe  |  Excalibur  |
| :------: | :------:| :------: | :------: |
| mtcnn-PNet(128*128)  |  5.013 |  3.391  |  4.316  |
| mtcnn-RNet(24*24)  |  0.463 |  0.426  |  0.324  |
| mtcnn-ONet(48*48)  |  1.071 |  0.838  |  0.764  |
| Unicorn(128*128)  |  98.820 |  70.379  |  90.619  |

### The 1000 average forward time(ms) on GPU(NVIDIA GTX1080-Ti) with/without cuDNN

| Net(input size)     | Caffe |  mini-Caffe  |  Excalibur  |
| :------: | :------:| :------: | :------: |
| mtcnn-PNet(128*128)  |  --/-- |  --/--  |  --/--  |
| mtcnn-RNet(24*24)  |  --/-- |  --/--  |  --/--  |
| mtcnn-ONet(48*48)  |  --/-- |  --/--  |  --/--  |
| Unicorn(128*128)  |  7.145/-- |  8.309/--  |  --/6.424  |
  
## Known bugs
  - Due to an unknown reason, the performance of OpenBLAS is very unstable(on Intel i7-7700k). When swtich to Intel MKL, it's slightly faster than caffe 
and similar to mini-caffe.
  - When accessing the col buffer in convolution layer(std::shared_ptr<tensor> col_buffer_) on GPU, it takes plenty of time. We tried to fix it by using a global buffer,
however, this problem will also be exposed while confronted with variable input size(FCN), Such as the PNet in MTCNN.

## Todo list
  * Change some layer implementation into MKL(MKLDNN) and CUDA(CUDNN) implementation.
  * Separate the net's weights and the images calculated to make it threadsafe.
  * Add ARM NEON support.
  * Support multi-machine, multi-uint(CPU) and multi-card(GPU)
  * Compress the model file.
  * More secure model encryption.
  * [ONNX](https://github.com/onnx/onnx) support.
