# Excalibur
A thread safe implementation of **CNN** forward propagation part.

This implementation, has been specifically optimized in response to **Intel CPU**, **Nvidia GPU** and **ARM** situation.

## Overview and Components

As we described above, the implementation is able to be divided into 2 parts: CPU and GPU. In order to accelerate the forward propagation, the following frameworks and libraries will be used in the project:
- [Intel® TBB](https://www.threadingbuildingblocks.org/)
- [Intel® Math Kernel Library](https://software.intel.com/en-us/intel-mkl)
- [Intel® MPI](https://software.intel.com/en-us/intel-mpi-library)
- [CUDA8.0](https://developer.nvidia.com/cuda-toolkit)
- [cuDNNv7](https://developer.nvidia.com/cudnn)
- [NCCL](https://github.com/NVIDIA/nccl)
- [gemmlowp](https://github.com/inlmouse/gemmlowp)
- [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page)
- [OpenBLAS](https://www.openblas.net/)

### Excalibur

Core component of Excalibur. Mainly math, IO and CNN topology operations.

### Cassius

A C++/CLI wrapper of Unicorn Net.

### Longinus

A C++/CLI wrapper of IPBBox_v2 and 5IPTs_v2 Net.

### exfastfacedetection

A CUDA NPD implementation.

### fastfacedetection

A Excalibur based MTCNN implementation.

### libcufacedetection

A C++ interface of all detection algorithms.

### Romancia

A C++/CLI interface of all detection algorithms.

### PersonalReality

A tool of transfering prototxt and caffemodel into hpp and cpp. The last preparatory step to build a CNN with Excalibur.

### Atalanta

A gpu infomation infer tool, depends on NVIDIA driver only.

## Features
  * Hard code your model into executable file in binary.
  * Faster implementation on CPU with MKL(specifically, with sgemm_batch() function).
  * Faster implementation on Convolution and Inner-product with sparse model.
  * Half pricision(float16) support on x86(Intel only); fixed pricision(int8) support on x86 and ARM arch.
  
## Implementation log

|  Layers  |   BLAS  |   Native CUDA   | cuDNN | MKLDNN | NEON |
| :------: | :------:| :------: | :------: | :------: | :------: |
| Convolution | TP |  TP  |  TE  |  F  |  F  |
| (P)ReLU | TP |  TP  |  F  |  F  |  F  |
| Pooling | TP |  TP  |  F  |  F  |  F  |
| Inner-Product | TP |  TP  |  F  |  F  |  F  |
| Softmax | TP |  TP  |  F  |  F  |  F  |
| Eltwise | TP |  TP  |  --  |  --  |  F  |
| Slice | TE |  TE  |  --  |  --  |  F  |
| Flip | TP |  TP  |  --  |  --  |  F  |
| Concat | TP |  TP  |  --  |  --  |  F  |
| Normalize | TP |  TP  |  --  |  --  |  F  |
| MirrorMax | TP |  TP  |  --  |  --  |  F  |
| PCA | NT |  NT  |  --  |  --  |  F  |

  - TP: Implementated and test passed;
  - F: Not implementated;
  - TE: Implementated but error exists;
  - --: No implementation;
  - NT: Implementated but no test yet;
 
 
|  Operations  |  CPU  |  GPU   |
| :------: | :------:| :------: |
| resize | NT | -- |
| flip | NT | -- |
| cut | NT | -- |
| copymakeborder | NT | -- |
| rotate | -- | -- |
| grayscale | NT | -- |

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
| Unicorn_with_MirrorFace(128*128)  |  201.603 |  --  |  160.554  |

### The 1000 average forward time(ms) on GPU(NVIDIA GTX1080-Ti) with/without cuDNN

| Net(input size)     | Caffe |  mini-Caffe  |  Excalibur  |
| :------: | :------:| :------: | :------: |
| mtcnn-PNet(128*128)  |  --/-- |  --/--  |  --/--  |
| mtcnn-RNet(24*24)  |  --/-- |  --/--  |  --/--  |
| mtcnn-ONet(48*48)  |  --/-- |  --/--  |  --/--  |
| Unicorn(128*128)  |  7.145/-- |  8.309/--  |  --/6.424  |
| Unicorn_with_MirrorFace(128*128)  |  8.347/-- |  --/--  |  --/12.880  |
  
## Known bugs
  - Due to an unknown reason, the performance of OpenBLAS is very unstable(on Intel i7-7700k). When swtich to Intel MKL, it's slightly faster than caffe 
and similar to mini-caffe.
  - When accessing the col buffer in convolution layer(std::shared_ptr<tensor> col_buffer_) on GPU, it takes plenty of time. We tried to fix it by using a global buffer,
however, this problem will also be exposed while confronted with variable input size(FCN), Such as the PNet in MTCNN.
  - An unknown heap corruption will occur when using Slice Forward_cpu operation. The mirror face trick should be implementated in another way.
  - The *cudnnGetConvolutionForwardAlgorithm* function in cuDNN does not get the correct workspace.

## Todo list
  * Change some layer implementation into MKL(MKLDNN) and CUDA(CUDNN) implementation.
  * ~~Separate the net's weights and the images calculated to make it threadsafe.~~
  * Add ARM NEON support.
  * Support multi-machine, multi-uint(CPU) and multi-card(GPU)
  * ~~Compress the model file.~~
  * [ONNX](https://github.com/onnx/onnx) support.
