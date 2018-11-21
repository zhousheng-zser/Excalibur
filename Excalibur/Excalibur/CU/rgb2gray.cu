
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <glasssix\accelerator.hpp>
#include "tensor_utils.hpp"
#include "tensor.hpp"
#include <iostream>
using namespace excalibur;

template<typename Dtype>
__global__
void kernel_rgb2gray(const Dtype* src_data, Dtype* dst_data, int height, int width, tensorType Ttype)
{
	int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	int rowID = totalID / width;
	int colID = totalID % width;

	int index = rowID * width + colID;
	int offset = height * width;

	//opencv读取RGB图像后，以B、G、R的顺序进行存储
	//转换公式为:gray=0.114*B+0.587*G+0.299*R
	if (Ttype == NCHW) 
	{
		dst_data[index] = Dtype(static_cast<float>(src_data[index]) * 0.114f +
			static_cast<float>(src_data[offset * 1 + index]) * 0.587f +
			static_cast<float>(src_data[offset * 2 + index]) * 0.299f);
	}
	else if (Ttype == NHWC) 
	{
		dst_data[index] = Dtype(static_cast<float>(src_data[3 * index]) * 0.114f +
			static_cast<float>(src_data[3 * index + 1]) * 0.587f +
			static_cast<float>(src_data[3 * index + 2]) * 0.299f);
	}
}

//unsigned char
void rgb2gray_gpu(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst)
{
	CHECK_EQ(src->num(), 1);
	int channels = src->channels();
	int height = src->height();
	int width = src->width();

	if (channels != 3)
	{
		LOG(ERROR) << "Incorrect input channel.";
		return;
	}

	if (src->type() == NCHW)
	{
		dst.reset(new tensor<unsigned char>(std::vector<int>{1, 1, height, width}, src->device(), src->type()));
	}
	else
	{
		dst.reset(new tensor<unsigned char>(std::vector<int>{1, height, width, 1}, src->device(), src->type()));
	}

	const unsigned char* src_data = src->gpu_data();
	unsigned char* dst_data = dst->mutable_gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_rgb2gray << <grid_size, block_size >> > (src_data, dst_data, height, width, src->type());
}