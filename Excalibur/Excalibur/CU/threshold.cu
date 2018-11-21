
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
void kernel_threshold(const Dtype* src_data, Dtype* dst_data, int src_height, int src_width, int thresh, int maxval, thresholdType type)
{
	int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

	switch (type)
	{
	case binary:
		dst_data[totalID] = src_data[totalID] > (Dtype)thresh ? (Dtype)maxval : (Dtype)0;
		break;

	case binary_inv:
		dst_data[totalID] = src_data[totalID] <= (Dtype)thresh ? (Dtype)maxval : (Dtype)0;
		break;

	case big_trunc:
		if (src_data[totalID] > (Dtype)thresh)
		{
			dst_data[totalID] = (Dtype)thresh;
		}
		else 
		{
			dst_data[totalID] = src_data[totalID];
		}
		break;

	case small_trunc:
		if (src_data[totalID] > (Dtype)thresh)
		{
			dst_data[totalID] = src_data[totalID];
		}
		else
		{
			dst_data[totalID] = (Dtype)thresh;
		}
		break;

	case small_to_zero:
		dst_data[totalID] = src_data[totalID] > (Dtype)thresh ? src_data[totalID] : (Dtype)0;
		break;

	case big_to_zero:
		dst_data[totalID] = src_data[totalID] <= (Dtype)thresh ? src_data[totalID] : (Dtype)0;
		break;

	default:
		break;
	}
}

//unsigned char
void threshold_gpu(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary)
{
	CHECK_EQ(src->num(), 1);
	CHECK_EQ(src->channels(), 1);
	int height = src->height();
	int width = src->width();
	int src_offset = height * width;

	if (src->type() == NCHW)
	{
		dst.reset(new tensor<unsigned char>(std::vector<int>{1, 1, height, width}, src->device(), src->type()));
	}
	else
	{
		dst.reset(new tensor<unsigned char>(std::vector<int>{1, height, width, 1}, src->device(), src->type()));
	}
	
	unsigned char* dst_data = dst->mutable_gpu_data();
	const unsigned char *src_data = src->gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_threshold<unsigned char> << <grid_size, block_size >> > (src_data, dst_data, height, width, thresh, maxval, type);
}