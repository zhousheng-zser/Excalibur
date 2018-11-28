
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
void kernel_matrix_transpose(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, tensorType Ttype)
{
	int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	int rowID = totalID / width;
	int colID = totalID % width;

	if (Ttype == NCHW) 
	{
		int offset = height * width;

		int dst_index = rowID * width + colID;
		int src_index = colID * height + rowID;

		for (int ch = 0; ch < channels; ++ch) {
			int channel_offset = ch * offset;
			dst_data[dst_index + channel_offset] = src_data[src_index + channel_offset];
		}
	}
	else if (Ttype == NHWC) 
	{
		for (int ch = 0; ch < channels; ++ch) 
		{
			int dst_index = (rowID * width + colID) * channels + ch;
			int src_index = (colID * height + rowID) * channels + ch;
			dst_data[dst_index] = src_data[src_index];
		}
	}
}

//unsigned char
void matrix_transpose_gpu(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst)
{
	CHECK_EQ(src->num(), 1);
	int channels = src->channels();
	int height = src->height();
	int width = src->width();
	int offset = height * width;

	if (src->type() == NCHW)
	{
		dst.reset(new tensor<unsigned char>(std::vector<int>{1, channels, width, height}, src->device(), src->type()));
	}
	else
	{
		dst.reset(new tensor<unsigned char>(std::vector<int>{1, width, height, channels}, src->device(), src->type()));
	}

	const unsigned char* src_data = src->gpu_data();
	unsigned char* dst_data = dst->mutable_gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(height, width, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_matrix_transpose << <grid_size, block_size >> > (src_data, dst_data, channels, width, height, src->type());
}