
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
void kernel_flip(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, flipType axis, tensorType Ttype)
{
	int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	int rowID = totalID / width;
	int colID = totalID % width;

	if (Ttype == NCHW)
	{
		int offset = height * width;

		if (axis == Width_Wise)
		{
			for (int ch = 0; ch < channels; ++ch)
			{
				int channel_offset = ch * offset;
				int index = channel_offset + rowID * width;
				dst_data[index + colID] = src_data[index + (width - colID - 1)];
			}
		}
		else if (axis == Height_Wise)
		{
			for (int ch = 0; ch < channels; ++ch)
			{
				int channel_offset = ch * offset;
				int dst_index = channel_offset + rowID * width;
				int src_index = channel_offset + (height - rowID - 1) * width;
				dst_data[dst_index + colID] = src_data[src_index + colID];
			}
		}
		else if (axis == Center_Wise)
		{
			for (int ch = 0; ch < channels; ++ch)
			{
				int channel_offset = ch * offset;
				int dst_index = channel_offset + rowID * width;
				int src_index = channel_offset + (height - rowID - 1) * width;
				dst_data[dst_index + colID] = src_data[src_index + (width - colID - 1)];
			}
		}
		else if (axis == Channel_Wise)
		{
			for (int ch = 0; ch < channels; ++ch)
			{
				int dst_channel_offset = ch * offset;
				int src_channel_offset = (channels - ch - 1) * offset;
				int dst_index = dst_channel_offset + rowID * width;
				int src_index = src_channel_offset + rowID * width;
				dst_data[dst_index + colID] = src_data[src_index + colID];
			}
		}
	}
	else if (Ttype == NHWC) 
	{
		if (axis == Width_Wise)
		{
			for (int ch = 0; ch < channels; ++ch)
			{
				int dst_index = (rowID * width + colID) * channels + ch;
				int src_index = (rowID * width + (width - 1 - colID)) * channels + ch;
				dst_data[dst_index] = src_data[src_index];
			}
		}
		else if (axis == Height_Wise)
		{
			for (int ch = 0; ch < channels; ++ch)
			{
				int dst_index = (rowID * width + colID) * channels + ch;
				int src_index = ((height - 1 - rowID) * width + colID) * channels + ch;
				dst_data[dst_index] = src_data[src_index];
			}
		}
		else if (axis == Center_Wise)
		{
			for (int ch = 0; ch < channels; ++ch)
			{
				int dst_index = (rowID * width + colID) * channels + ch;
				int src_index = ((height - 1 - rowID) * width + (width - 1 - colID)) * channels + ch;
				dst_data[dst_index] = src_data[src_index];
			}
		}
		else if (axis == Channel_Wise)
		{
			for (int ch = 0; ch < channels; ++ch)
			{
				int dst_index = (rowID * width + colID) * channels + ch;
				int src_index = (rowID * width + colID) * channels + channels - 1 - ch;
				dst_data[dst_index] = src_data[src_index];
			}
		}
	}
}

//unsigned char
void flip_gpu(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst, flipType axis = Width_Wise)
{
	CHECK_EQ(src->num(), 1);
	int channels = src->channels();
	int height = src->height();
	int width = src->width();

	if (src->type() == NCHW)
	{
		dst.reset(new tensor<unsigned char>(std::vector<int>{1, channels, height, width}, src->device(), src->type()));
	}
	else
	{
		dst.reset(new tensor<unsigned char>(std::vector<int>{1, height, width, channels}, src->device(), src->type()));
	}
	
	const unsigned char* src_data = src->gpu_data();
	unsigned char* dst_data = dst->mutable_gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_flip << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, axis, src->type());
}