
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
void kernel_resize(int channels, const Dtype* src_data, int src_height, int src_width,
	Dtype* dst_data, int dst_height, int dst_width,
	float height_ratio, float width_ratio, interpolationType type, tensorType Ttype)
{
	int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	int rowID = totalID / dst_width;
	int colID = totalID % dst_width;

	float xf = colID * width_ratio;
	float yf = rowID * height_ratio; 
	int x = (int)xf;
	int y = (int)yf;
	float xdiff = xf - x;
	float ydiff = yf - y;

	if (Ttype == NCHW)
	{
		int src_offset = src_height * src_width;
		int dst_offset = dst_height * dst_width;

		for (int ch = 0; ch < channels; ++ch)
		{
			int src_channel_offset = ch * src_offset;
			int dst_channel_offset = ch * dst_offset;
			int src_index = src_channel_offset + y * src_width + x;
			int dst_index = dst_channel_offset + rowID * dst_width + colID;

			if (type == Nearest)
			{
				dst_data[dst_index] = src_data[src_index];
			}
			else if (type == Bilinear)
			{
				Dtype A = src_data[src_index];
				Dtype B = src_data[src_index + 1];
				Dtype C = src_data[src_index + src_width];
				Dtype D = src_data[src_index + src_width + 1];
				dst_data[dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
					static_cast<float>(B) * xdiff * (1 - ydiff) +
					static_cast<float>(C) * ydiff * (1 - xdiff) +
					static_cast<float>(D) * xdiff * ydiff);
			}
		}
	}
	else if (Ttype == NHWC)
	{
		for (int ch = 0; ch < channels; ++ch)
		{
			int src_index = (y * src_width + x) * channels + ch;
			int dst_index = (rowID * dst_width + colID) * channels + ch;

			if (type == Nearest)
			{
				dst_data[dst_index] = src_data[src_index];
			}
			else if (type == Bilinear)
			{
				Dtype A = src_data[src_index];
				Dtype B = src_data[src_index + channels];
				Dtype C = src_data[src_index + src_width * channels];
				Dtype D = src_data[src_index + src_width * channels + channels];
				dst_data[dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
					static_cast<float>(B) * xdiff * (1 - ydiff) +
					static_cast<float>(C) * ydiff * (1 - xdiff) +
					static_cast<float>(D) * xdiff * ydiff);
			}
		}
	}
}

//unsigned char
void resize_gpu(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst,
	int dst_height, int dst_width, interpolationType type = Bilinear)
{
	if (dst_height * dst_width <= 0)
	{
		LOG(ERROR) << "Illegal input size.";
		return;
	}

	CHECK_EQ(src->num(), 1);
	int channels = src->channels();
	int height = src->height();
	int width = src->width();

	if (dst_width == width && dst_height == height)
	{
		LOG(WARNING) << "Just copy from the source.";
		dst = std::make_shared<tensor<unsigned char>>(src->clone());
		return;
	}

	if (src->type() == NCHW)
	{
		dst.reset(new tensor<unsigned char>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->type()));
	}
	else
	{
		dst.reset(new tensor<unsigned char>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->type()));
	}

	unsigned char* dst_data = dst->mutable_gpu_data();
	const unsigned char* src_data = src->gpu_data();

	float width_ratio = (float)(width - 1) / (dst_width - 1);
	float height_ratio = (float)(height - 1) / (dst_height - 1);

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(dst_width, dst_height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_resize << <grid_size, block_size >> > (channels, src_data, height, width, dst_data, dst_height, dst_width, height_ratio, width_ratio, type, src->type());
}