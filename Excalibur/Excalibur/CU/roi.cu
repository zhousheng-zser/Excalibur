
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

template<typename Dtype, typename Rtype>
__global__
void kernel_ROI(const Dtype* src_data, int channels, int src_height, int src_width,
	Dtype* dst_data, excalibur::rectangle<Rtype> rect, tensorType Ttype)
{
	int dst_height = (int)rect.h;
	int dst_width = (int)rect.w;

	int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	int rowID = totalID / dst_width;
	int colID = totalID % dst_width;

	if (Ttype == NCHW) 
	{
		int dst_offset = dst_height * dst_width;
		int src_offset = src_height * src_width;

		for (int ch = 0; ch < channels; ++ch)
		{
			int src_channel_offset = ch * src_offset;
			int dst_channel_offset = ch * dst_offset;
			int src_index = src_channel_offset + (rowID + rect.y) * src_width + (colID + rect.x);
			int dst_index = dst_channel_offset + rowID * dst_width + colID;

			dst_data[dst_index] = src_data[src_index];
		}
	}
	else if (Ttype == NHWC) 
	{
		for (int ch = 0; ch < channels; ++ch)
		{
			int src_index = (rowID + rect.y) * src_width * channels + (colID + rect.x) * channels + ch;
			int dst_index = (rowID * dst_width + colID) * channels + ch;

			dst_data[dst_index] = src_data[src_index];
		}
	}
}

//unsigned char
void roi_gpu(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst, excalibur::rectangle<int> rect)
{
	CHECK_EQ(src->num(), 1);
	int channels = src->channels();
	int height = src->height();
	int width = src->width();
	int src_offset = height * width;

	if (rect.x < 0 || rect.x + rect.w >= width || rect.y < 0 || rect.y + rect.h >= height) {
		LOG(WARNING) << "rect is out of image";
		return;
	}

	int dst_height = (int)rect.h;
	int dst_width = (int)rect.w;
	int dst_offset = dst_height * dst_width;

	if (dst_height == height && dst_width == width)
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

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(dst_width, dst_height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_ROI << <grid_size, block_size >> > (src_data, channels, height, width, dst_data, rect, src->type());
}
