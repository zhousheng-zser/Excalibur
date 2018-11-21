
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

#define PI 3.1415926

template<typename Dtype>
__global__
void kernel_morph(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, excalibur::morphType type, int ksize, tensorType Ttype)
{
	int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	int rowID = totalID / width;
	int colID = totalID % width;

	int half = (ksize - 1) * 0.5;
	int offset = height * width;

	if (Ttype == NCHW)
	{
		if (type == Dilate)
		{
			for (int ch = 0; ch < channels; ++ch)
			{
				int channel_offset = ch * offset;
				int index = channel_offset + rowID * width + colID;
				int max = -99999;
				for (int kernel_row = -1 * half; kernel_row <= half; ++kernel_row)
				{
					if (rowID + kernel_row < 0 || rowID + kernel_row >= height)
					{
						continue;
					}

					for (int kernel_col = -1 * half; kernel_col <= half; ++kernel_col)
					{
						if (colID + kernel_col < 0 || colID + kernel_col >= width)
						{
							continue;
						}

						int pos = index + kernel_row * width + kernel_col;
						if (src_data[pos] > max)
						{
							max = src_data[pos];
						}
					}
				}
				dst_data[index] = (Dtype)max;
			}
		}
		else if (type == Erode)
		{
			for (int ch = 0; ch < channels; ++ch)
			{
				int channel_offset = ch * offset;
				int index = channel_offset + rowID * width + colID;
				int min = 99999;
				for (int kernel_row = -1 * half; kernel_row <= half; ++kernel_row)
				{
					if (rowID + kernel_row < 0 || rowID + kernel_row >= height)
					{
						continue;
					}

					for (int kernel_col = -1 * half; kernel_col <= half; ++kernel_col)
					{
						if (colID + kernel_col < 0 || colID + kernel_col >= width)
						{
							continue;
						}

						int pos = index + kernel_row * width + kernel_col;
						if (src_data[pos] < min)
						{
							min = src_data[pos];
						}
					}
				}
				dst_data[index] = (Dtype)min;
			}
		}
	}
	else if (Ttype == NHWC)
	{
		if (type == Dilate)
		{
			for (int ch = 0; ch < channels; ++ch)
			{
				int index = (rowID * width + colID) * channels + ch;
				int max = -99999;
				for (int kernel_row = -1 * half; kernel_row <= half; ++kernel_row)
				{
					if (rowID + kernel_row < 0 || rowID + kernel_row >= height)
					{
						continue;
					}

					for (int kernel_col = -1 * half; kernel_col <= half; ++kernel_col)
					{
						if (colID + kernel_col < 0 || colID + kernel_col >= width)
						{
							continue;
						}

						int pos = index + (kernel_row * width + kernel_col) * channels;
						if (src_data[pos] > max)
						{
							max = src_data[pos];
						}
					}
				}
				dst_data[index] = (Dtype)max;
			}
		}
		else if (type == Erode)
		{
			for (int ch = 0; ch < channels; ++ch)
			{
				int index = (rowID * width + colID) * channels + ch;
				int min = 99999;
				for (int kernel_row = -1 * half; kernel_row <= half; ++kernel_row)
				{
					if (rowID + kernel_row < 0 || rowID + kernel_row >= height)
					{
						continue;
					}

					for (int kernel_col = -1 * half; kernel_col <= half; ++kernel_col)
					{
						if (colID + kernel_col < 0 || colID + kernel_col >= width)
						{
							continue;
						}

						int pos = index + (kernel_row * width + kernel_col) * channels;
						if (src_data[pos] < min)
						{
							min = src_data[pos];
						}
					}
				}
				dst_data[index] = (Dtype)min;
			}
		}
	}
}


//unsigned char
void morph_gpu(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>> &dst, excalibur::morphType type = Dilate, int ksize = 3)
{
	if (ksize % 2 != 1)
	{
		LOG(WARNING) << "ksize should be odd.";
		return;
	}

	if (ksize == 1)
	{
		dst = std::make_shared<tensor<unsigned char>>(src->clone());
		LOG(WARNING) << "Just copy from the source.";
		return;
	}

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
	
	unsigned char* dst_data = dst->mutable_gpu_data();
	const unsigned char* src_data = src->gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_morph << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, type, ksize, src->type());
}