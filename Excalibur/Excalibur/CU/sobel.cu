#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <glasssix\accelerator.hpp>
#include "tensor_utils.hpp"
#include "tensor.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace excalibur;

#define PI 3.1415926

template<typename Dtype>
__global__
void kernel_sobel(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, int dx, int dy, tensorType Ttype)
{
	int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	int rowID = totalID / width;
	int colID = totalID % width;

	//第一行、最后一行、第一列、最后一列不作处理
	if (rowID == 0 || rowID == height - 2 || colID == 0 || colID == width - 2) 
	{
		return;
	}

	if (Ttype == NCHW)
	{
		int offset = height * width;
		for (int ch = 0; ch < channels; ++ch)
		{
			int sumx = 0, sumy = 0, total = 0;
			int channel_offset = ch * offset;
			int pos = channel_offset + rowID * width + colID;
			int posAdd = pos + width;
			int posSub = pos - width;

			if (dx > 0)
			{
				sumx = src_data[posSub + 1] + 2 * src_data[pos + 1] + src_data[posAdd + 1]
					- src_data[posSub - 1] - 2 * src_data[pos - 1] - src_data[posAdd - 1];
			}

			if (dy > 0)
			{
				sumy = src_data[posSub - 1] + 2 * src_data[posSub] + src_data[posSub + 1]
					- src_data[posAdd - 1] - 2 * src_data[posAdd] - src_data[posAdd + 1];
			}

			total = abs(sumx) + abs(sumy);
			if (dx != 0 && dy != 0)
			{
				total = 0.35 * total;
			}
			else
			{
				total = 0.6 * total;
			}

			if (total > 255)
			{
				total = 255;
			}

			dst_data[pos] = (Dtype)total;
		}
	}
	else if (Ttype == NHWC)
	{
		for (int ch = 0; ch < channels; ++ch)
		{
			int sumx = 0, sumy = 0, total = 0;
			int pos = (rowID * width + colID) * channels + ch;
			int posAdd = pos + width * channels;
			int posSub = pos - width * channels;

			if (dx > 0)
			{
				sumx = src_data[posSub + channels] + 2 * src_data[pos + channels] + src_data[posAdd + channels]
					- src_data[posSub - channels] - 2 * src_data[pos - channels] - src_data[posAdd - channels];
			}

			if (dy > 0)
			{
				sumy = src_data[posSub - channels] + 2 * src_data[posSub] + src_data[posSub + channels]
					- src_data[posAdd - channels] - 2 * src_data[posAdd] - src_data[posAdd + channels];
			}

			total = abs(sumx) + abs(sumy);
			if (dx != 0 && dy != 0)
			{
				total = 0.35 * total;
			}
			else
			{
				total = 0.6 * total;
			}

			if (total > 255)
			{
				total = 255;
			}

			dst_data[pos] = (Dtype)total;
		}
	}
}


//unsigned char
void sobel_gpu(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>> &dst, int dx = 1, int dy = 1)
{
	if (dx == 0 && dy == 0) 
	{
		LOG(WARNING) << "dx, dy cannot be zero simultaneously.";
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
	kernel_sobel << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, dx, dy, src->type());
}