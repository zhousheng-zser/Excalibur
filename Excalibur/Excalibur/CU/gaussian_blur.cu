
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
void kernel_gaussian_blur(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, int ksize, double *paras, tensorType Ttype)
{
	int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	int rowID = totalID / width;
	int colID = totalID % width;

	int half = (ksize - 1) * 0.5;

	if (Ttype == NCHW) 
	{
		int offset = height * width;
		for (int ch = 0; ch < channels; ++ch)
		{
			int channel_offset = ch * offset;
			int index = channel_offset + rowID * width + colID;
			double sum = 0;
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

					sum += paras[(kernel_row + half) * ksize + (kernel_col + half)] * src_data[index + kernel_row * width + kernel_col];
				}
			}
			dst_data[index] = (Dtype)sum;
		}
	}
	else if (Ttype == NHWC)
	{
		for (int ch = 0; ch < channels; ++ch)
		{
			int index = (rowID * width + colID) * channels + ch;
			double sum = 0;
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

					sum += paras[(kernel_row + half) * ksize + (kernel_col + half)] * src_data[index + (kernel_row * width + kernel_col) * channels];
				}
			}
			dst_data[index] = (Dtype)sum;
		}
	}
}

//unsigned char
void gaussian_blur_gpu(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>> &dst, int ksize = 3)
{
	if (ksize % 2 != 1)
	{
		LOG(WARNING) << "convolution kernel: width and height should be odd.";
		return;
	}

	if (ksize == 1)
	{
		LOG(WARNING) << "Just copy from the source.";
		dst = std::make_shared<tensor<unsigned char>>(src->clone());
		return;
	}

	CHECK_EQ(src->num(), 1);
	int channels = src->channels();
	int height = src->height();
	int width = src->width();

	double sigma = ((ksize - 1)*0.5 - 1)*0.3 + 0.8;
	double scale2X = (double)1 / (2 * sigma * sigma);
	double prefix = scale2X / PI;

	double sum = 0;
	int half = (ksize - 1) * 0.5;
	double *convolution_kernel = (double*)malloc(ksize * ksize * sizeof(double));

	for (int row = 0; row < ksize; ++row)
	{
		double dy = row - half;
		for (int col = 0; col < ksize; ++col)
		{
			double dx = col - half;
			double distance = dx * dx + dy * dy;
			convolution_kernel[row * ksize + col] = prefix * std::exp(-1 * distance * scale2X);
			sum += convolution_kernel[row * ksize + col];
		}
	}

	for (int row = 0; row < ksize; ++row)
	{
		for (int col = 0; col < ksize; ++col)
		{
			convolution_kernel[row * ksize + col] /= sum;
		}
	}

	double *paras = nullptr;
	cudaMalloc(&paras, ksize * ksize * sizeof(double));
	cudaMemcpy(paras, convolution_kernel, ksize * ksize * sizeof(double), cudaMemcpyHostToDevice);
	
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
	kernel_gaussian_blur << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, ksize, paras, src->type());
}