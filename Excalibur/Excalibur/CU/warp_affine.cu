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

template<typename Dtype>
__global__
void kernel_warp_affine(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, double* M_data, int fill, excalibur::interpolationType type, tensorType Ttype)
{
	int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	int rowID = totalID / width;
	int colID = totalID % width;

	double xf = M_data[0] * colID + M_data[1] * rowID + M_data[2];
	double yf = M_data[3] * colID + M_data[4] * rowID + M_data[5];
	int x = (int)xf;
	int y = (int)yf;
	float xdiff = xf - x;
	float ydiff = yf - y;

	if (Ttype == NCHW)
	{
		for (int ch = 0; ch < channels; ++ch)
		{
			int channel_offset = ch * height * width;
			int src_index = channel_offset + y * width + x;
			int dst_index = channel_offset + rowID * width + colID;

			if (x < 0 || x >= width || y < 0 || y >= height)
			{
				dst_data[dst_index] = (Dtype)fill;
			}
			else
			{
				if (type == excalibur::Nearest)
				{
					dst_data[dst_index] = src_data[src_index];
				}
				else if (type == excalibur::Bilinear)
				{
					Dtype A = src_data[src_index];
					Dtype B = src_data[src_index + 1];
					Dtype C = src_data[src_index + width];
					Dtype D = src_data[src_index + width + 1];
					dst_data[dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
						static_cast<float>(B) * xdiff * (1 - ydiff) +
						static_cast<float>(C) * ydiff * (1 - xdiff) +
						static_cast<float>(D) * xdiff * ydiff);
				}
			}
		}
	}
	else if (Ttype == NHWC)
	{
		for (int ch = 0; ch < channels; ++ch)
		{
			int src_index = (y * width + x) * channels + ch;
			int dst_index = (rowID * width + colID) * channels + ch;

			if (x < 0 || x >= width || y < 0 || y >= height)
			{
				dst_data[dst_index] = (Dtype)fill;
			}
			else
			{
				if (type == excalibur::Nearest)
				{
					dst_data[dst_index] = src_data[src_index];
				}
				else if (type == excalibur::Bilinear)
				{
					Dtype A = src_data[src_index];
					Dtype B = src_data[src_index + channels];
					Dtype C = src_data[src_index + width * channels];
					Dtype D = src_data[src_index + width * channels + channels];
					dst_data[dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
						static_cast<float>(B) * xdiff * (1 - ydiff) +
						static_cast<float>(C) * ydiff * (1 - xdiff) +
						static_cast<float>(D) * xdiff * ydiff);
				}
			}
		}
	}
}


//unsigned char    point<float>
void warp_affine_gpu(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst,
	const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear)
{
	if (src_point.size() != 3 || dst_point.size() != 3)
	{
		LOG(WARNING) << "please use 3 src_points and 3 dst_points.";
		return;
	}

	double a[6 * 6], b[6];
	for (int i = 0; i < 3; i++)
	{
		int j = i * 12;
		int k = i * 12 + 6;
		a[j] = a[k + 3] = (double)dst_point[i].x;
		a[j + 1] = a[k + 4] = (double)dst_point[i].y;
		a[j + 2] = a[k + 5] = 1;
		a[j + 3] = a[j + 4] = a[j + 5] = 0;
		a[k] = a[k + 1] = a[k + 2] = 0;
		b[i * 2] = (double)src_point[i].x;
		b[i * 2 + 1] = (double)src_point[i].y;
	}

	cv::Mat A(6, 6, CV_64F, a), B(6, 1, CV_64F, b);
	cv::Mat M(2, 3, CV_64F), X(6, 1, CV_64F, M.ptr());
	cv::solve(A, B, X);

	double *M_data = nullptr;
	cudaMalloc(&M_data, 6 * sizeof(double));
	cudaMemcpy(M_data, M.data, 6 * sizeof(double), cudaMemcpyHostToDevice);

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
	kernel_warp_affine<< <grid_size, block_size >> > (src_data, dst_data, channels, height, width, M_data, fill, type, src->type());
}