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
void kernel_rotate_with_points(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, double* M_data, int fill = 0, interpolationType type = Bilinear, tensorType Ttype = NCHW)
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
	else if(Ttype == NHWC)
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
void rotate_with_points_gpu(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>> &dst,
	                        const point<float> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear)
{
	if (fabs(theta) <= 1e-6)
	{
		LOG(WARNING) << "Just copy from the source.";
		dst = std::make_shared<tensor<unsigned char>>(src->clone());
		return;
	}

	CHECK_EQ(src->num(), 1);
	int channels = src->channels();
	int height = src->height();
	int width = src->width();
	int offset = height * width;

	double rad = theta*(PI / 180);
	double cosa = cos(rad);
	double sina = sin(rad);

	double a = scale * cosa;
	double b = scale * sina;
	double M_data[9];
	M_data[0] = a;
	M_data[1] = b;
	M_data[2] = (1 - a) * (double)center.x - b * (double)center.y;
	M_data[3] = -1 * b;
	M_data[4] = a;
	M_data[5] = b * (double)center.x + (1 - a) * (double)center.y;
	M_data[6] = 0;
	M_data[7] = 0;
	M_data[8] = 1;

	cv::Mat M(3, 3, CV_64F, M_data);
	cv::Mat reverse_M(3, 3, CV_64F);
	cv::invert(M, reverse_M);
	double *reverse_M_data = nullptr;
	cudaMalloc(&reverse_M_data, 9 * sizeof(double));
	cudaMemcpy(reverse_M_data, reverse_M.data, 9 * sizeof(double), cudaMemcpyHostToDevice);

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
	kernel_rotate_with_points << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, reverse_M_data, fill, type, src->type());
}