
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
void kernel_rotate_with_center(int channels, const Dtype* src_data, int height, int width,
	                                Dtype* dst_data, int dst_height, int dst_width,
	                                float sina, float cosa,  float varX, float varY, int fill, interpolationType type, tensorType Ttype)
{
				int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
				int rowID = totalID / dst_width;
				int colID = totalID % dst_width;

				float xf = cosa * colID + sina * rowID + varX;
				float yf = -sina * colID + cosa * rowID + varY;

				int x = (int)(xf); 
				int y = (int)(yf);
				float xdiff = xf - x;
				float ydiff = yf - y;

				if (Ttype == NCHW) 
				{
					int src_offset = height * width;
					int dst_offset = dst_height * dst_width;

					for (int ch = 0; ch < channels; ++ch)
					{
						int src_channel_offset = ch * src_offset;
						int dst_channel_offset = ch * dst_offset;
						int src_index = src_channel_offset + y * width + x;
						int dst_index = dst_channel_offset + rowID * dst_width + colID;

						if (x >= width || x < 0 || y >= height || y < 0)
						{
							dst_data[dst_index] = (Dtype)fill;
						}
						else
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
				else if (Ttype == NHWC)
				{
					int src_pos1 = (y * width + x) * channels;
					int dst_pos1 = (rowID * dst_width + colID) * channels;

					for (int ch = 0; ch < channels; ++ch)
					{
						int src_pos2 = src_pos1 + ch;
						int dst_pos2 = dst_pos1 + ch;

						if (x >= width || x < 0 || y >= height || y < 0)
						{
							dst_data[dst_pos2] = (Dtype)fill;
						}
						else
						{
							Dtype A = src_data[src_pos2];
							Dtype B = src_data[src_pos2 + channels];
							Dtype C = src_data[src_pos2 + width * channels];
							Dtype D = src_data[src_pos2 + width * channels + channels];
							dst_data[dst_pos2] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
								static_cast<float>(B) * xdiff * (1 - ydiff) +
								static_cast<float>(C) * ydiff * (1 - xdiff) +
								static_cast<float>(D) * xdiff * ydiff);
						}
					}
				}
}

//unsigned char
void rotate_with_center_gpu(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst,
	float theta, int &dst_height, int &dst_width, int fill = 0, interpolationType type = Bilinear)
{
	if (fabs(theta) <= 1e-6)
	{
		LOG(WARNING) << "Just copy from the source.";
		dst = std::make_shared<excalibur::tensor<unsigned char>>(src->clone());
	}

	CHECK_EQ(src->num(), 1);
	int channels = src->channels();
	int height = src->height();
	int width = src->width();

	float rad = -1 * theta*(PI / 180);//逆时针为正
	float cosa = cos(rad);
	float sina = sin(rad);

	dst_width = (int)(width * abs(cosa) + height * abs(sina));
	dst_height = (int)(width * abs(sina) + height * abs(cosa));

	float VarX = (float)(-dst_width * cosa / 2.0f - dst_height * sina / 2.0f + width / 2.0f);
	float VarY = (float)(dst_width * sina / 2.0f - dst_height * cosa / 2.0f + height / 2.0f);

	if (src->type() == NCHW)
	{
		dst.reset(new excalibur::tensor<unsigned char>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->type()));
	}
	else 
	{
		dst.reset(new excalibur::tensor<unsigned char>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->type()));
	}

	unsigned char* dst_data = dst->mutable_gpu_data();
	const unsigned char* src_data = src->gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(dst_width, dst_height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_rotate_with_center << <grid_size, block_size >> > (channels, src_data, height, width, dst_data, dst_height, dst_width, sina, cosa, VarX, VarY, fill, type, src->type());
}