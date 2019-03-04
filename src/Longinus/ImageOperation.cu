#include "ImageOperation.hpp"
#ifdef USE_CUDA


#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void resize_gpu_bilinear_kernel(const unsigned char* src_data, int old_height, int old_width, int channels,
	unsigned char* dst_data, int new_height, int new_width, int device)
{
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (x >= new_width || y >= new_height)
	{
		return;
	}

	float srcXf = x* ((float)old_width / new_width);
	float srcYf = y* ((float)old_height / new_height);
	int srcX = (int)srcXf;
	int srcY = (int)srcYf;
	if (x == new_width - 1)
	{
		srcX = srcX - 1;
	}
	if (y == new_height - 1)
	{
		srcY = srcY - 1;
	}
	float u = srcXf - srcX;
	float v = srcYf - srcY;

	int dst_offset = new_height * new_width;
	int src_offset = old_height * old_width;

	for (int c = 0; c < channels; c++)
	{
		dst_data[c * dst_offset + y * new_width + x] =
			(1 - u)*(1 - v)*src_data[c * src_offset + srcY * old_width + srcX] +
			(1 - u)*(v)*src_data[c * src_offset + (srcY + 1) * old_width + srcX] +
			(u)*(1 - v)*src_data[c * src_offset + srcY * old_width + srcX + 1] +
			(u)*(v)*src_data[c * src_offset + (srcY + 1) * old_width + srcX + 1];
	}
}

void glasssix::longinus::resize_gpu_bilinear(const unsigned char* src_data, int old_height, int old_width, int channels,
	unsigned char* dst_data, int new_height, int new_width, int device)
{
	int uint = 16;
	dim3 grid((new_width + uint - 1) / uint, (new_height + uint - 1) / uint);
	dim3 block(uint, uint);
	resize_gpu_bilinear_kernel << <grid, block >> > (src_data, old_height, old_width, channels, dst_data, new_height, new_width, device);
}

__global__ void integral_cols_kernel(int* dst_data, int sum_width, int sum_height)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col >= sum_width)
		return;
	dst_data[col] = 0;
	for (int row = 1; row < sum_height; row++)
	{
		dst_data[row * sum_width + col] = dst_data[(row - 1) * sum_width + col] + dst_data[row * sum_width + col];
	}
}

__global__ void integral_rows_kernel(const unsigned char* src_data, int width, int height, int* dst_data, int sum_width)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= height)
		return;
	dst_data[(row + 1) * sum_width] = 0;
	for (int col = 1; col < sum_width; col++)
	{
		dst_data[(row + 1) * sum_width + col] = dst_data[(row + 1) * sum_width + col - 1] + src_data[row * width + col - 1];
	}
}

void glasssix::longinus::integral_gpu(const unsigned char* src_data, int width, int height, int* dst_data, int sum_width, int sum_height)
{
	dim3 gridDim_rows(height / 128 + ((height % 128) == 0 ? 0 : 1));
	dim3 blockDim_rows(128);

	integral_rows_kernel << <gridDim_rows, blockDim_rows >> > (src_data, width, height, dst_data, sum_width);

	dim3 gridDim_cols(sum_width / 128 + ((sum_width % 128) == 0 ? 0 : 1));
	dim3 blockDim_cols(128);

	integral_cols_kernel << <gridDim_cols, blockDim_cols >> > (dst_data, sum_width, sum_height);
}
#endif // USE_CUDA