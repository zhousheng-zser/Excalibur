
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "tensor_utils.hpp"
#include "tensor.hpp"
#include <iostream>
#include <opencv2\opencv.hpp>
#include <glasssix\timer.hpp>
#include "tensor_operation_gpu.hpp"
using namespace excalibur;
#define PI 3.1415926



/// <summary>
/// 根据数据类型获取对应的Mat类型
/// </summary>
/// <returns></returns>
template <typename Dtype>
static int get_cv_type()
{
	auto name = typeid(Dtype).name();
	if (std::string("unsigned char") == std::string(name))
	{
		return 0;//CV_8U   0
	}
	else if (std::string("char") == std::string(name))
	{
		return 1;//CV_8S   1
	}
	else if (std::string("int") == std::string(name))
	{
		return 4;//CV_32S  4
	}
	else if (std::string("float") == std::string(name))
	{
		return 5;//CV_32F  5
	}
	else
	{
		return -1;
	}
}



/// <summary>
/// 将NHWC排列的数据转换为NCHW排列
/// </summary>
/// <param name="src_data">NHWC排列的数据</param>
/// <param name="dst_data">NCHW排列的数据</param>
/// <param name="channels">通道数</param>
/// <param name="height">图像高度</param>
/// <param name="width">图像宽度</param>
template<typename Dtype>
__global__
void kernel_nhwc2nchw(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width)
{
	int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	int rowID = totalID / width;
	int colID = totalID % width;

	int offset = height * width;
	for (int ch = 0; ch < channels; ++ch)
	{
		dst_data[ch * offset + rowID * width + colID] = src_data[(rowID * width + colID) * channels + ch];
	}
}



/// <summary>
/// 将NCHW排列的数据转换为NHWC排列
/// </summary>
/// <param name="src_data">NCHW排列的数据</param>
/// <param name="dst_data">NHWC排列的数据</param>
/// <param name="channels">通道数</param>
/// <param name="height">图像高度</param>
/// <param name="width">图像宽度</param>
template<typename Dtype>
__global__
void kernel_nchw2nhwc(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width)
{
	int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	int rowID = totalID / width;
	int colID = totalID % width;

	int offset = height * width;
	for (int ch = 0; ch < channels; ++ch)
	{
		dst_data[(rowID * width + colID) * channels + ch] = src_data[ch * offset + rowID * width + colID];
	}
}



#ifdef USE_OPENCV

/// <summary>
/// 将tensor结构数据转换为opencv的Mat结构数据
/// </summary>
/// <param name="src">tensor结构数据</param>
/// <param name="dst">opencv的Mat结构数据</param>
template<typename Dtype>
void tensor_operation_gpu::tensor2mat_gpu(const std::shared_ptr<tensor<Dtype>> &src, cv::Mat& dst)
{
	CHECK_EQ(src->num(), 1);
	int channels = src->channels();
	if (channels>4)
	{
		LOG(ERROR) << "Too many channels.";
		return;
	}
	int width = src->width();
	int height = src->height();
	int type = get_cv_type<Dtype>();
	if (type<0)
	{
		LOG(ERROR) << "Un-support data type.";
		return;
	}
	dst = cv::Mat(height, width, CV_MAKETYPE(type, channels));
	const Dtype* src_data = src->gpu_data();


	if (src->type() == NHWC)
	{
		cudaMemcpy(dst.data, src->gpu_data(), height * width * channels * sizeof(Dtype), cudaMemcpyDeviceToHost);
		return;
	}

	std::shared_ptr<tensor<Dtype>> dst_ptr = std::make_shared<tensor<Dtype>>(tensor<Dtype>(std::vector<int>{1, height, width, channels}, 0, NHWC));


	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_nchw2nhwc << <grid_size, block_size >> > (src->gpu_data(), dst_ptr->mutable_gpu_data(), channels, height, width);

	cudaMemcpy(dst.data, dst_ptr->gpu_data(), height * width * channels * sizeof(Dtype), cudaMemcpyDeviceToHost);
}



/// <summary>
/// 将tensor结构数据转换为opencv的Mat结构数据
/// </summary>
/// <param name="src">tensor结构数据</param>
/// <param name="dst">opencv的Mat结构数据</param>
template<typename Dtype>
void tensor_operation_gpu::tensor2mat_gpu(const tensor<Dtype>& src, cv::Mat& dst)
{
	CHECK_EQ(src.num(), 1);
	int channels = src.channels();
	if (channels>4)
	{
		LOG(ERROR) << "Too many channels.";
		return;
	}
	int width = src.width();
	int height = src.height();
	int type = get_cv_type<Dtype>();
	if (type<0)
	{
		LOG(ERROR) << "Un-support data type.";
		return;
	}
	dst = cv::Mat(height, width, CV_MAKETYPE(type, channels));
	const Dtype* src_data = src.gpu_data();


	if (src.type() == NHWC)
	{
		cudaMemcpy(dst.data, src.gpu_data(), height * width * channels * sizeof(Dtype), cudaMemcpyDeviceToHost);
		return;
	}

	std::shared_ptr<tensor<Dtype>> dst_ptr = std::make_shared<tensor<Dtype>>(tensor<Dtype>(std::vector<int>{1, height, width, channels}, 0, NHWC));


	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_nchw2nhwc << <grid_size, block_size >> > (src.gpu_data(), dst_ptr->mutable_gpu_data(), channels, height, width);

	cudaMemcpy(dst.data, dst_ptr->gpu_data(), height * width * channels * sizeof(Dtype), cudaMemcpyDeviceToHost);
}






/// <summary>
/// 将opencv的Mat结构数据转换为tensor结构数据
/// </summary>
/// <param name="src">opencv的Mat结构数据</param>
/// <param name="dst">转换后的tensor结构数据</param>
/// <param name="Ttype">tensor类型：NHWC(tensor按NHWC排列，默认值)、NCHW(tensor按NCHW排列)</param>
template<typename Dtype>
void tensor_operation_gpu::mat2tensor_gpu(const cv::Mat &src, std::shared_ptr<tensor<Dtype>>& dst, tensorType Ttype = NHWC)
{
	int channels = src.channels();
	int width = src.cols;
	int height = src.rows;
	if (src.data == nullptr)
	{
		LOG(ERROR) << "No data.";
		return;
	}
	int type_id = src.type() % 8;
	auto type_name = std::string(typeid(Dtype).name());
	if (type_id == 0)
	{
		if (type_name != std::string("unsigned char"))
		{
			LOG(ERROR) << "Un-matched data type.";
			return;
		}
	}
	else if (type_id == 1)
	{
		if (type_name != std::string("char"))
		{
			LOG(ERROR) << "Un-matched data type.";
			return;
		}
	}
	else if (type_id == 4)
	{
		if (type_name != std::string("int"))
		{
			LOG(ERROR) << "Un-matched data type.";
			return;
		}
	}
	else if (type_id == 5)
	{
		if (type_name != std::string("float"))
		{
			LOG(ERROR) << "Un-matched data type.";
			return;
		}
	}
	else
	{
		LOG(ERROR) << "Un-support data type.";
		return;
	}


	if (Ttype == NHWC)
	{
		dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, 0, Ttype));
		cudaMemcpy(dst->mutable_gpu_data(), src.data, height * width * channels * sizeof(Dtype), cudaMemcpyHostToDevice);
		return;
	}

	dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, 0, Ttype));
	std::shared_ptr<tensor<Dtype>> src_ptr = std::make_shared<tensor<Dtype>>(tensor<Dtype>(std::vector<int>{1, channels, height, width}, 0, Ttype));
	cudaMemcpy(src_ptr->mutable_gpu_data(), src.data, height * width * channels * sizeof(Dtype), cudaMemcpyHostToDevice);

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_nhwc2nchw << <grid_size, block_size >> > (src_ptr->gpu_data(), dst->mutable_gpu_data(), channels, height, width);
}



/// <summary>
/// 将opencv的Mat结构数据转换为tensor结构数据
/// </summary>
/// <param name="src">opencv的Mat结构数据</param>
/// <param name="dst">转换后的tensor结构数据</param>
/// <param name="Ttype">tensor类型：NHWC(tensor按NHWC排列，默认值)、NCHW(tensor按NCHW排列)</param>
template<typename Dtype>
void tensor_operation_gpu::mat2tensor_gpu(const cv::Mat &src, tensor<Dtype>& dst, tensorType Ttype = NHWC)
{
	int channels = src.channels();
	int width = src.cols;
	int height = src.rows;
	if (src.data == nullptr)
	{
		LOG(ERROR) << "No data.";
		return;
	}
	int type_id = src.type() % 8;
	auto type_name = std::string(typeid(Dtype).name());
	if (type_id == 0)
	{
		if (type_name != std::string("unsigned char"))
		{
			LOG(ERROR) << "Un-matched data type.";
			return;
		}
	}
	else if (type_id == 1)
	{
		if (type_name != std::string("char"))
		{
			LOG(ERROR) << "Un-matched data type.";
			return;
		}
	}
	else if (type_id == 4)
	{
		if (type_name != std::string("int"))
		{
			LOG(ERROR) << "Un-matched data type.";
			return;
		}
	}
	else if (type_id == 5)
	{
		if (type_name != std::string("float"))
		{
			LOG(ERROR) << "Un-matched data type.";
			return;
		}
	}
	else
	{
		LOG(ERROR) << "Un-support data type.";
		return;
	}


	if (Ttype == NHWC)
	{
		dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, 0, Ttype);
		cudaMemcpy(dst.mutable_gpu_data(), src.data, height * width * channels * sizeof(Dtype), cudaMemcpyHostToDevice);
		return;
	}

	dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, 0, Ttype);
	std::shared_ptr<tensor<Dtype>> src_ptr = std::make_shared<tensor<Dtype>>(tensor<Dtype>(std::vector<int>{1, channels, height, width}, 0, Ttype));
	cudaMemcpy(src_ptr->mutable_gpu_data(), src.data, height * width * channels * sizeof(Dtype), cudaMemcpyHostToDevice);

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_nhwc2nchw << <grid_size, block_size >> > (src_ptr->gpu_data(), dst.mutable_gpu_data(), channels, height, width);
}

#endif



/// <summary>
/// 将NCHW排列的数据转换为NHWC排列
/// </summary>
/// <param name="src">NCHW排列的tensor数据</param>
/// <param name="dst">NHWC排列的tensor数据</param>
template<typename Dtype>
void tensor_operation_gpu::nchw2nhwc_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst)
{
	CHECK_EQ(src->type(), NCHW);
	CHECK_EQ(src->num(), 1);
	int height = src->height();
	int width = src->width();
	int channels = src->channels();
	int offset = height * width;

	dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), NHWC));

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_nchw2nhwc << <grid_size, block_size >> > (src->gpu_data(), dst->mutable_gpu_data(), channels, height, width);
}



/// <summary>
/// 将NCHW排列的数据转换为NHWC排列
/// </summary>
/// <param name="src">NCHW排列的tensor数据</param>
/// <param name="dst">NHWC排列的tensor数据</param>
template<typename Dtype>
void tensor_operation_gpu::nchw2nhwc_gpu(const tensor<Dtype> &src, tensor<Dtype> &dst)
{
	CHECK_EQ(src.type(), NCHW);
	CHECK_EQ(src.num(), 1);
	int height = src.height();
	int width = src.width();
	int channels = src.channels();
	int offset = height * width;

	dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), NHWC);

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_nchw2nhwc << <grid_size, block_size >> > (src.gpu_data(), dst.mutable_gpu_data(), channels, height, width);
}






/// <summary>
/// 将NHWC排列的数据转换为NCHW排列
/// </summary>
/// <param name="src">NHWC排列的tensor数据</param>
/// <param name="dst">NCHW排列的tensor数据</param>
template<typename Dtype>
void tensor_operation_gpu::nhwc2nchw_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst)
{
	CHECK_EQ(src->type(), NHWC);
	CHECK_EQ(src->num(), 1);
	int height = src->height();
	int width = src->width();
	int channels = src->channels();
	int offset = height * width;

	dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), NCHW));

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_nhwc2nchw << <grid_size, block_size >> > (src->gpu_data(), dst->mutable_gpu_data(), channels, height, width);
}



/// <summary>
/// 将NHWC排列的数据转换为NCHW排列
/// </summary>
/// <param name="src">NHWC排列的tensor数据</param>
/// <param name="dst">NCHW排列的tensor数据</param>
template<typename Dtype>
void tensor_operation_gpu::nhwc2nchw_gpu(const tensor<Dtype> &src, tensor<Dtype> &dst)
{
	CHECK_EQ(src.type(), NHWC);
	CHECK_EQ(src.num(), 1);
	int height = src.height();
	int width = src.width();
	int channels = src.channels();
	int offset = height * width;

	dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), NCHW);

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_nhwc2nchw << <grid_size, block_size >> > (src.gpu_data(), dst.mutable_gpu_data(), channels, height, width);
}






/// <summary>
/// 将图像变换到新的宽度和高度
/// </summary>
/// <param name="channels">图像通道数</param>
/// <param name="src_data">原图像数据</param>
/// <param name="src_height">原图像高度</param>
/// <param name="src_width">原图像宽度</param>
/// <param name="dst_data">尺寸变换后的新图像数据</param>
/// <param name="dst_height">新图像高度</param>
/// <param name="dst_width">新图像宽度</param>
/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值，默认）</param>
/// <param name="Ttype">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
template<typename Dtype>
__global__
void kernel_resize(int channels, const Dtype* src_data, int src_height, int src_width,
	Dtype* dst_data, int dst_height, int dst_width,
	interpolationType type, tensorType Ttype)
{
	int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	int rowID = totalID / dst_width;
	int colID = totalID % dst_width;
	float beta = 0.5f;

	float width_ratio = (float)src_width / dst_width;
	float height_ratio = (float)src_height / dst_height;

	float xf = colID * width_ratio + beta;
	float yf = rowID * height_ratio + beta;
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



/// <summary>
/// 将图像变换到新的宽度和高度
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">尺寸变换后的tensor</param>
/// <param name="dst_height">新的图像高度</param>
/// <param name="dst_width">新的图像宽度</param>
/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值，默认）</param>
template<typename Dtype>
void tensor_operation_gpu::resize_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst,
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
		dst = std::make_shared<tensor<Dtype>>(src->clone());
		return;
	}

	if (src->type() == NCHW)
	{
		dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->type()));
	}
	else
	{
		dst.reset(new tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->type()));
	}

	Dtype* dst_data = dst->mutable_gpu_data();
	const Dtype* src_data = src->gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(dst_width, dst_height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_resize << <grid_size, block_size >> > (channels, src_data, height, width, dst_data, dst_height, dst_width, type, src->type());
}



/// <summary>
/// 将图像变换到新的宽度和高度
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">尺寸变换后的tensor</param>
/// <param name="dst_height">新的图像高度</param>
/// <param name="dst_width">新的图像宽度</param>
/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值，默认）</param>
template<typename Dtype>
void tensor_operation_gpu::resize_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst,
	int dst_height, int dst_width, interpolationType type = Bilinear)
{
	if (dst_height * dst_width <= 0)
	{
		LOG(ERROR) << "Illegal input size.";
		return;
	}

	CHECK_EQ(src.num(), 1);
	int channels = src.channels();
	int height = src.height();
	int width = src.width();

	if (dst_width == width && dst_height == height)
	{
		LOG(WARNING) << "Just copy from the source.";
		dst = src.clone();
		return;
	}

	if (src.type() == NCHW)
	{
		dst = tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src.device(), src.type());
	}
	else
	{
		dst = tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src.device(), src.type());
	}

	Dtype* dst_data = dst.mutable_gpu_data();
	const Dtype* src_data = src.gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(dst_width, dst_height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_resize << <grid_size, block_size >> > (channels, src_data, height, width, dst_data, dst_height, dst_width, type, src.type());
}






/// <summary>
/// 图像绕中心点旋转某个角度，旋转后图像的宽度和高度随之改变
/// </summary>
/// <param name="channels">图像通道数</param>
/// <param name="src_data">原图像数据</param>
/// <param name="height">原图像高度</param>
/// <param name="width">原图像宽度</param>
/// <param name="dst_data">旋转变换后的新图像数据</param>
/// <param name="dst_height">新图像高度</param>
/// <param name="dst_width">新图像宽度</param>
/// <param name="sina">根据旋转角theta计算得到的正弦值</param>
/// <param name="cosa">根据旋转角theta计算得到的余弦值</param>
/// <param name="varX">X方向的偏置量</param>
/// <param name="varY">Y方向的偏置量</param>
/// <param name="fill">空白区域填充值</param>
/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值）</param>
/// <param name="Ttype">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
template<typename Dtype>
__global__
void kernel_rotate_with_center(int channels, const Dtype* src_data, int height, int width,
	Dtype* dst_data, int dst_height, int dst_width,
	float sina, float cosa, float varX, float varY, int fill, interpolationType type, tensorType Ttype)
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



/// <summary>
/// 图像绕中心点旋转某个角度，旋转后图像的宽度和高度随之改变
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">旋转变换后的tensor</param>
/// <param name="theta">旋转角度，逆时针方向为正</param>
/// <param name="dst_height">旋转后新的图像高度</param>
/// <param name="dst_width">旋转后新的图像宽度</param>
/// <param name="fill">空白区域填充值，默认为0</param>
/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值，默认）</param>
template<typename Dtype>
void tensor_operation_gpu::rotate_with_center_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst,
	float theta, int &dst_height, int &dst_width, int fill = 0, interpolationType type = Bilinear)
{
	if (fabs(theta) <= 1e-6)
	{
		LOG(WARNING) << "Just copy from the source.";
		dst = std::make_shared<excalibur::tensor<Dtype>>(src->clone());
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
		dst.reset(new excalibur::tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->type()));
	}
	else
	{
		dst.reset(new excalibur::tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->type()));
	}

	Dtype* dst_data = dst->mutable_gpu_data();
	const Dtype* src_data = src->gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(dst_width, dst_height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_rotate_with_center << <grid_size, block_size >> > (channels, src_data, height, width, dst_data, dst_height, dst_width, sina, cosa, VarX, VarY, fill, type, src->type());
}



/// <summary>
/// 图像绕中心点旋转某个角度，旋转后图像的宽度和高度随之改变
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">旋转变换后的tensor</param>
/// <param name="theta">旋转角度，逆时针方向为正</param>
/// <param name="dst_height">旋转后新的图像高度</param>
/// <param name="dst_width">旋转后新的图像宽度</param>
/// <param name="fill">空白区域填充值，默认为0</param>
/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值，默认）</param>
template<typename Dtype>
void tensor_operation_gpu::rotate_with_center_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst,
	float theta, int &dst_height, int &dst_width, int fill = 0, interpolationType type = Bilinear)
{
	if (fabs(theta) <= 1e-6)
	{
		LOG(WARNING) << "Just copy from the source.";
		dst = src.clone();
	}

	CHECK_EQ(src.num(), 1);
	int channels = src.channels();
	int height = src.height();
	int width = src.width();

	float rad = -1 * theta*(PI / 180);//逆时针为正
	float cosa = cos(rad);
	float sina = sin(rad);

	dst_width = (int)(width * abs(cosa) + height * abs(sina));
	dst_height = (int)(width * abs(sina) + height * abs(cosa));

	float VarX = (float)(-dst_width * cosa / 2.0f - dst_height * sina / 2.0f + width / 2.0f);
	float VarY = (float)(dst_width * sina / 2.0f - dst_height * cosa / 2.0f + height / 2.0f);

	if (src.type() == NCHW)
	{
		dst = excalibur::tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src.device(), src.type());
	}
	else
	{
		dst = excalibur::tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src.device(), src.type());
	}

	Dtype* dst_data = dst.mutable_gpu_data();
	const Dtype* src_data = src.gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(dst_width, dst_height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_rotate_with_center << <grid_size, block_size >> > (channels, src_data, height, width, dst_data, dst_height, dst_width, sina, cosa, VarX, VarY, fill, type, src.type());
}






/// <summary>
/// 图像绕任意点旋转某个角度，旋转后图像的宽度和高度不变
/// </summary>
/// <param name="src_data">原图像数据</param>
/// <param name="dst_data">旋转变化后的新图像数据</param>
/// <param name="channels">图像通道数</param>
/// <param name="height">图像高度</param>
/// <param name="width">图像宽度</param>
/// <param name="M_data">变换矩阵中的各参数</param>
/// <param name="fill">空白区域填充值</param>
/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值）</param>
/// <param name="Ttype">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
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



/// <summary>
/// 图像绕任意点旋转某个角度，旋转后图像的宽度和高度不变
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">旋转变换后的tensor</param>
/// <param name="center">指定的旋转中心</param>
/// <param name="theta">旋转角度，逆时针方向为正</param>
/// <param name="scale">图像缩放比例，默认为1，即保持不变</param>
/// <param name="fill">空白区域填充值，默认为0</param>
/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值，默认）</param>
template<typename Dtype, typename Ptype>
void tensor_operation_gpu::rotate_with_points_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst,
	const point<Ptype> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear)
{
	if (fabs(theta) <= 1e-6)
	{
		LOG(WARNING) << "Just copy from the source.";
		dst = std::make_shared<tensor<Dtype>>(src->clone());
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
		dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->type()));
	}
	else
	{
		dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->type()));
	}

	const Dtype* src_data = src->gpu_data();
	Dtype* dst_data = dst->mutable_gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_rotate_with_points << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, reverse_M_data, fill, type, src->type());
}



/// <summary>
/// 图像绕任意点旋转某个角度，旋转后图像的宽度和高度不变
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">旋转变换后的tensor</param>
/// <param name="center">指定的旋转中心</param>
/// <param name="theta">旋转角度，逆时针方向为正</param>
/// <param name="scale">图像缩放比例，默认为1，即保持不变</param>
/// <param name="fill">空白区域填充值，默认为0</param>
/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值，默认）</param>
template<typename Dtype, typename Ptype>
void tensor_operation_gpu::rotate_with_points_gpu(const tensor<Dtype> &src, tensor<Dtype> &dst,
	const point<Ptype> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear)
{
	if (fabs(theta) <= 1e-6)
	{
		LOG(WARNING) << "Just copy from the source.";
		dst = src.clone();
		return;
	}

	CHECK_EQ(src.num(), 1);
	int channels = src.channels();
	int height = src.height();
	int width = src.width();
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

	if (src.type() == NCHW)
	{
		dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.type());
	}
	else
	{
		dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.type());
	}

	const Dtype* src_data = src.gpu_data();
	Dtype* dst_data = dst.mutable_gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_rotate_with_points << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, reverse_M_data, fill, type, src.type());
}






/// <summary>
/// 在图像上完成四种翻转：仅宽度方向翻转、仅高度方向翻转、宽度和高度同时翻转、按图像通道翻转
/// </summary>
/// <param name="src_data">原图像数据</param>
/// <param name="dst_data">翻转后的新图像数据</param>
/// <param name="channels">图像通道数</param>
/// <param name="height">图像高度</param>
/// <param name="width">图像宽度</param>
/// <param name="axis">翻转轴：Width_Wise（仅宽度方向翻转）、Height_Wise（仅高度方向翻转）、Center_Wise（宽度和高度同时翻转）、Channel_Wise（按图像通道翻转）</param>
/// <param name="Ttype">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
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



/// <summary>
/// 在图像上完成四种翻转：仅宽度方向翻转、仅高度方向翻转、宽度和高度同时翻转、按图像通道翻转
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">翻转变换后的tensor</param>
/// <param name="axis">翻转轴：Width_Wise（仅宽度方向翻转，默认）、Height_Wise（仅高度方向翻转）、Center_Wise（宽度和高度同时翻转）、Channel_Wise（按图像通道翻转）</param>
template<typename Dtype>
void tensor_operation_gpu::flip_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst, flipType axis = Width_Wise)
{
	CHECK_EQ(src->num(), 1);
	int channels = src->channels();
	int height = src->height();
	int width = src->width();

	if (src->type() == NCHW)
	{
		dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->type()));
	}
	else
	{
		dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->type()));
	}

	const Dtype* src_data = src->gpu_data();
	Dtype* dst_data = dst->mutable_gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_flip << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, axis, src->type());
}



/// <summary>
/// 在图像上完成四种翻转：仅宽度方向翻转、仅高度方向翻转、宽度和高度同时翻转、按图像通道翻转
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">翻转变换后的tensor</param>
/// <param name="axis">翻转轴：Width_Wise（仅宽度方向翻转，默认）、Height_Wise（仅高度方向翻转）、Center_Wise（宽度和高度同时翻转）、Channel_Wise（按图像通道翻转）</param>
template<typename Dtype>
void tensor_operation_gpu::flip_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst, flipType axis = Width_Wise)
{
	CHECK_EQ(src.num(), 1);
	int channels = src.channels();
	int height = src.height();
	int width = src.width();

	if (src.type() == NCHW)
	{
		dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.type());
	}
	else
	{
		dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.type());
	}

	const Dtype* src_data = src.gpu_data();
	Dtype* dst_data = dst.mutable_gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_flip << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, axis, src.type());
}






/// <summary>
/// 三通道rgb图像转换为单通道灰度图像
/// </summary>
/// <param name="src_data">原图像数据</param>
/// <param name="dst_data">新图像数据</param>
/// <param name="height">图像高度</param>
/// <param name="width">图像宽度</param>
/// <param name="Ttype">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
template<typename Dtype>
__global__
void kernel_rgb2gray(const Dtype* src_data, Dtype* dst_data, int height, int width, tensorType Ttype)
{
	int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	int rowID = totalID / width;
	int colID = totalID % width;

	int index = rowID * width + colID;
	int offset = height * width;

	//opencv读取RGB图像后，以B、G、R的顺序进行存储
	//转换公式为:gray=0.114*B+0.587*G+0.299*R
	if (Ttype == NCHW)
	{
		dst_data[index] = Dtype(static_cast<float>(src_data[index]) * 0.114f +
			static_cast<float>(src_data[offset * 1 + index]) * 0.587f +
			static_cast<float>(src_data[offset * 2 + index]) * 0.299f);
	}
	else if (Ttype == NHWC)
	{
		dst_data[index] = Dtype(static_cast<float>(src_data[3 * index]) * 0.114f +
			static_cast<float>(src_data[3 * index + 1]) * 0.587f +
			static_cast<float>(src_data[3 * index + 2]) * 0.299f);
	}
}



/// <summary>
/// 三通道rgb图像转换为单通道灰度图像
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">转换后的tensor</param>
template<typename Dtype>
void tensor_operation_gpu::rgb2gray_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst)
{
	CHECK_EQ(src->num(), 1);
	int channels = src->channels();
	int height = src->height();
	int width = src->width();

	if (channels != 3)
	{
		LOG(ERROR) << "Incorrect input channel.";
		return;
	}

	if (src->type() == NCHW)
	{
		dst.reset(new tensor<Dtype>(std::vector<int>{1, 1, height, width}, src->device(), src->type()));
	}
	else
	{
		dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, 1}, src->device(), src->type()));
	}

	const Dtype* src_data = src->gpu_data();
	Dtype* dst_data = dst->mutable_gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_rgb2gray << <grid_size, block_size >> > (src_data, dst_data, height, width, src->type());
}



/// <summary>
/// 三通道rgb图像转换为单通道灰度图像
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">转换后的tensor</param>
template<typename Dtype>
void tensor_operation_gpu::rgb2gray_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst)
{
	CHECK_EQ(src.num(), 1);
	int channels = src.channels();
	int height = src.height();
	int width = src.width();

	if (channels != 3)
	{
		LOG(ERROR) << "Incorrect input channel.";
		return;
	}

	if (src.type() == NCHW)
	{
		dst = tensor<Dtype>(std::vector<int>{1, 1, height, width}, src.device(), src.type());
	}
	else
	{
		dst = tensor<Dtype>(std::vector<int>{1, height, width, 1}, src.device(), src.type());
	}

	const Dtype* src_data = src.gpu_data();
	Dtype* dst_data = dst.mutable_gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_rgb2gray << <grid_size, block_size >> > (src_data, dst_data, height, width, src.type());
}






/// <summary>
/// 矩阵转置
/// </summary>
/// <param name="src_data">原图像数据</param>
/// <param name="dst_data">转置后的新图像数据</param>
/// <param name="channels">图像通道数</param>
/// <param name="height">图像高度</param>
/// <param name="width">图像宽度</param>
/// <param name="Ttype">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
template<typename Dtype>
__global__
void kernel_matrix_transpose(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, tensorType Ttype)
{
	int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
	int rowID = totalID / width;
	int colID = totalID % width;

	if (Ttype == NCHW)
	{
		int offset = height * width;

		int dst_index = rowID * width + colID;
		int src_index = colID * height + rowID;

		for (int ch = 0; ch < channels; ++ch) {
			int channel_offset = ch * offset;
			dst_data[dst_index + channel_offset] = src_data[src_index + channel_offset];
		}
	}
	else if (Ttype == NHWC)
	{
		for (int ch = 0; ch < channels; ++ch)
		{
			int dst_index = (rowID * width + colID) * channels + ch;
			int src_index = (colID * height + rowID) * channels + ch;
			dst_data[dst_index] = src_data[src_index];
		}
	}
}



/// <summary>
/// 矩阵转置
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">转换后的tensor</param>
template<typename Dtype>
void tensor_operation_gpu::matrix_transpose_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst)
{
	CHECK_EQ(src->num(), 1);
	int channels = src->channels();
	int height = src->height();
	int width = src->width();
	int offset = height * width;

	if (src->type() == NCHW)
	{
		dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, width, height}, src->device(), src->type()));
	}
	else
	{
		dst.reset(new tensor<Dtype>(std::vector<int>{1, width, height, channels}, src->device(), src->type()));
	}

	const Dtype* src_data = src->gpu_data();
	Dtype* dst_data = dst->mutable_gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(height, width, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_matrix_transpose << <grid_size, block_size >> > (src_data, dst_data, channels, width, height, src->type());
}



/// <summary>
/// 矩阵转置
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">转换后的tensor</param>
template<typename Dtype>
void tensor_operation_gpu::matrix_transpose_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst)
{
	CHECK_EQ(src.num(), 1);
	int channels = src.channels();
	int height = src.height();
	int width = src.width();
	int offset = height * width;

	if (src.type() == NCHW)
	{
		dst = tensor<Dtype>(std::vector<int>{1, channels, width, height}, src.device(), src.type());
	}
	else
	{
		dst = tensor<Dtype>(std::vector<int>{1, width, height, channels}, src.device(), src.type());
	}

	const Dtype* src_data = src.gpu_data();
	Dtype* dst_data = dst.mutable_gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(height, width, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_matrix_transpose << <grid_size, block_size >> > (src_data, dst_data, channels, width, height, src.type());
}






/// <summary>
/// 从图像中获取感兴趣区域
/// </summary>
/// <param name="src_data">原图像数据</param>
/// <param name="channels">图像通道数</param>
/// <param name="src_height">原图像高度</param>
/// <param name="src_width">原图像宽度</param>
/// <param name="dst_data">截取出的新图像数据</param>
/// <param name="rect">矩形的感兴趣区域</param>
/// <param name="Ttype">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
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



/// <summary>
/// 从图像中获取感兴趣区域
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">包含感兴趣区域图像数据的tensor</param>
/// <param name="rect">矩形的感兴趣区域</param>
template<typename Dtype, typename Rtype>
void tensor_operation_gpu::roi_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst, excalibur::rectangle<Rtype> rect)
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
		dst = std::make_shared<tensor<Dtype>>(src->clone());
		return;
	}

	if (src->type() == NCHW)
	{
		dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->type()));
	}
	else
	{
		dst.reset(new tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->type()));
	}

	Dtype* dst_data = dst->mutable_gpu_data();
	const Dtype* src_data = src->gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(dst_width, dst_height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_ROI << <grid_size, block_size >> > (src_data, channels, height, width, dst_data, rect, src->type());
}



/// <summary>
/// 从图像中获取感兴趣区域
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">包含感兴趣区域图像数据的tensor</param>
/// <param name="rect">矩形的感兴趣区域</param>
template<typename Dtype, typename Rtype>
void tensor_operation_gpu::roi_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst, excalibur::rectangle<Rtype> rect)
{
	CHECK_EQ(src.num(), 1);
	int channels = src.channels();
	int height = src.height();
	int width = src.width();
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
		dst = src.clone();
		return;
	}

	if (src.type() == NCHW)
	{
		dst = tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src.device(), src.type());
	}
	else
	{
		dst = tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src.device(), src.type());
	}

	Dtype* dst_data = dst.mutable_gpu_data();
	const Dtype* src_data = src.gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(dst_width, dst_height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_ROI << <grid_size, block_size >> > (src_data, channels, height, width, dst_data, rect, src.type());
}






/// <summary>
/// 二值化灰度图像
/// </summary>
/// <param name="src_data">原图像数据</param>
/// <param name="dst_data">二值化后的新图像数据</param>
/// <param name="src_height">图像高度</param>
/// <param name="src_width">图像宽度</param>
/// <param name="thresh">阈值</param>
/// <param name="maxval">最大值</param>
/// <param name="type">二值化类型：binary（灰度值大于thresh的像素点，将灰度值设为maxval，反之设为0，默认值）、
///                           binary_inv（灰度值小于thresh的像素点，将灰度值设为maxval，反之设为0）、
///                          small_trunc（灰度值小于thresh的像素点，将灰度值设为thresh，其余点保持原有灰度值不变）、
///                            big_trunc（灰度值大于thresh的像素点，将灰度值设为thresh，其余点保持原有灰度值不变）、
///                        small_to_zero（灰度值小于thresh的像素点，将灰度值设为0，其余点保持原有灰度值不变）、
///                          big_to_zero（灰度值大于thresh的像素点，将灰度值设为0，其余点保持原有灰度值不变）</param>
template<typename Dtype>
__global__
void kernel_threshold(const Dtype* src_data, Dtype* dst_data, int src_height, int src_width, int thresh, int maxval, thresholdType type)
{
	int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

	switch (type)
	{
	case binary:
		dst_data[totalID] = src_data[totalID] > (Dtype)thresh ? (Dtype)maxval : (Dtype)0;
		break;

	case binary_inv:
		dst_data[totalID] = src_data[totalID] <= (Dtype)thresh ? (Dtype)maxval : (Dtype)0;
		break;

	case big_trunc:
		if (src_data[totalID] > (Dtype)thresh)
		{
			dst_data[totalID] = (Dtype)thresh;
		}
		else
		{
			dst_data[totalID] = src_data[totalID];
		}
		break;

	case small_trunc:
		if (src_data[totalID] > (Dtype)thresh)
		{
			dst_data[totalID] = src_data[totalID];
		}
		else
		{
			dst_data[totalID] = (Dtype)thresh;
		}
		break;

	case small_to_zero:
		dst_data[totalID] = src_data[totalID] > (Dtype)thresh ? src_data[totalID] : (Dtype)0;
		break;

	case big_to_zero:
		dst_data[totalID] = src_data[totalID] <= (Dtype)thresh ? src_data[totalID] : (Dtype)0;
		break;

	default:
		break;
	}
}



/// <summary>
/// 二值化灰度图像
/// </summary>
/// <param name="src">包含原图像数据的tensor，单通道灰度图</param>
/// <param name="dst">二值化后的tensor</param>
/// <param name="thresh">阈值，默认为128</param>
/// <param name="maxval">最大值，默认为255</param>
/// <param name="type">二值化类型：binary（灰度值大于thresh的像素点，将灰度值设为maxval，反之设为0，默认值）、
///                           binary_inv（灰度值小于thresh的像素点，将灰度值设为maxval，反之设为0）、
///                          small_trunc（灰度值小于thresh的像素点，将灰度值设为thresh，其余点保持原有灰度值不变）、
///                            big_trunc（灰度值大于thresh的像素点，将灰度值设为thresh，其余点保持原有灰度值不变）、
///                        small_to_zero（灰度值小于thresh的像素点，将灰度值设为0，其余点保持原有灰度值不变）、
///                          big_to_zero（灰度值大于thresh的像素点，将灰度值设为0，其余点保持原有灰度值不变）</param>
template<typename Dtype>
void tensor_operation_gpu::threshold_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary)
{
	CHECK_EQ(src->num(), 1);
	CHECK_EQ(src->channels(), 1);
	int height = src->height();
	int width = src->width();
	int src_offset = height * width;

	if (src->type() == NCHW)
	{
		dst.reset(new tensor<Dtype>(std::vector<int>{1, 1, height, width}, src->device(), src->type()));
	}
	else
	{
		dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, 1}, src->device(), src->type()));
	}

	Dtype* dst_data = dst->mutable_gpu_data();
	const Dtype *src_data = src->gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_threshold<Dtype> << <grid_size, block_size >> > (src_data, dst_data, height, width, thresh, maxval, type);
}



/// <summary>
/// 二值化灰度图像
/// </summary>
/// <param name="src">包含原图像数据的tensor，单通道灰度图</param>
/// <param name="dst">二值化后的tensor</param>
/// <param name="thresh">阈值，默认为128</param>
/// <param name="maxval">最大值，默认为255</param>
/// <param name="type">二值化类型：binary（灰度值大于thresh的像素点，将灰度值设为maxval，反之设为0，默认值）、
///                           binary_inv（灰度值小于thresh的像素点，将灰度值设为maxval，反之设为0）、
///                          small_trunc（灰度值小于thresh的像素点，将灰度值设为thresh，其余点保持原有灰度值不变）、
///                            big_trunc（灰度值大于thresh的像素点，将灰度值设为thresh，其余点保持原有灰度值不变）、
///                        small_to_zero（灰度值小于thresh的像素点，将灰度值设为0，其余点保持原有灰度值不变）、
///                          big_to_zero（灰度值大于thresh的像素点，将灰度值设为0，其余点保持原有灰度值不变）</param>
template<typename Dtype>
void tensor_operation_gpu::threshold_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary)
{
	CHECK_EQ(src.num(), 1);
	CHECK_EQ(src.channels(), 1);
	int height = src.height();
	int width = src.width();
	int src_offset = height * width;

	if (src.type() == NCHW)
	{
		dst = tensor<Dtype>(std::vector<int>{1, 1, height, width}, src.device(), src.type());
	}
	else
	{
		dst = tensor<Dtype>(std::vector<int>{1, height, width, 1}, src.device(), src.type());
	}

	Dtype* dst_data = dst.mutable_gpu_data();
	const Dtype *src_data = src.gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_threshold<Dtype> << <grid_size, block_size >> > (src_data, dst_data, height, width, thresh, maxval, type);
}






/// <summary>
/// 仿射变换
/// </summary>
/// <param name="src_data">原图像数据</param>
/// <param name="dst_data">仿射变换后的新图像数据</param>
/// <param name="channels">图像通道数</param>
/// <param name="height">图像高度</param>
/// <param name="width">图像宽度</param>
/// <param name="M_data">变换矩阵中的各参数</param>
/// <param name="fill">空白区域的填充值</param>
/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值）</param>
/// <param name="Ttype">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
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



/// <summary>
/// 仿射变换
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">变换后的tensor</param>
/// <param name="src_point">变换前的3个点</param>
/// <param name="dst_point">变换后对应的3个点，可通过src_point和dst_point计算得到变换矩阵，进而使用该变换矩阵对整张图像完成仿射变换</param>
/// <param name="fill">空白区域的填充值，默认为0</param>
/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值，默认）</param>
template<typename Dtype, typename Ptype>
void tensor_operation_gpu::warp_affine_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst,
	const std::vector<point<Ptype>> &src_point, const std::vector<point<Ptype>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear)
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
		dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->type()));
	}
	else
	{
		dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->type()));
	}

	const Dtype* src_data = src->gpu_data();
	Dtype* dst_data = dst->mutable_gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_warp_affine << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, M_data, fill, type, src->type());
}



/// <summary>
/// 仿射变换
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">变换后的tensor</param>
/// <param name="src_point">变换前的3个点</param>
/// <param name="dst_point">变换后对应的3个点，可通过src_point和dst_point计算得到变换矩阵，进而使用该变换矩阵对整张图像完成仿射变换</param>
/// <param name="fill">空白区域的填充值，默认为0</param>
/// <param name="type">插值类型：Nearest（最近邻插值）、Bilinear（双线性插值，默认）</param>
template<typename Dtype, typename Ptype>
void tensor_operation_gpu::warp_affine_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst,
	const std::vector<point<Ptype>> &src_point, const std::vector<point<Ptype>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear)
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

	CHECK_EQ(src.num(), 1);
	int channels = src.channels();
	int height = src.height();
	int width = src.width();

	if (src.type() == NCHW)
	{
		dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.type());
	}
	else
	{
		dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.type());
	}

	const Dtype* src_data = src.gpu_data();
	Dtype* dst_data = dst.mutable_gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_warp_affine << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, M_data, fill, type, src.type());
}






/// <summary>
/// 高斯滤波
/// </summary>
/// <param name="src_data">原图像数据</param>
/// <param name="dst_data">高斯滤波后的新图像数据</param>
/// <param name="channels">图像通道数</param>
/// <param name="height">图像高度</param>
/// <param name="width">图像宽度</param>
/// <param name="ksize">高斯核的尺寸，只能为奇数。实际使用的高斯模板规格为ksize×ksize的矩形窗</param>
/// <param name="paras">一维高斯模板中的各权重值（使用了可分离滤波）</param>
/// <param name="Ttype">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
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



/// <summary>
/// 高斯滤波
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">高斯滤波后的tensor</param>
/// <param name="ksize">高斯核的尺寸，只能为奇数，默认为3。实际使用的高斯模板规格为ksize×ksize的矩形窗</param>
template<typename Dtype>
void tensor_operation_gpu::gaussian_blur_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, int ksize = 3)
{
	if (ksize % 2 != 1)
	{
		LOG(WARNING) << "convolution kernel: width and height should be odd.";
		return;
	}

	if (ksize == 1)
	{
		LOG(WARNING) << "Just copy from the source.";
		dst = std::make_shared<tensor<Dtype>>(src->clone());
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
		dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->type()));
	}
	else
	{
		dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->type()));
	}

	const Dtype* src_data = src->gpu_data();
	Dtype* dst_data = dst->mutable_gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_gaussian_blur << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, ksize, paras, src->type());
}



/// <summary>
/// 高斯滤波
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">高斯滤波后的tensor</param>
/// <param name="ksize">高斯核的尺寸，只能为奇数，默认为3。实际使用的高斯模板规格为ksize×ksize的矩形窗</param>
template<typename Dtype>
void tensor_operation_gpu::gaussian_blur_gpu(const tensor<Dtype> &src, tensor<Dtype> &dst, int ksize = 3)
{
	if (ksize % 2 != 1)
	{
		LOG(WARNING) << "convolution kernel: width and height should be odd.";
		return;
	}

	if (ksize == 1)
	{
		LOG(WARNING) << "Just copy from the source.";
		dst = src.clone();
		return;
	}

	CHECK_EQ(src.num(), 1);
	int channels = src.channels();
	int height = src.height();
	int width = src.width();

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

	if (src.type() == NCHW)
	{
		dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.type());
	}
	else
	{
		dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.type());
	}

	const Dtype* src_data = src.gpu_data();
	Dtype* dst_data = dst.mutable_gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_gaussian_blur << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, ksize, paras, src.type());
}






/// <summary>
/// 计算水平、竖直方向的梯度边缘
/// </summary>
/// <param name="src_data">原图像数据</param>
/// <param name="dst_data">新图像数据</param>
/// <param name="channels">图像通道数</param>
/// <param name="height">图像高度</param>
/// <param name="width">图像宽度</param>
/// <param name="dx">dx>0时，沿着x方向求导，得到竖直边缘；dx小于等于0时，不计算竖直边缘。</param>
/// <param name="dy">dy>0时，沿着y方向求导，得到水平边缘；dy小于等于0时，不计算水平边缘。</param>
/// <param name="Ttype">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
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
				total = 0.25 * total;
			}
			else
			{
				total = 0.5 * total;
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
				total = 0.25 * total;
			}
			else
			{
				total = 0.5 * total;
			}

			if (total > 255)
			{
				total = 255;
			}

			dst_data[pos] = (Dtype)total;
		}
	}
}



/// <summary>
/// 计算水平、竖直方向的梯度边缘
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">计算后的tensor</param>
/// <param name="dx">dx>0时，沿着x方向求导，得到竖直边缘；dx小于等于0时，不计算竖直边缘。默认为1</param>
/// <param name="dy">dy>0时，沿着y方向求导，得到水平边缘；dy小于等于0时，不计算水平边缘。默认为1</param>
template<typename Dtype>
void tensor_operation_gpu::sobel_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, int dx = 1, int dy = 1)
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
		dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->type()));
	}
	else
	{
		dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->type()));
	}

	Dtype* dst_data = dst->mutable_gpu_data();
	const Dtype* src_data = src->gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_sobel << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, dx, dy, src->type());
}



/// <summary>
/// 计算水平、竖直方向的梯度边缘
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">计算后的tensor</param>
/// <param name="dx">dx>0时，沿着x方向求导，得到竖直边缘；dx小于等于0时，不计算竖直边缘。默认为1</param>
/// <param name="dy">dy>0时，沿着y方向求导，得到水平边缘；dy小于等于0时，不计算水平边缘。默认为1</param>
template<typename Dtype>
void tensor_operation_gpu::sobel_gpu(const tensor<Dtype> &src, tensor<Dtype> &dst, int dx = 1, int dy = 1)
{
	if (dx == 0 && dy == 0)
	{
		LOG(WARNING) << "dx, dy cannot be zero simultaneously.";
		return;
	}

	CHECK_EQ(src.num(), 1);
	int channels = src.channels();
	int height = src.height();
	int width = src.width();

	if (src.type() == NCHW)
	{
		dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.type());
	}
	else
	{
		dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.type());
	}

	Dtype* dst_data = dst.mutable_gpu_data();
	const Dtype* src_data = src.gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_sobel << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, dx, dy, src.type());
}






/// <summary>
/// 使用ksize×ksize的矩形窗对图像进行膨胀或腐蚀
/// </summary>
/// <param name="src_data">原图像数据</param>
/// <param name="dst_data">膨胀或腐蚀后的新图像数据</param>
/// <param name="channels">图像通道数</param>
/// <param name="height">图像高度</param>
/// <param name="width">图像宽度</param>
/// <param name="type">形态学类型：Dilate（膨胀）、Erode（腐蚀）</param>
/// <param name="ksize">用于膨胀或腐蚀的结构元尺寸，实际使用的是ksize×ksize的矩形窗，ksize应为奇数</param>
/// <param name="Ttype">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
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



/// <summary>
/// 对图像进行膨胀、腐蚀处理
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">膨胀、腐蚀后的tensor</param>
/// <param name="type">形态学类型：Dilate（膨胀，默认值）、Erode（腐蚀）</param>
/// <param name="ksize">矩形核的尺寸，只能为奇数，默认为3。实际使用的矩形规格为ksize×ksize</param>
template<typename Dtype>
void tensor_operation_gpu::morph_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, excalibur::morphType type = Dilate, int ksize = 3)
{
	if (ksize % 2 != 1)
	{
		LOG(WARNING) << "ksize should be odd.";
		return;
	}

	if (ksize == 1)
	{
		dst = std::make_shared<tensor<Dtype>>(src->clone());
		LOG(WARNING) << "Just copy from the source.";
		return;
	}

	CHECK_EQ(src->num(), 1);
	int channels = src->channels();
	int height = src->height();
	int width = src->width();

	if (src->type() == NCHW)
	{
		dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->type()));
	}
	else
	{
		dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->type()));
	}

	Dtype* dst_data = dst->mutable_gpu_data();
	const Dtype* src_data = src->gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_morph << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, type, ksize, src->type());
}



/// <summary>
/// 对图像进行膨胀、腐蚀处理
/// </summary>
/// <param name="src">包含原图像数据的tensor</param>
/// <param name="dst">膨胀、腐蚀后的tensor</param>
/// <param name="type">形态学类型：Dilate（膨胀，默认值）、Erode（腐蚀）</param>
/// <param name="ksize">矩形核的尺寸，只能为奇数，默认为3。实际使用的矩形规格为ksize×ksize</param>
template<typename Dtype>
void tensor_operation_gpu::morph_gpu(const tensor<Dtype> &src, tensor<Dtype> &dst, excalibur::morphType type = Dilate, int ksize = 3)
{
	if (ksize % 2 != 1)
	{
		LOG(WARNING) << "ksize should be odd.";
		return;
	}

	if (ksize == 1)
	{
		dst = src.clone();
		LOG(WARNING) << "Just copy from the source.";
		return;
	}

	CHECK_EQ(src.num(), 1);
	int channels = src.channels();
	int height = src.height();
	int width = src.width();

	if (src.type() == NCHW)
	{
		dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.type());
	}
	else
	{
		dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.type());
	}

	Dtype* dst_data = dst.mutable_gpu_data();
	const Dtype* src_data = src.gpu_data();

	const dim3 block_size(1, 1, 1);
	const dim3 grid_size(width, height, 1);

	//按照设置的blockSize和gridSize启动内核函数
	kernel_morph << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, type, ksize, src.type());
}









template void tensor_operation_gpu::tensor2mat_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src, cv::Mat& dst);
template void tensor_operation_gpu::tensor2mat_gpu<char>(const std::shared_ptr<tensor<char>> &src, cv::Mat& dst);
template void tensor_operation_gpu::tensor2mat_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src, cv::Mat& dst);
template void tensor_operation_gpu::tensor2mat_gpu<int>(const std::shared_ptr<tensor<int>> &src, cv::Mat& dst);
template void tensor_operation_gpu::tensor2mat_gpu<float>(const std::shared_ptr<tensor<float>> &src, cv::Mat& dst);



template void tensor_operation_gpu::tensor2mat_gpu<unsigned char>(const tensor<unsigned char>& src, cv::Mat& dst);
template void tensor_operation_gpu::tensor2mat_gpu<char>(const tensor<char>& src, cv::Mat& dst);
template void tensor_operation_gpu::tensor2mat_gpu<unsigned int>(const tensor<unsigned int>& src, cv::Mat& dst);
template void tensor_operation_gpu::tensor2mat_gpu<int>(const tensor<int>& src, cv::Mat& dst);
template void tensor_operation_gpu::tensor2mat_gpu<float>(const tensor<float>& src, cv::Mat& dst);



template void tensor_operation_gpu::mat2tensor_gpu<unsigned char>(const cv::Mat &src, std::shared_ptr<tensor<unsigned char>>& dst, tensorType Ttype);
template void tensor_operation_gpu::mat2tensor_gpu<char>(const cv::Mat &src, std::shared_ptr<tensor<char>>& dst, tensorType Ttype);
template void tensor_operation_gpu::mat2tensor_gpu<unsigned int>(const cv::Mat &src, std::shared_ptr<tensor<unsigned int>>& dst, tensorType Ttype);
template void tensor_operation_gpu::mat2tensor_gpu<int>(const cv::Mat &src, std::shared_ptr<tensor<int>>& dst, tensorType Ttype);
template void tensor_operation_gpu::mat2tensor_gpu<float>(const cv::Mat &src, std::shared_ptr<tensor<float>>& dst, tensorType Ttype);



template void tensor_operation_gpu::mat2tensor_gpu<unsigned char>(const cv::Mat &src, tensor<unsigned char>& dst, tensorType Ttype = NHWC);
template void tensor_operation_gpu::mat2tensor_gpu<char>(const cv::Mat &src, tensor<char>& dst, tensorType Ttype = NHWC);
template void tensor_operation_gpu::mat2tensor_gpu<unsigned int>(const cv::Mat &src, tensor<unsigned int>& dst, tensorType Ttype = NHWC);
template void tensor_operation_gpu::mat2tensor_gpu<int>(const cv::Mat &src, tensor<int>& dst, tensorType Ttype = NHWC);
template void tensor_operation_gpu::mat2tensor_gpu<float>(const cv::Mat &src, tensor<float>& dst, tensorType Ttype = NHWC);



template void tensor_operation_gpu::nchw2nhwc_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>> &dst);
template void tensor_operation_gpu::nchw2nhwc_gpu<char>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>> &dst);
template void tensor_operation_gpu::nchw2nhwc_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>> &dst);
template void tensor_operation_gpu::nchw2nhwc_gpu<int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>> &dst);
template void tensor_operation_gpu::nchw2nhwc_gpu<float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>> &dst);



template void tensor_operation_gpu::nchw2nhwc_gpu<unsigned char>(const tensor<unsigned char> &src, tensor<unsigned char> &dst);
template void tensor_operation_gpu::nchw2nhwc_gpu<char>(const tensor<char> &src, tensor<char> &dst);
template void tensor_operation_gpu::nchw2nhwc_gpu<unsigned int>(const tensor<unsigned int> &src, tensor<unsigned int> &dst);
template void tensor_operation_gpu::nchw2nhwc_gpu<int>(const tensor<int> &src, tensor<int> &dst);
template void tensor_operation_gpu::nchw2nhwc_gpu<float>(const tensor<float> &src, tensor<float> &dst);



template void tensor_operation_gpu::nhwc2nchw_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>> &dst);
template void tensor_operation_gpu::nhwc2nchw_gpu<char>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>> &dst);
template void tensor_operation_gpu::nhwc2nchw_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>> &dst);
template void tensor_operation_gpu::nhwc2nchw_gpu<int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>> &dst);
template void tensor_operation_gpu::nhwc2nchw_gpu<float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>> &dst);



template void tensor_operation_gpu::nhwc2nchw_gpu<unsigned char>(const tensor<unsigned char> &src, tensor<unsigned char> &dst);
template void tensor_operation_gpu::nhwc2nchw_gpu<char>(const tensor<char> &src, tensor<char> &dst);
template void tensor_operation_gpu::nhwc2nchw_gpu<unsigned int>(const tensor<unsigned int> &src, tensor<unsigned int> &dst);
template void tensor_operation_gpu::nhwc2nchw_gpu<int>(const tensor<int> &src, tensor<int> &dst);
template void tensor_operation_gpu::nhwc2nchw_gpu<float>(const tensor<float> &src, tensor<float> &dst);



template void tensor_operation_gpu::resize_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst,
	int dst_height, int dst_width, interpolationType type);
template void tensor_operation_gpu::resize_gpu<char>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>>& dst,
	int dst_height, int dst_width, interpolationType type);
template void tensor_operation_gpu::resize_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>>& dst,
	int dst_height, int dst_width, interpolationType type);
template void tensor_operation_gpu::resize_gpu<int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>>& dst,
	int dst_height, int dst_width, interpolationType type);
template void tensor_operation_gpu::resize_gpu<float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>>& dst,
	int dst_height, int dst_width, interpolationType type);



template void tensor_operation_gpu::resize_gpu<unsigned char>(const tensor<unsigned char> &src, tensor<unsigned char>& dst,
	int dst_height, int dst_width, interpolationType type = Bilinear);
template void tensor_operation_gpu::resize_gpu<char>(const tensor<char> &src, tensor<char>& dst,
	int dst_height, int dst_width, interpolationType type = Bilinear);
template void tensor_operation_gpu::resize_gpu<unsigned int>(const tensor<unsigned int> &src, tensor<unsigned int>& dst,
	int dst_height, int dst_width, interpolationType type = Bilinear);
template void tensor_operation_gpu::resize_gpu<int>(const tensor<int> &src, tensor<int>& dst,
	int dst_height, int dst_width, interpolationType type = Bilinear);
template void tensor_operation_gpu::resize_gpu<float>(const tensor<float> &src, tensor<float>& dst,
	int dst_height, int dst_width, interpolationType type = Bilinear);



template void tensor_operation_gpu::rotate_with_center_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst,
	float theta, int &dst_height, int &dst_width, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_center_gpu<char>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>>& dst,
	float theta, int &dst_height, int &dst_width, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_center_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>>& dst,
	float theta, int &dst_height, int &dst_width, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_center_gpu<int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>>& dst,
	float theta, int &dst_height, int &dst_width, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_center_gpu<float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>>& dst,
	float theta, int &dst_height, int &dst_width, int fill = 0, interpolationType type = Bilinear);



template void tensor_operation_gpu::rotate_with_center_gpu<unsigned char>(const tensor<unsigned char> &src, tensor<unsigned char>& dst,
	float theta, int &dst_height, int &dst_width, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_center_gpu<char>(const tensor<char> &src, tensor<char>& dst,
	float theta, int &dst_height, int &dst_width, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_center_gpu<unsigned int>(const tensor<unsigned int> &src, tensor<unsigned int>& dst,
	float theta, int &dst_height, int &dst_width, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_center_gpu<int>(const tensor<int> &src, tensor<int>& dst,
	float theta, int &dst_height, int &dst_width, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_center_gpu<float>(const tensor<float> &src, tensor<float>& dst,
	float theta, int &dst_height, int &dst_width, int fill = 0, interpolationType type = Bilinear);



template void tensor_operation_gpu::rotate_with_points_gpu<unsigned char, int>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>> &dst,
	const point<int> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_points_gpu<char, int>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>> &dst,
	const point<int> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_points_gpu<unsigned int, int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>> &dst,
	const point<int> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_points_gpu<int, int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>> &dst,
	const point<int> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_points_gpu<float, int>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>> &dst,
	const point<int> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);

template void tensor_operation_gpu::rotate_with_points_gpu<unsigned char, float>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>> &dst,
	const point<float> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_points_gpu<char, float>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>> &dst,
	const point<float> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_points_gpu<unsigned int, float>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>> &dst,
	const point<float> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_points_gpu<int, float>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>> &dst,
	const point<float> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_points_gpu<float, float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>> &dst,
	const point<float> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);



template void tensor_operation_gpu::rotate_with_points_gpu<unsigned char, int>(const tensor<unsigned char> &src, tensor<unsigned char> &dst,
	const point<int> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_points_gpu<char, int>(const tensor<char> &src, tensor<char> &dst,
	const point<int> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_points_gpu<unsigned int, int>(const tensor<unsigned int> &src, tensor<unsigned int> &dst,
	const point<int> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_points_gpu<int, int>(const tensor<int> &src, tensor<int> &dst,
	const point<int> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_points_gpu<float, int>(const tensor<float> &src, tensor<float> &dst,
	const point<int> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);

template void tensor_operation_gpu::rotate_with_points_gpu<unsigned char, float>(const tensor<unsigned char> &src, tensor<unsigned char> &dst,
	const point<float> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_points_gpu<char, float>(const tensor<char> &src, tensor<char> &dst,
	const point<float> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_points_gpu<unsigned int, float>(const tensor<unsigned int> &src, tensor<unsigned int> &dst,
	const point<float> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_points_gpu<int, float>(const tensor<int> &src, tensor<int> &dst,
	const point<float> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);
template void tensor_operation_gpu::rotate_with_points_gpu<float, float>(const tensor<float> &src, tensor<float> &dst,
	const point<float> &center, float theta, float scale = 1.0f, int fill = 0, interpolationType type = Bilinear);



template void tensor_operation_gpu::flip_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src,
	std::shared_ptr<tensor<unsigned char>>& dst, flipType axis);
template void tensor_operation_gpu::flip_gpu<char>(const std::shared_ptr<tensor<char>> &src,
	std::shared_ptr<tensor<char>>& dst, flipType axis);
template void tensor_operation_gpu::flip_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src,
	std::shared_ptr<tensor<unsigned int>>& dst, flipType axis);
template void tensor_operation_gpu::flip_gpu<int>(const std::shared_ptr<tensor<int>> &src,
	std::shared_ptr<tensor<int>>& dst, flipType axis);
template void tensor_operation_gpu::flip_gpu<float>(const std::shared_ptr<tensor<float>> &src,
	std::shared_ptr<tensor<float>>& dst, flipType axis);



template void tensor_operation_gpu::flip_gpu<unsigned char>(const tensor<unsigned char> &src, tensor<unsigned char>& dst, flipType axis);
template void tensor_operation_gpu::flip_gpu<char>(const tensor<char> &src, tensor<char>& dst, flipType axis);
template void tensor_operation_gpu::flip_gpu<unsigned int>(const tensor<unsigned int> &src, tensor<unsigned int>& dst, flipType axis);
template void tensor_operation_gpu::flip_gpu<int>(const tensor<int> &src, tensor<int>& dst, flipType axis);
template void tensor_operation_gpu::flip_gpu<float>(const tensor<float> &src, tensor<float>& dst, flipType axis);



template void tensor_operation_gpu::rgb2gray_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst);
template void tensor_operation_gpu::rgb2gray_gpu<char>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>>& dst);
template void tensor_operation_gpu::rgb2gray_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>>& dst);
template void tensor_operation_gpu::rgb2gray_gpu<int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>>& dst);
template void tensor_operation_gpu::rgb2gray_gpu<float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>>& dst);



template void tensor_operation_gpu::rgb2gray_gpu<unsigned char>(const tensor<unsigned char> &src, tensor<unsigned char>& dst);
template void tensor_operation_gpu::rgb2gray_gpu<char>(const tensor<char> &src, tensor<char>& dst);
template void tensor_operation_gpu::rgb2gray_gpu<unsigned int>(const tensor<unsigned int> &src, tensor<unsigned int>& dst);
template void tensor_operation_gpu::rgb2gray_gpu<int>(const tensor<int> &src, tensor<int>& dst);
template void tensor_operation_gpu::rgb2gray_gpu<float>(const tensor<float> &src, tensor<float>& dst);



template void tensor_operation_gpu::matrix_transpose_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst);
template void tensor_operation_gpu::matrix_transpose_gpu<char>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>>& dst);
template void tensor_operation_gpu::matrix_transpose_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>>& dst);
template void tensor_operation_gpu::matrix_transpose_gpu<int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>>& dst);
template void tensor_operation_gpu::matrix_transpose_gpu<float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>>& dst);



template void tensor_operation_gpu::matrix_transpose_gpu<unsigned char>(const tensor<unsigned char> &src, tensor<unsigned char>& dst);
template void tensor_operation_gpu::matrix_transpose_gpu<char>(const tensor<char> &src, tensor<char>& dst);
template void tensor_operation_gpu::matrix_transpose_gpu<unsigned int>(const tensor<unsigned int> &src, tensor<unsigned int>& dst);
template void tensor_operation_gpu::matrix_transpose_gpu<int>(const tensor<int> &src, tensor<int>& dst);
template void tensor_operation_gpu::matrix_transpose_gpu<float>(const tensor<float> &src, tensor<float>& dst);



template void tensor_operation_gpu::roi_gpu<unsigned char, unsigned int>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst, excalibur::rectangle<unsigned int> rect);
template void tensor_operation_gpu::roi_gpu<unsigned char, int>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst, excalibur::rectangle<int> rect);

template void tensor_operation_gpu::roi_gpu<char, unsigned int>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>>& dst, excalibur::rectangle<unsigned int> rect);
template void tensor_operation_gpu::roi_gpu<char, int>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>>& dst, excalibur::rectangle<int> rect);

template void tensor_operation_gpu::roi_gpu<unsigned int, unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>>& dst, excalibur::rectangle<unsigned int> rect);
template void tensor_operation_gpu::roi_gpu<unsigned int, int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>>& dst, excalibur::rectangle<int> rect);

template void tensor_operation_gpu::roi_gpu<int, unsigned int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>>& dst, excalibur::rectangle<unsigned int> rect);
template void tensor_operation_gpu::roi_gpu<int, int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>>& dst, excalibur::rectangle<int> rect);

template void tensor_operation_gpu::roi_gpu<float, unsigned int>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>>& dst, excalibur::rectangle<unsigned int> rect);
template void tensor_operation_gpu::roi_gpu<float, int>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>>& dst, excalibur::rectangle<int> rect);



template void tensor_operation_gpu::roi_gpu<unsigned char, unsigned int>(const tensor<unsigned char> &src, tensor<unsigned char>& dst, excalibur::rectangle<unsigned int> rect);
template void tensor_operation_gpu::roi_gpu<unsigned char, int>(const tensor<unsigned char> &src, tensor<unsigned char>& dst, excalibur::rectangle<int> rect);

template void tensor_operation_gpu::roi_gpu<char, unsigned int>(const tensor<char> &src, tensor<char>& dst, excalibur::rectangle<unsigned int> rect);
template void tensor_operation_gpu::roi_gpu<char, int>(const tensor<char> &src, tensor<char>& dst, excalibur::rectangle<int> rect);

template void tensor_operation_gpu::roi_gpu<unsigned int, unsigned int>(const tensor<unsigned int> &src, tensor<unsigned int>& dst, excalibur::rectangle<unsigned int> rect);
template void tensor_operation_gpu::roi_gpu<unsigned int, int>(const tensor<unsigned int> &src, tensor<unsigned int>& dst, excalibur::rectangle<int> rect);

template void tensor_operation_gpu::roi_gpu<int, unsigned int>(const tensor<int> &src, tensor<int>& dst, excalibur::rectangle<unsigned int> rect);
template void tensor_operation_gpu::roi_gpu<int, int>(const tensor<int> &src, tensor<int>& dst, excalibur::rectangle<int> rect);

template void tensor_operation_gpu::roi_gpu<float, unsigned int>(const tensor<float> &src, tensor<float>& dst, excalibur::rectangle<unsigned int> rect);
template void tensor_operation_gpu::roi_gpu<float, int>(const tensor<float> &src, tensor<float>& dst, excalibur::rectangle<int> rect);



template void tensor_operation_gpu::threshold_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src,
	std::shared_ptr<tensor<unsigned char>>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary);
template void tensor_operation_gpu::threshold_gpu<char>(const std::shared_ptr<tensor<char>> &src,
	std::shared_ptr<tensor<char>>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary);
template void tensor_operation_gpu::threshold_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src,
	std::shared_ptr<tensor<unsigned int>>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary);
template void tensor_operation_gpu::threshold_gpu<int>(const std::shared_ptr<tensor<int>> &src,
	std::shared_ptr<tensor<int>>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary);
template void tensor_operation_gpu::threshold_gpu<float>(const std::shared_ptr<tensor<float>> &src,
	std::shared_ptr<tensor<float>>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary);



template void tensor_operation_gpu::threshold_gpu<unsigned char>(const tensor<unsigned char> &src,
	tensor<unsigned char>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary);
template void tensor_operation_gpu::threshold_gpu<char>(const tensor<char> &src,
	tensor<char>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary);
template void tensor_operation_gpu::threshold_gpu<unsigned int>(const tensor<unsigned int> &src,
	tensor<unsigned int>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary);
template void tensor_operation_gpu::threshold_gpu<int>(const tensor<int> &src,
	tensor<int>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary);
template void tensor_operation_gpu::threshold_gpu<float>(const tensor<float> &src,
	tensor<float>& dst, int thresh = 128, int maxval = 255, thresholdType type = binary);



template void tensor_operation_gpu::warp_affine_gpu<unsigned char, int>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst,
	const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);
template void tensor_operation_gpu::warp_affine_gpu<unsigned char, float>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst,
	const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);

template void tensor_operation_gpu::warp_affine_gpu<char, int>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>>& dst,
	const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);
template void tensor_operation_gpu::warp_affine_gpu<char, float>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>>& dst,
	const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);

template void tensor_operation_gpu::warp_affine_gpu<unsigned int, int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>>& dst,
	const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);
template void tensor_operation_gpu::warp_affine_gpu<unsigned int, float>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>>& dst,
	const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);

template void tensor_operation_gpu::warp_affine_gpu<int, int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>>& dst,
	const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);
template void tensor_operation_gpu::warp_affine_gpu<int, float>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>>& dst,
	const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);

template void tensor_operation_gpu::warp_affine_gpu<float, int>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>>& dst,
	const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);
template void tensor_operation_gpu::warp_affine_gpu<float, float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>>& dst,
	const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);



template void tensor_operation_gpu::warp_affine_gpu<unsigned char, int>(const tensor<unsigned char> &src, tensor<unsigned char>& dst,
	const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);
template void tensor_operation_gpu::warp_affine_gpu<unsigned char, float>(const tensor<unsigned char> &src, tensor<unsigned char>& dst,
	const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);

template void tensor_operation_gpu::warp_affine_gpu<char, int>(const tensor<char> &src, tensor<char>& dst,
	const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);
template void tensor_operation_gpu::warp_affine_gpu<char, float>(const tensor<char> &src, tensor<char>& dst,
	const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);

template void tensor_operation_gpu::warp_affine_gpu<unsigned int, int>(const tensor<unsigned int> &src, tensor<unsigned int>& dst,
	const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);
template void tensor_operation_gpu::warp_affine_gpu<unsigned int, float>(const tensor<unsigned int> &src, tensor<unsigned int>& dst,
	const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);

template void tensor_operation_gpu::warp_affine_gpu<int, int>(const tensor<int> &src, tensor<int>& dst,
	const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);
template void tensor_operation_gpu::warp_affine_gpu<int, float>(const tensor<int> &src, tensor<int>& dst,
	const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);

template void tensor_operation_gpu::warp_affine_gpu<float, int>(const tensor<float> &src, tensor<float>& dst,
	const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);
template void tensor_operation_gpu::warp_affine_gpu<float, float>(const tensor<float> &src, tensor<float>& dst,
	const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill = 0, excalibur::interpolationType type = Bilinear);



template void tensor_operation_gpu::gaussian_blur_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>> &dst, int ksize = 3);
template void tensor_operation_gpu::gaussian_blur_gpu<char>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>> &dst, int ksize = 3);
template void tensor_operation_gpu::gaussian_blur_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>> &dst, int ksize = 3);
template void tensor_operation_gpu::gaussian_blur_gpu<int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>> &dst, int ksize = 3);
template void tensor_operation_gpu::gaussian_blur_gpu<float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>> &dst, int ksize = 3);



template void tensor_operation_gpu::gaussian_blur_gpu<unsigned char>(const tensor<unsigned char> &src, tensor<unsigned char> &dst, int ksize = 3);
template void tensor_operation_gpu::gaussian_blur_gpu<char>(const tensor<char> &src, tensor<char> &dst, int ksize = 3);
template void tensor_operation_gpu::gaussian_blur_gpu<unsigned int>(const tensor<unsigned int> &src, tensor<unsigned int> &dst, int ksize = 3);
template void tensor_operation_gpu::gaussian_blur_gpu<int>(const tensor<int> &src, tensor<int> &dst, int ksize = 3);
template void tensor_operation_gpu::gaussian_blur_gpu<float>(const tensor<float> &src, tensor<float> &dst, int ksize = 3);



template void tensor_operation_gpu::sobel_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src,
	std::shared_ptr<tensor<unsigned char>> &dst, int dx = 1, int dy = 1);
template void tensor_operation_gpu::sobel_gpu<char>(const std::shared_ptr<tensor<char>> &src,
	std::shared_ptr<tensor<char>> &dst, int dx = 1, int dy = 1);
template void tensor_operation_gpu::sobel_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src,
	std::shared_ptr<tensor<unsigned int>> &dst, int dx = 1, int dy = 1);
template void tensor_operation_gpu::sobel_gpu<int>(const std::shared_ptr<tensor<int>> &src,
	std::shared_ptr<tensor<int>> &dst, int dx = 1, int dy = 1);
template void tensor_operation_gpu::sobel_gpu<float>(const std::shared_ptr<tensor<float>> &src,
	std::shared_ptr<tensor<float>> &dst, int dx = 1, int dy = 1);



template void tensor_operation_gpu::sobel_gpu<unsigned char>(const tensor<unsigned char> &src, tensor<unsigned char> &dst, int dx = 1, int dy = 1);
template void tensor_operation_gpu::sobel_gpu<char>(const tensor<char> &src, tensor<char> &dst, int dx = 1, int dy = 1);
template void tensor_operation_gpu::sobel_gpu<unsigned int>(const tensor<unsigned int> &src, tensor<unsigned int> &dst, int dx = 1, int dy = 1);
template void tensor_operation_gpu::sobel_gpu<int>(const tensor<int> &src, tensor<int> &dst, int dx = 1, int dy = 1);
template void tensor_operation_gpu::sobel_gpu<float>(const tensor<float> &src, tensor<float> &dst, int dx = 1, int dy = 1);



template void tensor_operation_gpu::morph_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src,
	std::shared_ptr<tensor<unsigned char>> &dst, excalibur::morphType type = Dilate, int ksize = 3);
template void tensor_operation_gpu::morph_gpu<char>(const std::shared_ptr<tensor<char>> &src,
	std::shared_ptr<tensor<char>> &dst, excalibur::morphType type = Dilate, int ksize = 3);
template void tensor_operation_gpu::morph_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src,
	std::shared_ptr<tensor<unsigned int>> &dst, excalibur::morphType type = Dilate, int ksize = 3);
template void tensor_operation_gpu::morph_gpu<int>(const std::shared_ptr<tensor<int>> &src,
	std::shared_ptr<tensor<int>> &dst, excalibur::morphType type = Dilate, int ksize = 3);
template void tensor_operation_gpu::morph_gpu<float>(const std::shared_ptr<tensor<float>> &src,
	std::shared_ptr<tensor<float>> &dst, excalibur::morphType type = Dilate, int ksize = 3);



template void tensor_operation_gpu::morph_gpu<unsigned char>(const tensor<unsigned char> &src, tensor<unsigned char> &dst, excalibur::morphType type = Dilate, int ksize = 3);
template void tensor_operation_gpu::morph_gpu<char>(const tensor<char> &src, tensor<char> &dst, excalibur::morphType type = Dilate, int ksize = 3);
template void tensor_operation_gpu::morph_gpu<unsigned int>(const tensor<unsigned int> &src, tensor<unsigned int> &dst, excalibur::morphType type = Dilate, int ksize = 3);
template void tensor_operation_gpu::morph_gpu<int>(const tensor<int> &src, tensor<int> &dst, excalibur::morphType type = Dilate, int ksize = 3);
template void tensor_operation_gpu::morph_gpu<float>(const tensor<float> &src, tensor<float> &dst, excalibur::morphType type = Dilate, int ksize = 3);