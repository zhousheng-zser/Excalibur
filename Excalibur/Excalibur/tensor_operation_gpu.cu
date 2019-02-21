#ifdef USE_CUDA
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "tensor_utils.hpp"
#include <glasssix\tensor.hpp>
#include "math_functions.hpp"
#include <iostream>
#ifdef USE_OPENCV
#include <opencv2\opencv.hpp>
#endif
#include <glasssix\timer.hpp>
#include "tensor_operation_gpu.hpp"
#include <algorithm>

#define PI 3.1415926

namespace glasssix
{
	namespace excalibur
	{
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


			if (src->order() == NHWC)
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


			if (src.order() == NHWC)
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
		/// <param name="order">tensor类型：NHWC(tensor按NHWC排列，默认值)、NCHW(tensor按NCHW排列)</param>
		template<typename Dtype>
		void tensor_operation_gpu::mat2tensor_gpu(const cv::Mat &src, std::shared_ptr<tensor<Dtype>>& dst, orderType order = NHWC)
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


			if (order == NHWC)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, 0, order));
				cudaMemcpy(dst->mutable_gpu_data(), src.data, height * width * channels * sizeof(Dtype), cudaMemcpyHostToDevice);
				return;
			}

			dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, 0, order));
			std::shared_ptr<tensor<Dtype>> src_ptr = std::make_shared<tensor<Dtype>>(tensor<Dtype>(std::vector<int>{1, channels, height, width}, 0, order));
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
		/// <param name="order">tensor类型：NHWC(tensor按NHWC排列，默认值)、NCHW(tensor按NCHW排列)</param>
		template<typename Dtype>
		void tensor_operation_gpu::mat2tensor_gpu(const cv::Mat &src, tensor<Dtype>& dst, orderType order = NHWC)
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


			if (order == NHWC)
			{
				dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, 0, order);
				cudaMemcpy(dst.mutable_gpu_data(), src.data, height * width * channels * sizeof(Dtype), cudaMemcpyHostToDevice);
				return;
			}

			dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, 0, order);
			std::shared_ptr<tensor<Dtype>> src_ptr = std::make_shared<tensor<Dtype>>(tensor<Dtype>(std::vector<int>{1, channels, height, width}, 0, order));
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
			CHECK_EQ(src->order(), NCHW);
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
			CHECK_EQ(src.order(), NCHW);
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
			CHECK_EQ(src->order(), NHWC);
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
			CHECK_EQ(src.order(), NHWC);
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
		/// <param name="order">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
		template<typename Dtype>
		__global__
			void kernel_resize(int channels, const Dtype* src_data, int src_height, int src_width,
				Dtype* dst_data, int dst_height, int dst_width,
				interpolationType type, orderType order)
		{
			int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int rowID = totalID / dst_width;
			int colID = totalID % dst_width;
			unsigned maxIndex = src_height * src_width * channels - 1;
			float beta = 0.5f;

			float width_ratio = (float)src_width / dst_width;
			float height_ratio = (float)src_height / dst_height;

			float xf = colID * width_ratio + beta;
			float yf = rowID * height_ratio + beta;
			int x = (int)xf;
			int y = (int)yf;
			float xdiff = xf - x;
			float ydiff = yf - y;

			if (order == NCHW)
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
						unsigned indexA = min(unsigned(src_index), maxIndex);
						unsigned indexB = min(unsigned(src_index + 1), maxIndex);
						unsigned indexC = min(unsigned(src_index + src_width), maxIndex);
						unsigned indexD = min(unsigned(src_index + src_width + 1), maxIndex);
						Dtype A = src_data[indexA];
						Dtype B = src_data[indexB];
						Dtype C = src_data[indexC];
						Dtype D = src_data[indexD];

						dst_data[dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
							static_cast<float>(B) * xdiff * (1 - ydiff) +
							static_cast<float>(C) * ydiff * (1 - xdiff) +
							static_cast<float>(D) * xdiff * ydiff);
					}
				}
			}
			else if (order == NHWC)
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
						unsigned indexA = min(unsigned(src_index), maxIndex);
						unsigned indexB = min(unsigned(src_index + channels), maxIndex);
						unsigned indexC = min(unsigned(src_index + src_width * channels), maxIndex);
						unsigned indexD = min(unsigned(src_index + (src_width + 1) * channels), maxIndex);
						Dtype A = src_data[indexA];
						Dtype B = src_data[indexB];
						Dtype C = src_data[indexC];
						Dtype D = src_data[indexD];

						dst_data[dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
							static_cast<float>(B) * xdiff * (1 - ydiff) +
							static_cast<float>(C) * ydiff * (1 - xdiff) +
							static_cast<float>(D) * xdiff * ydiff);
					}
				}
			}
			else
			{
				return;
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

			if (src->order() == NCHW)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->order()));
			}
			else if (src->order() == NHWC)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->order()));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			Dtype* dst_data = dst->mutable_gpu_data();
			const Dtype* src_data = src->gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(dst_width, dst_height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_resize << <grid_size, block_size >> > (channels, src_data, height, width, dst_data, dst_height, dst_width, type, src->order());
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

			if (src.order() == NCHW)
			{
				dst = tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src.device(), src.order());
			}
			else if (src.order() == NHWC)
			{
				dst = tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src.device(), src.order());
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			Dtype* dst_data = dst.mutable_gpu_data();
			const Dtype* src_data = src.gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(dst_width, dst_height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_resize << <grid_size, block_size >> > (channels, src_data, height, width, dst_data, dst_height, dst_width, type, src.order());
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
		/// <param name="order">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
		template<typename Dtype>
		__global__
			void kernel_rotate_with_center(int channels, const Dtype* src_data, int height, int width,
				Dtype* dst_data, int dst_height, int dst_width,
				float sina, float cosa, float varX, float varY, int fill, interpolationType type, orderType order)
		{
			int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int rowID = totalID / dst_width;
			int colID = totalID % dst_width;
			unsigned maxIndex = height * width * channels - 1;

			float xf = cosa * colID + sina * rowID + varX;
			float yf = -sina * colID + cosa * rowID + varY;

			int x = (int)(xf);
			int y = (int)(yf);
			float xdiff = xf - x;
			float ydiff = yf - y;

			if (order == NCHW)
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
						unsigned indexA = min(unsigned(src_index), maxIndex);
						unsigned indexB = min(unsigned(src_index + 1), maxIndex);
						unsigned indexC = min(unsigned(src_index + width), maxIndex);
						unsigned indexD = min(unsigned(src_index + width + 1), maxIndex);
						Dtype A = src_data[indexA];
						Dtype B = src_data[indexB];
						Dtype C = src_data[indexC];
						Dtype D = src_data[indexD];

						dst_data[dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
							static_cast<float>(B) * xdiff * (1 - ydiff) +
							static_cast<float>(C) * ydiff * (1 - xdiff) +
							static_cast<float>(D) * xdiff * ydiff);
					}
				}
			}
			else if (order == NHWC)
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
						unsigned indexA = min(unsigned(src_pos2), maxIndex);
						unsigned indexB = min(unsigned(src_pos2 + channels), maxIndex);
						unsigned indexC = min(unsigned(src_pos2 + width * channels), maxIndex);
						unsigned indexD = min(unsigned(src_pos2 + (width + 1) * channels), maxIndex);
						Dtype A = src_data[indexA];
						Dtype B = src_data[indexB];
						Dtype C = src_data[indexC];
						Dtype D = src_data[indexD];

						dst_data[dst_pos2] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
							static_cast<float>(B) * xdiff * (1 - ydiff) +
							static_cast<float>(C) * ydiff * (1 - xdiff) +
							static_cast<float>(D) * xdiff * ydiff);
					}
				}
			}
			else
			{
				return;
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

			if (src->order() == NCHW)
			{
				dst.reset(new excalibur::tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->order()));
			}
			else if (src->order() == NHWC)
			{
				dst.reset(new excalibur::tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->order()));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			Dtype* dst_data = dst->mutable_gpu_data();
			const Dtype* src_data = src->gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(dst_width, dst_height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_rotate_with_center << <grid_size, block_size >> > (channels, src_data, height, width, dst_data, dst_height, dst_width, sina, cosa, VarX, VarY, fill, type, src->order());
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

			if (src.order() == NCHW)
			{
				dst = excalibur::tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src.device(), src.order());
			}
			else if (src.order() == NHWC)
			{
				dst = excalibur::tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src.device(), src.order());
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			Dtype* dst_data = dst.mutable_gpu_data();
			const Dtype* src_data = src.gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(dst_width, dst_height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_rotate_with_center << <grid_size, block_size >> > (channels, src_data, height, width, dst_data, dst_height, dst_width, sina, cosa, VarX, VarY, fill, type, src.order());
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
		/// <param name="order">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
		template<typename Dtype>
		__global__
			void kernel_rotate_with_points(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, double* M_data, int fill = 0, interpolationType type = Bilinear, orderType order = NCHW)
		{
			int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int rowID = totalID / width;
			int colID = totalID % width;
			unsigned maxIndex = height * width * channels - 1;

			double xf = M_data[0] * colID + M_data[1] * rowID + M_data[2];
			double yf = M_data[3] * colID + M_data[4] * rowID + M_data[5];
			int x = (int)xf;
			int y = (int)yf;
			float xdiff = xf - x;
			float ydiff = yf - y;

			if (order == NCHW)
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
							unsigned indexA = min(unsigned(src_index), maxIndex);
							unsigned indexB = min(unsigned(src_index + 1), maxIndex);
							unsigned indexC = min(unsigned(src_index + width), maxIndex);
							unsigned indexD = min(unsigned(src_index + width + 1), maxIndex);
							Dtype A = src_data[indexA];
							Dtype B = src_data[indexB];
							Dtype C = src_data[indexC];
							Dtype D = src_data[indexD];

							dst_data[dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
								static_cast<float>(B) * xdiff * (1 - ydiff) +
								static_cast<float>(C) * ydiff * (1 - xdiff) +
								static_cast<float>(D) * xdiff * ydiff);
						}
					}
				}
			}
			else if (order == NHWC)
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
							unsigned indexA = min(unsigned(src_index), maxIndex);
							unsigned indexB = min(unsigned(src_index + channels), maxIndex);
							unsigned indexC = min(unsigned(src_index + width * channels), maxIndex);
							unsigned indexD = min(unsigned(src_index + (width + 1) * channels), maxIndex);
							Dtype A = src_data[indexA];
							Dtype B = src_data[indexB];
							Dtype C = src_data[indexC];
							Dtype D = src_data[indexD];

							dst_data[dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
								static_cast<float>(B) * xdiff * (1 - ydiff) +
								static_cast<float>(C) * ydiff * (1 - xdiff) +
								static_cast<float>(D) * xdiff * ydiff);
						}
					}
				}
			}
			else
			{
				return;
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

			std::vector<std::vector<double> > M;
			std::vector<std::vector<double> > reverse_M;
			M.resize(3);
			for (size_t i = 0; i < M.size(); i++)
			{
				M[i].resize(3);
			}

			M[0][0] = a;
			M[0][1] = b;
			M[0][2] = (1 - a) * (double)center.x - b * (double)center.y;
			M[1][0] = -1 * b;
			M[1][1] = a;
			M[1][2] = b * (double)center.x + (1 - a) * (double)center.y;
			M[2][0] = 0;
			M[2][1] = 0;
			M[2][2] = 1;

			bool isInverted = math_functions::GetMatrixInverse(M, reverse_M);
			if (!isInverted)
			{
				LOG(FATAL) << "cannot rotate!!!";
				return;
			}

			double *reverse_M_data = nullptr;
			cudaMalloc(&reverse_M_data, 9 * sizeof(double));
			cudaMemcpy(reverse_M_data, &(reverse_M[0][0]), 3 * sizeof(double), cudaMemcpyDefault);
			cudaMemcpy(reverse_M_data + 3, &(reverse_M[1][0]), 3 * sizeof(double), cudaMemcpyDefault);
			cudaMemcpy(reverse_M_data + 6, &(reverse_M[2][0]), 3 * sizeof(double), cudaMemcpyDefault);

			if (src->order() == NCHW)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->order()));
			}
			else if (src->order() == NHWC)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->order()));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			const Dtype* src_data = src->gpu_data();
			Dtype* dst_data = dst->mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_rotate_with_points << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, reverse_M_data, fill, type, src->order());

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

			std::vector<std::vector<double> > M;
			std::vector<std::vector<double> > reverse_M;
			M.resize(3);
			for (size_t i = 0; i < M.size(); i++)
			{
				M[i].resize(3);
			}

			M[0][0] = a;
			M[0][1] = b;
			M[0][2] = (1 - a) * (double)center.x - b * (double)center.y;
			M[1][0] = -1 * b;
			M[1][1] = a;
			M[1][2] = b * (double)center.x + (1 - a) * (double)center.y;
			M[2][0] = 0;
			M[2][1] = 0;
			M[2][2] = 1;

			bool isInverted = math_functions::GetMatrixInverse(M, reverse_M);
			if (!isInverted)
			{
				LOG(FATAL) << "cannot rotate!!!";
				return;
			}

			double *reverse_M_data = nullptr;
			cudaMalloc(&reverse_M_data, 9 * sizeof(double));
			cudaMemcpy(reverse_M_data, &(reverse_M[0][0]), 3 * sizeof(double), cudaMemcpyDefault);
			cudaMemcpy(reverse_M_data + 3, &(reverse_M[1][0]), 3 * sizeof(double), cudaMemcpyDefault);
			cudaMemcpy(reverse_M_data + 6, &(reverse_M[2][0]), 3 * sizeof(double), cudaMemcpyDefault);

			if (src.order() == NCHW)
			{
				dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.order());
			}
			else if (src.order() == NHWC)
			{
				dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.order());
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			const Dtype* src_data = src.gpu_data();
			Dtype* dst_data = dst.mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_rotate_with_points << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, reverse_M_data, fill, type, src.order());

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
		/// <param name="order">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
		template<typename Dtype>
		__global__
			void kernel_flip(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, flipType axis, orderType order)
		{
			int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int rowID = totalID / width;
			int colID = totalID % width;

			if (order == NCHW)
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
			else if (order == NHWC)
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
			else
			{
				return;
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

			if (src->order() == NCHW)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->order()));
			}
			else if (src->order() == NHWC)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->order()));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			const Dtype* src_data = src->gpu_data();
			Dtype* dst_data = dst->mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_flip << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, axis, src->order());
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

			if (src.order() == NCHW)
			{
				dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.order());
			}
			else if (src.order() == NHWC)
			{
				dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.order());
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			const Dtype* src_data = src.gpu_data();
			Dtype* dst_data = dst.mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_flip << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, axis, src.order());
		}






		/// <summary>
		/// 三通道rgb图像转换为单通道灰度图像
		/// </summary>
		/// <param name="src_data">原图像数据</param>
		/// <param name="dst_data">新图像数据</param>
		/// <param name="height">图像高度</param>
		/// <param name="width">图像宽度</param>
		/// <param name="order">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
		template<typename Dtype>
		__global__
			void kernel_rgb2gray(const Dtype* src_data, Dtype* dst_data, int height, int width, orderType order)
		{
			int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int rowID = totalID / width;
			int colID = totalID % width;

			int index = rowID * width + colID;
			int offset = height * width;

			//opencv读取RGB图像后，以B、G、R的顺序进行存储
			//转换公式为:gray=0.114*B+0.587*G+0.299*R
			if (order == NCHW)
			{
				dst_data[index] = Dtype(static_cast<float>(src_data[index]) * 0.114f +
					static_cast<float>(src_data[offset * 1 + index]) * 0.587f +
					static_cast<float>(src_data[offset * 2 + index]) * 0.299f);
			}
			else if (order == NHWC)
			{
				dst_data[index] = Dtype(static_cast<float>(src_data[3 * index]) * 0.114f +
					static_cast<float>(src_data[3 * index + 1]) * 0.587f +
					static_cast<float>(src_data[3 * index + 2]) * 0.299f);
			}
			else
			{
				return;
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

			if (src->order() == NCHW)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, 1, height, width}, src->device(), src->order()));
			}
			else if (src->order() == NHWC)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, 1}, src->device(), src->order()));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			const Dtype* src_data = src->gpu_data();
			Dtype* dst_data = dst->mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_rgb2gray << <grid_size, block_size >> > (src_data, dst_data, height, width, src->order());
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

			if (src.order() == NCHW)
			{
				dst = tensor<Dtype>(std::vector<int>{1, 1, height, width}, src.device(), src.order());
			}
			else if (src.order() == NHWC)
			{
				dst = tensor<Dtype>(std::vector<int>{1, height, width, 1}, src.device(), src.order());
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			const Dtype* src_data = src.gpu_data();
			Dtype* dst_data = dst.mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_rgb2gray << <grid_size, block_size >> > (src_data, dst_data, height, width, src.order());
		}






		/// <summary>
		/// 矩阵转置
		/// </summary>
		/// <param name="src_data">原图像数据</param>
		/// <param name="dst_data">转置后的新图像数据</param>
		/// <param name="channels">图像通道数</param>
		/// <param name="height">图像高度</param>
		/// <param name="width">图像宽度</param>
		/// <param name="order">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
		template<typename Dtype>
		__global__
			void kernel_matrix_transpose(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, orderType order)
		{
			int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int rowID = totalID / width;
			int colID = totalID % width;

			if (order == NCHW)
			{
				int offset = height * width;

				int dst_index = rowID * width + colID;
				int src_index = colID * height + rowID;

				for (int ch = 0; ch < channels; ++ch) {
					int channel_offset = ch * offset;
					dst_data[dst_index + channel_offset] = src_data[src_index + channel_offset];
				}
			}
			else if (order == NHWC)
			{
				for (int ch = 0; ch < channels; ++ch)
				{
					int dst_index = (rowID * width + colID) * channels + ch;
					int src_index = (colID * height + rowID) * channels + ch;
					dst_data[dst_index] = src_data[src_index];
				}
			}
			else
			{
				return;
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

			if (src->order() == NCHW)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, width, height}, src->device(), src->order()));
			}
			else if (src->order() == NHWC)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, width, height, channels}, src->device(), src->order()));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			const Dtype* src_data = src->gpu_data();
			Dtype* dst_data = dst->mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(height, width, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_matrix_transpose << <grid_size, block_size >> > (src_data, dst_data, channels, width, height, src->order());
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

			if (src.order() == NCHW)
			{
				dst = tensor<Dtype>(std::vector<int>{1, channels, width, height}, src.device(), src.order());
			}
			else if (src.order() == NHWC)
			{
				dst = tensor<Dtype>(std::vector<int>{1, width, height, channels}, src.device(), src.order());
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			const Dtype* src_data = src.gpu_data();
			Dtype* dst_data = dst.mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(height, width, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_matrix_transpose << <grid_size, block_size >> > (src_data, dst_data, channels, width, height, src.order());
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
		/// <param name="order">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
		template<typename Dtype, typename Rtype>
		__global__
			void kernel_ROI(const Dtype* src_data, int channels, int src_height, int src_width,
				Dtype* dst_data, excalibur::rectangle<Rtype> rect, orderType order)
		{
			int dst_height = (int)rect.h;
			int dst_width = (int)rect.w;

			int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int rowID = totalID / dst_width;
			int colID = totalID % dst_width;

			if (order == NCHW)
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
			else if (order == NHWC)
			{
				for (int ch = 0; ch < channels; ++ch)
				{
					int src_index = (rowID + rect.y) * src_width * channels + (colID + rect.x) * channels + ch;
					int dst_index = (rowID * dst_width + colID) * channels + ch;

					dst_data[dst_index] = src_data[src_index];
				}
			}
			else
			{
				return;
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

			if (src->order() == NCHW)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->order()));
			}
			else if (src->order() == NHWC)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->order()));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			Dtype* dst_data = dst->mutable_gpu_data();
			const Dtype* src_data = src->gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(dst_width, dst_height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_ROI << <grid_size, block_size >> > (src_data, channels, height, width, dst_data, rect, src->order());
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

			if (src.order() == NCHW)
			{
				dst = tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src.device(), src.order());
			}
			else if (src.order() == NHWC)
			{
				dst = tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src.device(), src.order());
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			Dtype* dst_data = dst.mutable_gpu_data();
			const Dtype* src_data = src.gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(dst_width, dst_height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_ROI << <grid_size, block_size >> > (src_data, channels, height, width, dst_data, rect, src.order());
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

			if (src->order() == NCHW)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, 1, height, width}, src->device(), src->order()));
			}
			else if (src->order() == NHWC)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, 1}, src->device(), src->order()));
			}
			else
			{
				NOT_IMPLEMENTED;
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

			if (src.order() == NCHW)
			{
				dst = tensor<Dtype>(std::vector<int>{1, 1, height, width}, src.device(), src.order());
			}
			else if (src.order() == NHWC)
			{
				dst = tensor<Dtype>(std::vector<int>{1, height, width, 1}, src.device(), src.order());
			}
			else
			{
				NOT_IMPLEMENTED;
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
		/// <param name="order">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
		template<typename Dtype>
		__global__
			void kernel_warp_affine(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, float* X, int fill, interpolationType type, orderType order)
		{
			int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int rowID = totalID / width;
			int colID = totalID % width;
			unsigned maxIndex = height * width * channels - 1;

			double xf = X[0] * colID + X[1] * rowID + X[2];
			double yf = X[3] * colID + X[4] * rowID + X[5];
			int x = (int)xf;
			int y = (int)yf;
			float xdiff = xf - x;
			float ydiff = yf - y;

			if (order == NCHW)
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
						if (type == Nearest)
						{
							dst_data[dst_index] = src_data[src_index];
						}
						else if (type == Bilinear)
						{
							unsigned indexA = min(unsigned(src_index), maxIndex);
							unsigned indexB = min(unsigned(src_index + 1), maxIndex);
							unsigned indexC = min(unsigned(src_index + width), maxIndex);
							unsigned indexD = min(unsigned(src_index + width + 1), maxIndex);
							Dtype A = src_data[indexA];
							Dtype B = src_data[indexB];
							Dtype C = src_data[indexC];
							Dtype D = src_data[indexD];

							dst_data[dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
								static_cast<float>(B) * xdiff * (1 - ydiff) +
								static_cast<float>(C) * ydiff * (1 - xdiff) +
								static_cast<float>(D) * xdiff * ydiff);
						}
					}
				}
			}
			else if (order == NHWC)
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
						if (type == Nearest)
						{
							dst_data[dst_index] = src_data[src_index];
						}
						else if (type == Bilinear)
						{
							unsigned indexA = min(unsigned(src_index), maxIndex);
							unsigned indexB = min(unsigned(src_index + channels), maxIndex);
							unsigned indexC = min(unsigned(src_index + width * channels), maxIndex);
							unsigned indexD = min(unsigned(src_index + (width + 1) * channels), maxIndex);
							Dtype A = src_data[indexA];
							Dtype B = src_data[indexB];
							Dtype C = src_data[indexC];
							Dtype D = src_data[indexD];

							dst_data[dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
								static_cast<float>(B) * xdiff * (1 - ydiff) +
								static_cast<float>(C) * ydiff * (1 - xdiff) +
								static_cast<float>(D) * xdiff * ydiff);
						}
					}
				}
			}
			else
			{
				return;
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
			const std::vector<point<Ptype>> &src_point, const std::vector<point<Ptype>> &dst_point, int fill = 0, interpolationType type = Bilinear)
		{
			if (src_point.size() != 3 || dst_point.size() != 3)
			{
				LOG(WARNING) << "please use 3 src_points and 3 dst_points.";
				return;
			}

			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();

			//AX=B，先扩充至6×6和6×1
			std::vector<std::vector<float> > A;
			std::vector<float> B, X;
			A.resize(6);
			for (size_t i = 0; i < 6; i = i + 2)
			{
				A[i].push_back(float(dst_point[i / 2].x));
				A[i].push_back(float(dst_point[i / 2].y));
				A[i].push_back(1.0f);
				A[i].push_back(0.0f);
				A[i].push_back(0.0f);
				A[i].push_back(0.0f);

				A[i + 1].push_back(0.0f);
				A[i + 1].push_back(0.0f);
				A[i + 1].push_back(0.0f);
				A[i + 1].push_back(float(dst_point[i / 2].x));
				A[i + 1].push_back(float(dst_point[i / 2].y));
				A[i + 1].push_back(1.0f);
			}

			for (size_t i = 0; i < 3; i++)
			{
				B.push_back(float(src_point[i].x));
				B.push_back(float(src_point[i].y));
			}

			X = math_functions::gauss_all(A, B);
			float *X_data = nullptr;
			cudaMalloc(&X_data, 6 * sizeof(float));
			cudaMemcpy(X_data, X.data(), 6 * sizeof(float), cudaMemcpyDefault);

			if (src->order() == NCHW)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->order()));
			}
			else if (src->order() == NHWC)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->order()));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			const Dtype* src_data = src->gpu_data();
			Dtype* dst_data = dst->mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_warp_affine << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, X_data, fill, type, src->order());
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

			CHECK_EQ(src.num(), 1);
			int channels = src.channels();
			int height = src.height();
			int width = src.width();

			//AX=B，先扩充至6×6和6×1
			std::vector<std::vector<float> > A;
			std::vector<float> B, X;
			A.resize(6);
			for (size_t i = 0; i < 6; i = i + 2)
			{
				A[i].push_back(float(dst_point[i / 2].x));
				A[i].push_back(float(dst_point[i / 2].y));
				A[i].push_back(1.0f);
				A[i].push_back(0.0f);
				A[i].push_back(0.0f);
				A[i].push_back(0.0f);

				A[i + 1].push_back(0.0f);
				A[i + 1].push_back(0.0f);
				A[i + 1].push_back(0.0f);
				A[i + 1].push_back(float(dst_point[i / 2].x));
				A[i + 1].push_back(float(dst_point[i / 2].y));
				A[i + 1].push_back(1.0f);
			}

			for (size_t i = 0; i < 3; i++)
			{
				B.push_back(float(src_point[i].x));
				B.push_back(float(src_point[i].y));
			}

			X = math_functions::gauss_all(A, B);
			float *X_data = nullptr;
			cudaMalloc(&X_data, 6 * sizeof(float));
			cudaMemcpy(X_data, X.data(), 6 * sizeof(float), cudaMemcpyDefault);

			if (src.order() == NCHW)
			{
				dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.order());
			}
			else if (src.order() == NHWC)
			{
				dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.order());
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			const Dtype* src_data = src.gpu_data();
			Dtype* dst_data = dst.mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_warp_affine << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, X_data, fill, type, src.order());
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
		/// <param name="order">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
		template<typename Dtype>
		__global__
			void kernel_gaussian_blur(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, int ksize, double *paras, orderType order)
		{
			int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int rowID = totalID / width;
			int colID = totalID % width;

			int half = (ksize - 1) * 0.5;

			if (order == NCHW)
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
			else if (order == NHWC)
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
			else
			{
				return;
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

			if (src->order() == NCHW)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->order()));
			}
			else if (src->order() == NHWC)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->order()));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			const Dtype* src_data = src->gpu_data();
			Dtype* dst_data = dst->mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_gaussian_blur << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, ksize, paras, src->order());
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

			if (src.order() == NCHW)
			{
				dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.order());
			}
			else if (src.order() == NHWC)
			{
				dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.order());
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			const Dtype* src_data = src.gpu_data();
			Dtype* dst_data = dst.mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_gaussian_blur << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, ksize, paras, src.order());
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
		/// <param name="order">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
		template<typename Dtype>
		__global__
			void kernel_sobel(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, int dx, int dy, orderType order)
		{
			int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int rowID = totalID / width;
			int colID = totalID % width;

			//第一行、最后一行、第一列、最后一列不作处理
			if (rowID == 0 || rowID == height - 2 || colID == 0 || colID == width - 2)
			{
				return;
			}

			if (order == NCHW)
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
			else if (order == NHWC)
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
			else
			{
				return;
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

			if (src->order() == NCHW)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->order()));
			}
			else if (src->order() == NHWC)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->order()));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			Dtype* dst_data = dst->mutable_gpu_data();
			const Dtype* src_data = src->gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_sobel << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, dx, dy, src->order());
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

			if (src.order() == NCHW)
			{
				dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.order());
			}
			else if (src.order() == NHWC)
			{
				dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.order());
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			Dtype* dst_data = dst.mutable_gpu_data();
			const Dtype* src_data = src.gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_sobel << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, dx, dy, src.order());
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
		/// <param name="order">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
		template<typename Dtype>
		__global__
			void kernel_morph(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, excalibur::morphType type, int ksize, orderType order)
		{
			int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int rowID = totalID / width;
			int colID = totalID % width;

			int half = (ksize - 1) * 0.5;
			int offset = height * width;

			if (order == NCHW)
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
			else if (order == NHWC)
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
			else
			{
				return;
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

			if (src->order() == NCHW)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, src->device(), src->order()));
			}
			else if (src->order() == NHWC)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, src->device(), src->order()));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			Dtype* dst_data = dst->mutable_gpu_data();
			const Dtype* src_data = src->gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_morph << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, type, ksize, src->order());
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

			if (src.order() == NCHW)
			{
				dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, src.device(), src.order());
			}
			else if (src.order() == NHWC)
			{
				dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, src.device(), src.order());
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			Dtype* dst_data = dst.mutable_gpu_data();
			const Dtype* src_data = src.gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_morph << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, type, ksize, src.order());
		}



		/// <summary>
		/// tensor类型转换
		/// </summary>
		/// <param name="src_data">原图像数据</param>
		/// <param name="dst_data">转换后的新图像数据</param>
		template <typename DtypeSRC, typename DtypeDST>
		__global__
			void kernel_type_converter(const DtypeSRC* src_data, DtypeDST* dst_data)
		{
			int index = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			dst_data[index] = DtypeDST(src_data[index]);
		}



		/// <summary>
		/// tensor类型转换
		/// </summary>
		/// <param name="src">源tensor</param>
		/// <param name="dst">转换后的tensor</param>
		template <typename DtypeSRC, typename DtypeDST>
		void tensor_operation_gpu::type_converter_gpu(const std::shared_ptr<tensor<DtypeSRC>> &src, std::shared_ptr<tensor<DtypeDST>> &dst)
		{
			const DtypeSRC* src_data = src->gpu_data();
			DtypeDST* dst_data = dst->mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(src->count(), 1, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_type_converter << <grid_size, block_size >> > (src_data, dst_data);
		}



		/// <summary>
		/// tensor类型转换
		/// </summary>
		/// <param name="src">源tensor</param>
		/// <param name="dst">转换后的tensor</param>
		template <typename DtypeSRC, typename DtypeDST>
		void tensor_operation_gpu::type_converter_gpu(const tensor<DtypeSRC>* src, tensor<DtypeDST>* dst)
		{
			const DtypeSRC* src_data = src->gpu_data();
			DtypeDST* dst_data = dst->mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(src->count(), 1, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_type_converter << <grid_size, block_size >> > (src_data, dst_data);
		}



		/// <summary>
		/// tensor预处理，按通道对每个像素点进行相同的乘法和加法变换
		/// </summary>
		/// <param name="dst_data">输入及输出的图像数据</param>
		/// <param name="channels">图像通道数</param>
		/// <param name="height">图像高度</param>
		/// <param name="width">图像宽度</param>
		/// <param name="order">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
		template<typename Dtype>
		__global__
			void kernel_preprocess_tensors(Dtype* dst_data, int num, int channels, int height, int width, orderType order)
		{
			int totalID = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int offset = channels * width * height;
			int numID = totalID / offset;
			int remainID = totalID % offset;
			int rowID = remainID / width;
			int colID = remainID % width;

			for (size_t n = 0; n < num; n++)
			{
				int num_offset = n * offset;
				if (channels == 1)
				{
					float means[] = { 127.5f };
					float var = 0.0078125f;
					int index = num_offset + rowID * width + colID;
					dst_data[index] = Dtype((dst_data[index] - means[0]) * var);
				}
				else if (channels == 3)
				{
					float means[] = { 104.f, 117.0f, 124.f };
					float var = 0.0078125f;

					if (order == NCHW)
					{
						for (int ch = 0; ch < channels; ++ch)
						{
							int channel_offset = ch * height * width;
							int index = num_offset + channel_offset + rowID * width + colID;
							dst_data[index] = Dtype((dst_data[index] - means[ch]) * var);
						}
					}
					else if (order == NHWC)
					{
						for (int ch = 0; ch < channels; ++ch)
						{
							int index = num_offset + (rowID * width + colID) * channels + ch;
							dst_data[index] = Dtype((dst_data[index] - means[ch]) * var);
						}
			        }
			        else
			        {
				        return;
			        }
				}
			}

		}



		/// <summary>
		/// tensor预处理，按通道对每个像素点进行相同的乘法和加法变换
		/// </summary>
		/// <param name="dst">包含图像数据的tensor</param>
		template <typename Dtype>
		void tensor_operation_gpu::preprocess_tensors_gpu(std::shared_ptr<tensor<Dtype>> dst)
		{
			int num = dst->num();
			int channels = dst->channels();
			int height = dst->height();
			int width = dst->width();


			Dtype* dst_data = dst->mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_preprocess_tensors << <grid_size, block_size >> > (dst_data, num, channels, height, width, dst->order());
		}



		/// <summary>
		/// tensor预处理，按通道对每个像素点进行相同的乘法和加法变换
		/// </summary>
		/// <param name="dst">包含图像数据的tensor</param>
		template <typename Dtype>
		void tensor_operation_gpu::preprocess_tensors_gpu(tensor<Dtype>* dst)
		{
			int num = dst->num();
			int channels = dst->channels();
			int height = dst->height();
			int width = dst->width();

			Dtype* dst_data = dst->mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, 1);

			//按照设置的blockSize和gridSize启动内核函数
			kernel_preprocess_tensors << <grid_size, block_size >> > (dst_data, num, channels, height, width, dst->order());
		}



		/// <summary>
		/// 边界填充
		/// </summary>
		/// <param name="src">包含原图像数据的tensor</param>
		/// <param name="dst">填充后的tensor</param>
		/// <param name="top">图像上方的填充高度</param>
		/// <param name="bottom">图像下方的填充高度</param>
		/// <param name="left">图像左侧的填充宽度</param>
		/// <param name="right">图像右侧的填充宽度</param>
		/// <param name="type">边界类型：Border_Constant（用第8个参数fill设定的常量值进行填充，默认值）、Border_Replicate（复制相邻像素点的值进行填充）</param>
		/// <param name="fill">边界类型为Border_Constant时有效，默认为0</param>
		template <typename Dtype>
		void tensor_operation_gpu::make_border_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst,
			int top, int bottom, int left, int right, borderType type = Border_Constant, int fill = 0)
		{
			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			int src_offset = height * width;

			int dst_height = height + top + bottom;
			int dst_width = width + left + right;
			int dst_offset = dst_height * dst_width;

			if (top < 0 || bottom < 0 || left < 0 || right < 0)
			{
				LOG(ERROR) << "top, bottom, left, right: should all be non-negtive.";
				return;
			}

			if (dst_height == height && dst_width == width)
			{
				LOG(WARNING) << "Just copy from the source.";
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				return;
			}

			if (src->order() == NCHW)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->order()));
				Dtype* dst_data = dst->mutable_gpu_data();
				const Dtype* src_data = src->gpu_data();

				if (type == Border_Constant)
				{
					for (int ch = 0; ch < channels; ++ch)
					{
						int src_channel_offset = ch * src_offset;
						int dst_channel_offset = ch * dst_offset;

						//top
						for (int row = 0; row < top; row++)
						{
							int dst_index = dst_channel_offset + row * dst_width;
							for (int col = 0; col < dst_width; col++)
							{
								dst_data[dst_index + col] = (Dtype)fill;
							}
						}

						//center
						for (int row = top; row < top + height; ++row)
						{
							int src_index = src_channel_offset + (row - top) * width;
							int dst_index = dst_channel_offset + row * dst_width;

							for (int col = 0; col < left; col++)
							{
								dst_data[dst_index + col] = (Dtype)fill;
							}

							cudaMemcpy(dst_data + dst_index + left, src_data + src_index, width * sizeof(Dtype), cudaMemcpyDefault);

							for (int col = left + width; col < dst_width; col++)
							{
								dst_data[dst_index + col] = (Dtype)fill;
							}
						}

						//bottom
						for (int row = top + height; row < dst_height; row++)
						{
							int dst_index = dst_channel_offset + row * dst_width;
							for (int col = 0; col < dst_width; col++)
							{
								dst_data[dst_index + col] = (Dtype)fill;
							}
						}
					}
				}
				else if (type == Border_Replicate)
				{
					for (int ch = 0; ch < channels; ++ch)
					{
						int src_channel_offset = ch * src_offset;
						int dst_channel_offset = ch * dst_offset;

						//top
						for (int row = 0; row < top; ++row)
						{
							int dst_index = dst_channel_offset + row * dst_width;

							//left
							for (int col = 0; col < left; col++)
							{
								dst_data[dst_index + col] = src_data[src_channel_offset];
							}

							//center
							cudaMemcpy(dst_data + dst_index + left, src_data + src_channel_offset, width * sizeof(Dtype), cudaMemcpyDefault);

							//right
							for (int col = left + width; col < dst_width; col++)
							{
								dst_data[dst_index + col] = src_data[src_channel_offset + width - 1];
							}
						}

						//center
						for (int row = top; row < top + height; ++row)
						{
							int src_index = src_channel_offset + (row - top) * width;
							int dst_index = dst_channel_offset + row * dst_width;

							//left
							for (int col = 0; col < left; col++)
							{
								dst_data[dst_index + col] = src_data[src_index];
							}

							//center
							cudaMemcpy(dst_data + dst_index + left, src_data + src_index, width * sizeof(Dtype), cudaMemcpyDefault);

							//right
							for (int col = left + width; col < dst_width; col++)
							{
								dst_data[dst_index + col] = src_data[src_index + width - 1];
							}
						}

						//bottom
						for (int row = top + height; row < dst_height; ++row)
						{
							int src_index = src_channel_offset + (height - 1) * width;
							int dst_index = dst_channel_offset + row * dst_width;

							//left
							for (int col = 0; col < left; col++)
							{
								dst_data[dst_index + col] = src_data[src_index];
							}

							//center
							cudaMemcpy(dst_data + dst_index + left, src_data + src_index, width * sizeof(Dtype), cudaMemcpyDefault);

							//right
							for (int col = left + width; col < dst_width; col++)
							{
								dst_data[dst_index + col] = src_data[src_index + width - 1];
							}
						}
					}
				}
				else
				{
					LOG(ERROR) << "Un-support border type.";
					return;
				}
			}
			else if (src->order() == NHWC)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->order()));
				Dtype* dst_data = dst->mutable_gpu_data();
				const Dtype* src_data = src->gpu_data();

				if (type == Border_Constant)
				{
					//top
					for (int row = 0; row < top; row++)
					{
						int dst_index1 = row * dst_width * channels;
						for (int col = 0; col < dst_width; col++)
						{
							int dst_index2 = dst_index1 + col * channels;
							for (int ch = 0; ch < channels; ch++)
							{
								dst_data[dst_index2 + ch] = (Dtype)fill;
							}
						}
					}

					//center
					for (int row = top; row < top + height; ++row)
					{
						int src_index = (row - top) * width * channels;

						//left
						int dst_index1 = row * dst_width * channels;
						for (int col = 0; col < left; col++)
						{
							int dst_index2 = dst_index1 + col * channels;
							for (int ch = 0; ch < channels; ch++)
							{
								dst_data[dst_index2 + ch] = (Dtype)fill;
							}
						}

						//center
						cudaMemcpy(dst_data + dst_index1 + left * channels, src_data + src_index, width * channels * sizeof(Dtype), cudaMemcpyDefault);

						//right
						for (int col = left + width; col < dst_width; col++)
						{
							int dst_index2 = dst_index1 + col * channels;
							for (int ch = 0; ch < channels; ch++)
							{
								dst_data[dst_index2 + ch] = (Dtype)fill;
							}
						}
					}

					//bottom
					for (int row = top + height; row < dst_height; row++)
					{
						int dst_index1 = row * dst_width * channels;
						for (int col = 0; col < dst_width; col++)
						{
							int dst_index2 = dst_index1 + col * channels;
							for (int ch = 0; ch < channels; ch++)
							{
								dst_data[dst_index2 + ch] = (Dtype)fill;
							}
						}
					}
				}
				else if (type == Border_Replicate)
				{
					//top
					for (int row = 0; row < top; ++row)
					{
						int dst_index1 = row * dst_width * channels;

						//left
						for (int col = 0; col < left; ++col)
						{
							int dst_index2 = dst_index1 + col * channels;
							for (int ch = 0; ch < channels; ++ch)
							{
								dst_data[dst_index2 + ch] = src_data[ch];
							}
						}

						//center
						cudaMemcpy(dst_data + dst_index1 + left * channels, src_data, width * channels * sizeof(Dtype), cudaMemcpyDefault);

						//right
						for (int col = left + width; col < dst_width; ++col)
						{
							int dst_index2 = dst_index1 + col * channels;
							int src_index = (width - 1) * channels;
							for (int ch = 0; ch < channels; ++ch)
							{
								dst_data[dst_index2 + ch] = src_data[src_index + ch];
							}
						}
					}


					//center
					for (int row = top; row < top + height; ++row)
					{
						int src_index1 = (row - top) * width * channels;
						int dst_index1 = row * dst_width * channels;

						//left
						for (int col = 0; col < left; ++col)
						{
							int dst_index2 = dst_index1 + col * channels;
							for (int ch = 0; ch < channels; ++ch)
							{
								dst_data[dst_index2 + ch] = src_data[src_index1 + ch];
							}
						}

						//center
						cudaMemcpy(dst_data + dst_index1 + left * channels, src_data + src_index1, width * channels * sizeof(Dtype), cudaMemcpyDefault);

						//right
						int src_index2 = src_index1 + (width - 1) * channels;
						for (int col = left + width; col < dst_width; ++col)
						{
							int dst_index2 = dst_index1 + col * channels;
							for (int ch = 0; ch < channels; ++ch)
							{
								dst_data[dst_index2 + ch] = src_data[src_index2 + ch];
							}
						}
					}


					//bottom
					for (int row = top + height; row < dst_height; ++row)
					{
						int dst_index1 = row * dst_width * channels;
						int src_index1 = (height - 1) * width * channels;

						//left
						for (int col = 0; col < left; ++col)
						{
							int dst_index2 = dst_index1 + col * channels;
							for (int ch = 0; ch < channels; ++ch)
							{
								dst_data[dst_index2 + ch] = src_data[src_index1 + ch];
							}
						}

						//center
						cudaMemcpy(dst_data + dst_index1 + left * channels, src_data + src_index1, width * channels * sizeof(Dtype), cudaMemcpyDefault);

						//right
						int src_index2 = src_index1 + (width - 1) * channels;
						for (int col = left + width; col < dst_width; ++col)
						{
							int dst_index2 = dst_index1 + col * channels;
							for (int ch = 0; ch < channels; ++ch)
							{
								dst_data[dst_index2 + ch] = src_data[src_index2 + ch];
							}
						}
					}
				}
				else
				{
					LOG(ERROR) << "Un-support border type.";
					return;
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}



		/// <summary>
		/// 边界裁剪
		/// </summary>
		/// <param name="src">包含原图像数据的tensor</param>
		/// <param name="dst">裁剪后的tensor</param>
		/// <param name="top">图像上方的裁剪高度</param>
		/// <param name="bottom">图像下方的裁剪高度</param>
		/// <param name="left">图像左侧的裁剪宽度</param>
		/// <param name="right">图像右侧的裁剪宽度</param>
		template <typename Dtype>
		void tensor_operation_gpu::cut_border_gpu(const std::shared_ptr<tensor<Dtype>> &src,
			std::shared_ptr<tensor<Dtype>>& dst, int top, int bottom, int left, int right)
		{
			if (top < 0 || bottom < 0 || left < 0 || right < 0)
			{
				LOG(ERROR) << "top, bottom, left, right: should all be non-negtive.";
				return;
			}

			CHECK_EQ(src->num(), 1);
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			int src_offset = height * width;

			int dst_height = height - top - bottom;
			int dst_width = width - left - right;
			int dst_offset = dst_height * dst_width;

			if (dst_height == height && dst_width == width)
			{
				LOG(WARNING) << "Just copy from the source.";
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				return;
			}

			if (dst_height <= 0 || dst_width <= 0)
			{
				LOG(ERROR) << "Illegal input size.";
				return;
			}

			if (src->order() == NCHW)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, dst_height, dst_width}, src->device(), src->order()));
				Dtype* dst_data = dst->mutable_gpu_data();
				const Dtype* src_data = src->gpu_data();

				for (int ch = 0; ch < channels; ++ch)
				{
					int src_channel_offset = ch * src_offset;
					int dst_channel_offset = ch * dst_offset;

					for (int row = 0; row < dst_height; ++row)
					{
						int src_index = src_channel_offset + (row + top) * width + left;
						int dst_index = dst_channel_offset + row * dst_width;
						cudaMemcpy(dst_data + dst_index, src_data + src_index, dst_width * sizeof(Dtype), cudaMemcpyDefault);
					}
				}
			}
			else if (src->order() == NHWC)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, dst_height, dst_width, channels}, src->device(), src->order()));
				Dtype* dst_data = dst->mutable_gpu_data();
				const Dtype* src_data = src->gpu_data();

				for (int row = 0; row < dst_height; ++row)
				{
					int src_index = ((row + top) * width + left) * channels;
					int dst_index = row * dst_width * channels;
					cudaMemcpy(dst_data + dst_index, src_data + src_index, dst_width * channels * sizeof(Dtype), cudaMemcpyDefault);
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}



		/// <summary>
		/// 根据指定的矩形裁剪图像（类似于ROI），如果矩形超出图像边界，则用0填充对应像素值
		/// </summary>
		/// <param name="src">包含原图像数据的tensor</param>
		/// <param name="dst">裁剪后的tensor，包含了rect范围内的图像数据</param>
		/// <param name="rect">指定的矩形</param>
		template <typename Dtype, typename Rtype>
		void tensor_operation_gpu::safty_cut_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, rectangle<Rtype>* rect)
		{
			if (rect->x >= 0 && rect->y >= 0 && (rect->x + rect->w <= src->width()) && (rect->y + rect->h <= src->height()))
			{
				cut_border_gpu(src, dst, rect->y, (src->height() - rect->y - rect->h), rect->x, (src->width() - rect->x - rect->w));
			}
			else
			{
				int top = std::max(int(0), int(-1 * rect->y));
				int bottom = std::max(int(rect->y + rect->h - src->height()), int(0));
				int left = std::max(int(0), int(-1 * rect->x));
				int right = std::max(int(rect->x + rect->w - src->width()), int(0));
				std::shared_ptr<tensor<Dtype>> temp;
				if (src->order() == NCHW)
				{
					temp.reset(new tensor<Dtype>(
						std::vector<int>{src->num(), src->channels(), src->height() + top + bottom, src->width() + left + right},
						src->device(), src->order()));
				}
				else if (src->order() == NHWC)
				{
					temp.reset(new tensor<Dtype>(
						std::vector<int>{src->num(), src->height() + top + bottom, src->width() + left + right, src->channels()},
						src->device(), src->order()));
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				make_border_gpu(src, temp, top, bottom, left, right, Border_Constant);
				cut_border_gpu(temp, dst, rect->y + top, temp->height() - rect->y - rect->h - top, rect->x + left, temp->width() - rect->x - rect->w - left);
			}
		}



		/// <summary>
		/// tensor预处理，按通道对每个像素点进行相同的乘法和加法变换
		/// </summary>
		/// <param name="dst_data">输入及输出的图像数据</param>
		/// <param name="channels">图像通道数</param>
		/// <param name="height">图像高度</param>
		/// <param name="width">图像宽度</param>
		/// <param name="order">tensor类型：NHWC(tensor按NHWC排列)、NCHW(tensor按NCHW排列)</param>
		template<typename Dtype>
		__global__
			void kernel_equalize_hist(int offset, const Dtype* src_data,  Dtype* dst_data, int height, int width, 
				int* gray_value, int* normalized_gray_value, float* probability_distribution, float* accumulate_probability_distribution)
		{
			unsigned int *s_h;
			cudaMalloc(&s_h, 256 * sizeof(unsigned int));
			
			for (int i = threadIdx.x; i < 256; i += blockDim.x)
			{
				s_h[i] = 0;
			}
			__syncthreads();


			for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < offset; i += blockDim.x * gridDim.x)
			{
				unsigned char index = src_data[i];
				atomicInc(s_h + index, 0xffffffff);
			}
			__syncthreads();

			for (int i = threadIdx.x; i < 256; i += blockDim.x)
			{
				atomicAdd(gray_value + i, s_h[i]);
			}

			for (int i = 0; i < 256; i++)
			{
				//计算概率密度分布
				probability_distribution[i] = static_cast<float>(gray_value[i]) / offset;

				//计算累积概率密度分布
				if (i > 0)
				{
					accumulate_probability_distribution[i] = accumulate_probability_distribution[i - 1] + probability_distribution[i];
				}
				else
				{
					accumulate_probability_distribution[0] = probability_distribution[0];
				}

				//计算归一化后的像素值
				normalized_gray_value[i] = static_cast<unsigned char>(255 * accumulate_probability_distribution[i] + 0.5);
			}

			for (size_t i = 0; i < offset; i++)
			{
				dst_data[i] = Dtype(normalized_gray_value[static_cast<unsigned char>(src_data[i])]);
			}
		}

		///// <summary>
		///// 直方图均衡化
		///// </summary>
		///// <param name="src">包含原图像数据的tensor</param>
		///// <param name="dst">直方图均衡化后的tensor</param>
		//template <typename Dtype>
		//void tensor_operation_gpu::equalize_hist_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst)
		//{
		//	CHECK_EQ(src->num(), 1);
		//	CHECK_EQ(src->channels(), 1);
		//	int height = src->height();
		//	int width = src->width();
		//	int offset = height * width;

		//	if (dst->count() != src->count())
		//	{
		//		dst.reset(new tensor<Dtype>(std::vector<int>{1, 1, height, width}, src->device()));
		//	}
		//	const Dtype* src_data = src->gpu_data();
		//	Dtype* dst_data = dst->mutable_gpu_data();

		//	int *gray_value, *normalized_gray_value;
		//	float *probability_distribution, *accumulate_probability_distribution;

		//	cudaMalloc(&gray_value, 256 * sizeof(int));
		//	cudaMalloc(&normalized_gray_value, 256 * sizeof(int));
		//	cudaMalloc(&probability_distribution, 256 * sizeof(float));
		//	cudaMalloc(&accumulate_probability_distribution, 256 * sizeof(float));
		//	cudaMemset(&gray_value, 0, 256 * sizeof(int));
		//	cudaMemset(&normalized_gray_value, 0, 256 * sizeof(int));
		//	cudaMemset(&probability_distribution, 0, 256 * sizeof(float));
		//	cudaMemset(&accumulate_probability_distribution, 0, 256 * sizeof(float));

		//	//按照设置的blockSize和gridSize启动内核函数
		//	kernel_equalize_hist << <CUDA_GET_BLOCKS(offset),   >> >(
		//		offset, src_data, dst_data, height, width, 
		//		gray_value, normalized_gray_value, probability_distribution, accumulate_probability_distribution);
		//}


		/// <summary>
		/// 直方图均衡化
		/// </summary>
		/// <param name="src">包含原图像数据的tensor</param>
		/// <param name="dst">直方图均衡化后的tensor</param>
		template <typename Dtype>
		void tensor_operation_gpu::equalize_hist_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst)
		{
			CHECK_EQ(src->num(), 1);
			CHECK_EQ(src->channels(), 1);
			int height = src->height();
			int width = src->width();
			int offset = height * width;
			if (dst->count() != src->count())
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, 1, height, width}, src->device()));
			}
			const Dtype* src_data = src->gpu_data();
			Dtype* dst_data = dst->mutable_gpu_data();

			unsigned char gray_value[256] = { 0 };//灰度值
			float probability_distribution[256] = { 0 };//概率密度分布
			float accumulate_probability_distribution[256] = { 0 };//累积概率密度
			unsigned char normalized_gray_value[256] = { 0 };//归一化后的灰度值
			Dtype* temp_dst = new Dtype[offset];

			//Count the number of pixels in each grayscale
			for (size_t i = 0; i < offset; i++)
			{
				int value = static_cast<unsigned char>(src->cpu_data()[i]);
				gray_value[value]++;
			}

			for (int i = 0; i < 256; i++)
			{
				//计算概率密度分布
				probability_distribution[i] = static_cast<float>(gray_value[i]) / offset;

				//计算累积概率密度分布
				if (i > 0)
				{
					accumulate_probability_distribution[i] = accumulate_probability_distribution[i - 1] + probability_distribution[i];
				}
				else
				{
					accumulate_probability_distribution[0] = probability_distribution[0];
				}

				//计算归一化后的像素值
				normalized_gray_value[i] = static_cast<unsigned char>(255 * accumulate_probability_distribution[i] + 0.5);
			}

			for (size_t i = 0; i < offset; i++)
			{
				temp_dst[i] = Dtype(normalized_gray_value[static_cast<unsigned char>(src->cpu_data()[i])]);
			}

			cudaMemcpy(dst_data, temp_dst, offset * sizeof(Dtype), cudaMemcpyDefault);
			delete temp_dst;
		}


		/// <summary>
		/// 将多个tensor图像数据按vector索引下标顺序堆叠在一起，各tensor图像之间仅需保证宽度、高度、GPU/CPU、排列方式相同；每个tensor可以有不同的图像数量和通道数量
		/// </summary>
		/// <param name="src_vector">一组tensor，每个tensor中包含了宽度、高度、GPU/CPU、排列方式相同的图像数据</param>
		/// <param name="dst">堆叠后产生的tensor，按照vector索引下标顺序堆叠</param>
		template <typename Dtype>
		void tensor_operation_gpu::merge_channel_gpu(const std::vector<std::shared_ptr<tensor<Dtype>>> &src_vector, std::shared_ptr<tensor<Dtype>> &dst)
		{
			CHECK_EQ(src_vector.size(), 3);
			int height, width, device;
			orderType type;
			for (int i = 0; i < src_vector.size(); ++i)
			{
				CHECK_EQ(src_vector.at(i)->channels(), 1);
				if (i == 0)
				{
					height = src_vector.at(i)->height();
					width = src_vector.at(i)->width();
					device = src_vector.at(i)->device();
					type = src_vector.at(i)->order();
				}
				else
				{
					if (height != src_vector.at(i)->height() ||
						width != src_vector.at(i)->width() ||
						device != src_vector.at(i)->device() ||
						type != src_vector.at(i)->order())
					{
						LOG(WARNING) << "the element of vector<mat> should have the exact same height/width/device/type.";
						return;
					}
				}
			}

			if (type == NCHW)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, 3, height, width}, device, type));
			}
			else if (type == NHWC)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, 3}, device, type));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			Dtype* dst_data = dst->mutable_gpu_data();
			int offset = height * width;

			for (int i = 0; i < src_vector.size(); ++i)
			{
				const Dtype* temp_data = src_vector.at(i)->gpu_data();
				cudaMemcpy((void*)(dst_data + i * offset), (void*)(temp_data), offset * sizeof(Dtype), cudaMemcpyDefault);
			}
		}


		template void tensor_operation_gpu::make_border_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst,
			int top, int bottom, int left, int right, borderType type = Border_Constant, int fill = 0);
		template void tensor_operation_gpu::make_border_gpu<char>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>>& dst,
			int top, int bottom, int left, int right, borderType type = Border_Constant, int fill = 0);
		template void tensor_operation_gpu::make_border_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>>& dst,
			int top, int bottom, int left, int right, borderType type = Border_Constant, int fill = 0);
		template void tensor_operation_gpu::make_border_gpu<int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>>& dst,
			int top, int bottom, int left, int right, borderType type = Border_Constant, int fill = 0);
		template void tensor_operation_gpu::make_border_gpu<float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>>& dst,
			int top, int bottom, int left, int right, borderType type = Border_Constant, int fill = 0);



		template void tensor_operation_gpu::cut_border_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src,
			std::shared_ptr<tensor<unsigned char>>& dst, int top, int bottom, int left, int right);
		template void tensor_operation_gpu::cut_border_gpu<char>(const std::shared_ptr<tensor<char>> &src,
			std::shared_ptr<tensor<char>>& dst, int top, int bottom, int left, int right);
		template void tensor_operation_gpu::cut_border_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src,
			std::shared_ptr<tensor<unsigned int>>& dst, int top, int bottom, int left, int right);
		template void tensor_operation_gpu::cut_border_gpu<int>(const std::shared_ptr<tensor<int>> &src,
			std::shared_ptr<tensor<int>>& dst, int top, int bottom, int left, int right);
		template void tensor_operation_gpu::cut_border_gpu<float>(const std::shared_ptr<tensor<float>> &src,
			std::shared_ptr<tensor<float>>& dst, int top, int bottom, int left, int right);



		template void tensor_operation_gpu::safty_cut_gpu<unsigned char, int>(const std::shared_ptr<tensor<unsigned char>> &src, 
			std::shared_ptr<tensor<unsigned char>> &dst, rectangle<int>* rect);
		template void tensor_operation_gpu::safty_cut_gpu<char, int>(const std::shared_ptr<tensor<char>> &src,
			std::shared_ptr<tensor<char>> &dst, rectangle<int>* rect);
		template void tensor_operation_gpu::safty_cut_gpu<unsigned int, int>(const std::shared_ptr<tensor<unsigned int>> &src,
			std::shared_ptr<tensor<unsigned int>> &dst, rectangle<int>* rect);
		template void tensor_operation_gpu::safty_cut_gpu<int, int>(const std::shared_ptr<tensor<int>> &src,
			std::shared_ptr<tensor<int>> &dst, rectangle<int>* rect);
		template void tensor_operation_gpu::safty_cut_gpu<float, int>(const std::shared_ptr<tensor<float>> &src,
			std::shared_ptr<tensor<float>> &dst, rectangle<int>* rect);



		template void tensor_operation_gpu::safty_cut_gpu<unsigned char, float>(const std::shared_ptr<tensor<unsigned char>> &src,
			std::shared_ptr<tensor<unsigned char>> &dst, rectangle<float>* rect);
		template void tensor_operation_gpu::safty_cut_gpu<char, float>(const std::shared_ptr<tensor<char>> &src,
			std::shared_ptr<tensor<char>> &dst, rectangle<float>* rect);
		template void tensor_operation_gpu::safty_cut_gpu<unsigned int, float>(const std::shared_ptr<tensor<unsigned int>> &src,
			std::shared_ptr<tensor<unsigned int>> &dst, rectangle<float>* rect);
		template void tensor_operation_gpu::safty_cut_gpu<int, float>(const std::shared_ptr<tensor<int>> &src,
			std::shared_ptr<tensor<int>> &dst, rectangle<float>* rect);
		template void tensor_operation_gpu::safty_cut_gpu<float, float>(const std::shared_ptr<tensor<float>> &src,
			std::shared_ptr<tensor<float>> &dst, rectangle<float>* rect);


		template void tensor_operation_gpu::equalize_hist_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst);
		template void tensor_operation_gpu::equalize_hist_gpu<char>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>>& dst);
		template void tensor_operation_gpu::equalize_hist_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>>& dst);
		template void tensor_operation_gpu::equalize_hist_gpu<int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>>& dst);
		template void tensor_operation_gpu::equalize_hist_gpu<float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>>& dst);



		template void tensor_operation_gpu::merge_channel_gpu<unsigned char>(const std::vector<std::shared_ptr<tensor<unsigned char>>> &src_vector, std::shared_ptr<tensor<unsigned char>> &dst);
		template void tensor_operation_gpu::merge_channel_gpu<char>(const std::vector<std::shared_ptr<tensor<char>>> &src_vector, std::shared_ptr<tensor<char>> &dst);
		template void tensor_operation_gpu::merge_channel_gpu<unsigned int>(const std::vector<std::shared_ptr<tensor<unsigned int>>> &src_vector, std::shared_ptr<tensor<unsigned int>> &dst);
		template void tensor_operation_gpu::merge_channel_gpu<int>(const std::vector<std::shared_ptr<tensor<int>>> &src_vector, std::shared_ptr<tensor<int>> &dst);
		template void tensor_operation_gpu::merge_channel_gpu<float>(const std::vector<std::shared_ptr<tensor<float>>> &src_vector, std::shared_ptr<tensor<float>> &dst);



		template void tensor_operation_gpu::preprocess_tensors_gpu<unsigned char>(std::shared_ptr<tensor<unsigned char>> dst);
		template void tensor_operation_gpu::preprocess_tensors_gpu<char>(std::shared_ptr<tensor<char>> dst);
		template void tensor_operation_gpu::preprocess_tensors_gpu<unsigned int>(std::shared_ptr<tensor<unsigned int>> dst);
		template void tensor_operation_gpu::preprocess_tensors_gpu<int>(std::shared_ptr<tensor<int>> dst);
		template void tensor_operation_gpu::preprocess_tensors_gpu<float>(std::shared_ptr<tensor<float>> dst);



		template void tensor_operation_gpu::preprocess_tensors_gpu<unsigned char>(tensor<unsigned char>* dst);
		template void tensor_operation_gpu::preprocess_tensors_gpu<char>(tensor<char>* dst);
		template void tensor_operation_gpu::preprocess_tensors_gpu<unsigned int>(tensor<unsigned int>* dst);
		template void tensor_operation_gpu::preprocess_tensors_gpu<int>(tensor<int>* dst);
		template void tensor_operation_gpu::preprocess_tensors_gpu<float>(tensor<float>* dst);



		template void tensor_operation_gpu::type_converter_gpu<unsigned char, unsigned int>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned int>> &dst);
		template void tensor_operation_gpu::type_converter_gpu<unsigned char, int>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<int>> &dst);
		template void tensor_operation_gpu::type_converter_gpu<unsigned char, float>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<float>> &dst);
		template void tensor_operation_gpu::type_converter_gpu<float, unsigned char>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<unsigned char>> &dst);
		template void tensor_operation_gpu::type_converter_gpu<float, int>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<int>> &dst);
		template void tensor_operation_gpu::type_converter_gpu<float, unsigned int>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<unsigned int>> &dst);



		template void tensor_operation_gpu::type_converter_gpu<unsigned char, unsigned int>(const tensor<unsigned char> *src, tensor<unsigned int> *dst);
		template void tensor_operation_gpu::type_converter_gpu<unsigned char, int>(const tensor<unsigned char> *src, tensor<int> *dst);
		template void tensor_operation_gpu::type_converter_gpu<unsigned char, float>(const tensor<unsigned char> *src, tensor<float> *dst);
		template void tensor_operation_gpu::type_converter_gpu<float, unsigned char>(const tensor<float> *src, tensor<unsigned char> *dst);
		template void tensor_operation_gpu::type_converter_gpu<float, int>(const tensor<float> *src, tensor<int> *dst);
		template void tensor_operation_gpu::type_converter_gpu<float, unsigned int>(const tensor<float> *src, tensor<unsigned int> *dst);

#ifdef USE_OPENCV
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


		template void tensor_operation_gpu::mat2tensor_gpu<unsigned char>(const cv::Mat &src, std::shared_ptr<tensor<unsigned char>>& dst, orderType order = NHWC);
		template void tensor_operation_gpu::mat2tensor_gpu<char>(const cv::Mat &src, std::shared_ptr<tensor<char>>& dst, orderType order = NHWC);
		template void tensor_operation_gpu::mat2tensor_gpu<unsigned int>(const cv::Mat &src, std::shared_ptr<tensor<unsigned int>>& dst, orderType order = NHWC);
		template void tensor_operation_gpu::mat2tensor_gpu<int>(const cv::Mat &src, std::shared_ptr<tensor<int>>& dst, orderType order = NHWC);
		template void tensor_operation_gpu::mat2tensor_gpu<float>(const cv::Mat &src, std::shared_ptr<tensor<float>>& dst, orderType order = NHWC);


		template void tensor_operation_gpu::mat2tensor_gpu<unsigned char>(const cv::Mat &src, tensor<unsigned char>& dst, orderType order = NHWC);
		template void tensor_operation_gpu::mat2tensor_gpu<char>(const cv::Mat &src, tensor<char>& dst, orderType order = NHWC);
		template void tensor_operation_gpu::mat2tensor_gpu<unsigned int>(const cv::Mat &src, tensor<unsigned int>& dst, orderType order = NHWC);
		template void tensor_operation_gpu::mat2tensor_gpu<int>(const cv::Mat &src, tensor<int>& dst, orderType order = NHWC);
		template void tensor_operation_gpu::mat2tensor_gpu<float>(const cv::Mat &src, tensor<float>& dst, orderType order = NHWC);
#endif // USE_OPENCV

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
	}
}
#endif // USE_CUDA