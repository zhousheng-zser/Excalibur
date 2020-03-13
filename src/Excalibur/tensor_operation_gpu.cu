#ifdef USE_CUDA
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <glasssix/tensor.hpp>
#include "math_functions.hpp"
#include <iostream>
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif
#include <glasssix/timer.hpp>
#include "tensor_operation_gpu.hpp"
#include <algorithm>

#define PI 3.1415926
extern const unsigned char LBPMAP[][256];
namespace glasssix
{
	namespace excalibur
	{
		/// <summary>
		/// get depth according to data type
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
		/// convert NHWC data to NCHW data
		/// </summary>
		/// <param name="src_data">NHWC data</param>
		/// <param name="dst_data">NCHW data</param>
		/// <param name="channels">image channel</param>
		/// <param name="height">image height</param>
		/// <param name="width">image width</param>
		template<typename Dtype>
		__global__
			void kernel_nhwc2nchw(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width)
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int channelID = totalID % channels;
			int offset = channels * height * width;
			int numID = totalID / offset;
			int remainID = totalID % offset;
			remainID /= channels;
			int rowID = remainID / width;
			int colID = remainID % width;
			int num_offset = numID * channels * height * width;

			dst_data[num_offset + channelID * height * width + rowID * width + colID] = src_data[num_offset + (rowID * width + colID) * channels + channelID];
		}



		/// <summary>
		/// convert NCHW data to NHWC data
		/// </summary>
		/// <param name="src_data">NCHW data</param>
		/// <param name="dst_data">NHWC data</param>
		/// <param name="channels">image channel</param>
		/// <param name="height">image height</param>
		/// <param name="width">image width</param>
		template<typename Dtype>
		__global__
			void kernel_nchw2nhwc(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width)
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int channelID = totalID % channels;
			int offset = channels * height * width;
			int numID = totalID / offset;
			int remainID = totalID % offset;
			remainID /= channels;
			int rowID = remainID / width;
			int colID = remainID % width;
			int num_offset = numID * channels * height * width;

			dst_data[num_offset + (rowID * width + colID) * channels + channelID] = src_data[num_offset + channelID * height * width + rowID * width + colID];
		}



#ifdef USE_OPENCV

		/// <summary>
		/// transfer tensor to mat
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new mat</param>
		template<typename Dtype>
		void tensor_operation_gpu::tensor2mat_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::vector<cv::Mat>& dst)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src->num();
			int channels = src->channels();
			if (channels>4)
			{
				LOG(ERROR) << "Too many channels.";
				return;
			}
			int width = src->width();
			int height = src->height();
			int num_offset = channels * height * width;
			int type = get_cv_type<Dtype>();
			if (type<0)
			{
				LOG(ERROR) << "Un-support data type.";
				return;
			}
			
			const Dtype* src_data = src->gpu_data();
			dst.clear();

			if (src->order() == NHWC)
			{
				for (size_t n = 0; n < num; n++)
				{
					int n_offset = n * num_offset;
					cv::Mat temp = cv::Mat(height, width, CV_MAKETYPE(type, channels));
					CUDA_CHECK(cudaMemcpy(temp.data, src->gpu_data() + n_offset, height * width * channels * sizeof(Dtype), cudaMemcpyDefault));
					dst.push_back(temp);
				}
				
				return;
			}

			std::shared_ptr<tensor<Dtype>> dst_ptr = std::make_shared<tensor<Dtype>>(tensor<Dtype>(std::vector<int>{num, height, width, channels}, src->device(), NHWC));

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_nchw2nhwc << <grid_size, block_size >> > (src->gpu_data(), dst_ptr->mutable_gpu_data(), channels, height, width);

			for (size_t n = 0; n < num; n++)
			{
				int n_offset = n * num_offset;
				cv::Mat temp = cv::Mat(height, width, CV_MAKETYPE(type, channels));
				CUDA_CHECK(cudaMemcpy(temp.data, dst_ptr->gpu_data() + n_offset, height * width * channels * sizeof(Dtype), cudaMemcpyDefault));
				dst.push_back(temp);
			}
		}



		/// <summary>
		/// transfer tensor to mat
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new mat</param>
		template<typename Dtype>
		void tensor_operation_gpu::tensor2mat_gpu(const tensor<Dtype>& src, std::vector<cv::Mat>& dst)
		{
			if (src.device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src.num();
			int channels = src.channels();
			if (channels>4)
			{
				LOG(ERROR) << "Too many channels.";
				return;
			}
			int width = src.width();
			int height = src.height();
			int num_offset = channels * height * width;
			int type = get_cv_type<Dtype>();
			if (type<0)
			{
				LOG(ERROR) << "Un-support data type.";
				return;
			}
			
			const Dtype* src_data = src.gpu_data();
			dst.clear();

			if (src.order() == NHWC)
			{
				for (size_t n = 0; n < num; n++)
				{
					int n_offset = n * num_offset;
					cv::Mat temp = cv::Mat(height, width, CV_MAKETYPE(type, channels));
					CUDA_CHECK(cudaMemcpy(temp.data, src.gpu_data() + n_offset, height * width * channels * sizeof(Dtype), cudaMemcpyDefault));
					dst.push_back(temp);
				}
				
				return;
			}

			std::shared_ptr<tensor<Dtype>> dst_ptr = std::make_shared<tensor<Dtype>>(tensor<Dtype>(std::vector<int>{num, height, width, channels}, src.device(), NHWC));

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_nchw2nhwc << <grid_size, block_size >> > (src.gpu_data(), dst_ptr->mutable_gpu_data(), channels, height, width);

			for (size_t n = 0; n < num; n++)
			{
				int n_offset = n * num_offset;
				cv::Mat temp = cv::Mat(height, width, CV_MAKETYPE(type, channels));
				CUDA_CHECK(cudaMemcpy(temp.data, dst_ptr->gpu_data() + n_offset, height * width * channels * sizeof(Dtype), cudaMemcpyDefault));
				dst.push_back(temp);
			}
		}



		/// <summary>
		/// transfer mat to tensor
		/// </summary>
		/// <param name="src">original mat</param>
		/// <param name="dst">new tensor</param>
		/// <param name="order">order type of new tensor: NCHW / NHWC(default)</param>
		template<typename Dtype>
		void tensor_operation_gpu::mat2tensor_gpu(const cv::Mat &src, std::shared_ptr<tensor<Dtype>>& dst, orderType order)
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

#ifdef _MSC_VER

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

#elif defined __linux__

			if (type_id == 0)
			{
				if (type_name != std::string("h"))
				{
					LOG(ERROR) << "Un-matched data type.";
					return;
				}
			}
			else if (type_id == 1)
			{
				if (type_name != std::string("c"))
				{
					LOG(ERROR) << "Un-matched data type.";
					return;
				}
			}
			else if (type_id == 4)
			{
				if (type_name != std::string("i"))
				{
					LOG(ERROR) << "Un-matched data type.";
					return;
				}
			}
			else if (type_id == 5)
			{
				if (type_name != std::string("f"))
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

#else
			NOT_IMPLEMENTED;
			return;
#endif // _MSC_VER

			if (order == NHWC)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, channels}, 0, order));
				CUDA_CHECK(cudaMemcpy(dst->mutable_gpu_data(), src.data, height * width * channels * sizeof(Dtype), cudaMemcpyDefault));
				return;
			}

			dst.reset(new tensor<Dtype>(std::vector<int>{1, channels, height, width}, 0, order));
			std::shared_ptr<tensor<Dtype>> src_ptr = std::make_shared<tensor<Dtype>>(tensor<Dtype>(std::vector<int>{1, channels, height, width}, 0, order));
			CUDA_CHECK(cudaMemcpy(src_ptr->mutable_gpu_data(), src.data, height * width * channels * sizeof(Dtype), cudaMemcpyDefault));

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, 1);

			
			kernel_nhwc2nchw << <grid_size, block_size >> > (src_ptr->gpu_data(), dst->mutable_gpu_data(), channels, height, width);
		}



		/// <summary>
		/// transfer mat to tensor
		/// </summary>
		/// <param name="src">original mat</param>
		/// <param name="dst">new tensor</param>
		/// <param name="order">order type of new tensor: NCHW / NHWC(default)</param>
		template<typename Dtype>
		void tensor_operation_gpu::mat2tensor_gpu(const cv::Mat &src, tensor<Dtype>& dst, orderType order)
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

#ifdef _MSC_VER

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

#elif defined __linux__

			if (type_id == 0)
			{
				if (type_name != std::string("h"))
				{
					LOG(ERROR) << "Un-matched data type.";
					return;
				}
			}
			else if (type_id == 1)
			{
				if (type_name != std::string("c"))
				{
					LOG(ERROR) << "Un-matched data type.";
					return;
				}
			}
			else if (type_id == 4)
			{
				if (type_name != std::string("i"))
				{
					LOG(ERROR) << "Un-matched data type.";
					return;
				}
			}
			else if (type_id == 5)
			{
				if (type_name != std::string("f"))
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

#else
			NOT_IMPLEMENTED;
			return;
#endif // _MSC_VER

			if (order == NHWC)
			{
				dst = tensor<Dtype>(std::vector<int>{1, height, width, channels}, 0, order);
				CUDA_CHECK(cudaMemcpy(dst.mutable_gpu_data(), src.data, height * width * channels * sizeof(Dtype), cudaMemcpyDefault));
				return;
			}

			dst = tensor<Dtype>(std::vector<int>{1, channels, height, width}, 0, order);
			std::shared_ptr<tensor<Dtype>> src_ptr = std::make_shared<tensor<Dtype>>(tensor<Dtype>(std::vector<int>{1, channels, height, width}, 0, order));
			CUDA_CHECK(cudaMemcpy(src_ptr->mutable_gpu_data(), src.data, height * width * channels * sizeof(Dtype), cudaMemcpyDefault));

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, 1);

			
			kernel_nhwc2nchw << <grid_size, block_size >> > (src_ptr->gpu_data(), dst.mutable_gpu_data(), channels, height, width);
		}

#endif



		/// <summary>
		/// convert tensor(NCHW) to tensor(NHWC)
		/// </summary>
		/// <param name="src">original tensor(NCHW)</param>
		/// <param name="dst">new tensor(NHWC)</param>
		template<typename Dtype>
		void tensor_operation_gpu::nchw2nhwc_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			CHECK_EQ(src->order(), NCHW);
			int num = src->num();
			int height = src->height();
			int width = src->width();
			int channels = src->channels();
			int offset = height * width;

			std::shared_ptr<tensor<Dtype>> dst_temp;
			dst_temp.reset(new tensor<Dtype>(std::vector<int>{num, height, width, channels}, src->device(), NHWC));

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_nchw2nhwc << <grid_size, block_size >> > (src->gpu_data(), dst_temp->mutable_gpu_data(), channels, height, width);
			dst = std::make_shared<tensor<Dtype>>(dst_temp->clone());
		}



		/// <summary>
		/// convert tensor(NCHW) to tensor(NHWC)
		/// </summary>
		/// <param name="src">original tensor(NCHW)</param>
		/// <param name="dst">new tensor(NHWC)</param>
		template<typename Dtype>
		void tensor_operation_gpu::nchw2nhwc_gpu(const tensor<Dtype> &src, tensor<Dtype> &dst)
		{
			if (src.device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			CHECK_EQ(src.order(), NCHW);
			int num = src.num();
			int height = src.height();
			int width = src.width();
			int channels = src.channels();
			int offset = height * width;

			tensor<Dtype> dst_temp = tensor<Dtype>(std::vector<int>{num, height, width, channels}, src.device(), NHWC);

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_nchw2nhwc << <grid_size, block_size >> > (src.gpu_data(), dst_temp.mutable_gpu_data(), channels, height, width);
			dst = dst_temp.clone();
		}



		/// <summary>
		/// convert tensor(NHWC) to tensor(NCHW)
		/// </summary>
		/// <param name="src">original tensor(NHWC)</param>
		/// <param name="dst">new tensor(NCHW)</param>
		template<typename Dtype>
		void tensor_operation_gpu::nhwc2nchw_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			CHECK_EQ(src->order(), NHWC);
			int num = src->num();
			int height = src->height();
			int width = src->width();
			int channels = src->channels();
			int offset = height * width;

			std::shared_ptr<tensor<Dtype>> dst_temp;
			dst_temp.reset(new tensor<Dtype>(std::vector<int>{num, channels, height, width}, src->device(), NCHW));

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_nhwc2nchw << <grid_size, block_size >> > (src->gpu_data(), dst_temp->mutable_gpu_data(), channels, height, width);
			dst = std::make_shared<tensor<Dtype>>(dst_temp->clone());
		}



		/// <summary>
		/// convert tensor(NHWC) to tensor(NCHW)
		/// </summary>
		/// <param name="src">original tensor(NHWC)</param>
		/// <param name="dst">new tensor(NCHW)</param>
		template<typename Dtype>
		void tensor_operation_gpu::nhwc2nchw_gpu(const tensor<Dtype> &src, tensor<Dtype> &dst)
		{
			if (src.device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			CHECK_EQ(src.order(), NHWC);
			int num = src.num();
			int height = src.height();
			int width = src.width();
			int channels = src.channels();
			int offset = height * width;

			tensor<Dtype> dst_temp = tensor<Dtype>(std::vector<int>{num, channels, height, width}, src.device(), NCHW);

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_nhwc2nchw << <grid_size, block_size >> > (src.gpu_data(), dst_temp.mutable_gpu_data(), channels, height, width);
			dst = dst_temp.clone();
		}



		/// <summary>
		/// resize image data
		/// </summary>
		/// <param name="channels">image channel</param>
		/// <param name="src_data">original data</param>
		/// <param name="src_height">original height</param>
		/// <param name="src_width">original width</param>
		/// <param name="dst_data">resized image data</param>
		/// <param name="dst_height">new height</param>
		/// <param name="dst_width">new width</param>
		/// <param name="type">interpolationType: Nearest / Bilinear</param>
		/// <param name="order">orderType: NHWC / NCHW</param>
		template<typename Dtype>
		__global__
			void kernel_resize(int channels, const Dtype* src_data, int src_height, int src_width,
				Dtype* dst_data, int dst_height, int dst_width,
				interpolationType type, orderType order)
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int channelID = totalID % channels;
			int dst_offset = channels * dst_height * dst_width;
			int numID = totalID / dst_offset;
			int remainID = totalID % dst_offset;
			remainID /= channels;
			int rowID = remainID / dst_width;
			int colID = remainID % dst_width;
			int dst_num_offset = numID * channels * dst_height * dst_width;
			int src_num_offset = numID * channels * src_height * src_width;
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
				int src_channel_offset = channelID * src_height * src_width;
				int dst_channel_offset = channelID * dst_height * dst_width;
				int src_index = src_channel_offset + y * src_width + x;
				int dst_index = dst_channel_offset + rowID * dst_width + colID;

				if (type == Nearest)
				{
					dst_data[dst_num_offset + dst_index] = src_data[src_num_offset + src_index];
				}
				else if (type == Bilinear)
				{
					unsigned indexA = min(unsigned(src_index), maxIndex);
					unsigned indexB = min(unsigned(src_index + 1), maxIndex);
					unsigned indexC = min(unsigned(src_index + src_width), maxIndex);
					unsigned indexD = min(unsigned(src_index + src_width + 1), maxIndex);
					Dtype A = src_data[src_num_offset + indexA];
					Dtype B = src_data[src_num_offset + indexB];
					Dtype C = src_data[src_num_offset + indexC];
					Dtype D = src_data[src_num_offset + indexD];

					dst_data[dst_num_offset + dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
						static_cast<float>(B) * xdiff * (1 - ydiff) +
						static_cast<float>(C) * ydiff * (1 - xdiff) +
						static_cast<float>(D) * xdiff * ydiff);
				}
			}
			else if (order == NHWC)
			{
				int src_index = (y * src_width + x) * channels + channelID;
				int dst_index = (rowID * dst_width + colID) * channels + channelID;

				if (type == Nearest)
				{
					dst_data[dst_num_offset + dst_index] = src_data[src_num_offset + src_index];
				}
				else if (type == Bilinear)
				{
					unsigned indexA = min(unsigned(src_index), maxIndex);
					unsigned indexB = min(unsigned(src_index + channels), maxIndex);
					unsigned indexC = min(unsigned(src_index + src_width * channels), maxIndex);
					unsigned indexD = min(unsigned(src_index + (src_width + 1) * channels), maxIndex);
					Dtype A = src_data[src_num_offset + indexA];
					Dtype B = src_data[src_num_offset + indexB];
					Dtype C = src_data[src_num_offset + indexC];
					Dtype D = src_data[src_num_offset + indexD];

					dst_data[dst_num_offset + dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
						static_cast<float>(B) * xdiff * (1 - ydiff) +
						static_cast<float>(C) * ydiff * (1 - xdiff) +
						static_cast<float>(D) * xdiff * ydiff);
				}
			}
			else
			{
				return;
			}
		}



		/// <summary>
		/// resize image data
		/// </summary>
		/// <param name="src">tensor of image with original size(height and width)</param>
		/// <param name="dst">tensor of image with new size</param>
		/// <param name="dst_height">new height</param>
		/// <param name="dst_width">new width</param>
		/// <param name="type">interpolationType: Nearest / Bilinear</param>
		template<typename Dtype>
		void tensor_operation_gpu::resize_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst,
			int dst_height, int dst_width, interpolationType type)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			if (dst_height * dst_width <= 0)
			{
				LOG(ERROR) << "Illegal input size.";
				return;
			}

			int num = src->num();
			int channels = src->channels();
			int height = src->height();
			int width = src->width();

			if (dst_width == width && dst_height == height)
			{
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				return;
			}

			std::shared_ptr<tensor<Dtype>> dst_temp;
			if (src->order() == NCHW)
			{
				dst_temp.reset(new tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src->device(), src->order()));
			}
			else if (src->order() == NHWC)
			{
				dst_temp.reset(new tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src->device(), src->order()));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			Dtype* dst_data = dst_temp->mutable_gpu_data();
			const Dtype* src_data = src->gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(dst_width, dst_height, num);

			
			kernel_resize << <grid_size, block_size >> > (channels, src_data, height, width, dst_data, dst_height, dst_width, type, src->order());
			dst = std::make_shared<tensor<Dtype>>(dst_temp->clone());
		}



		/// <summary>
		/// resize image data
		/// </summary>
		/// <param name="src">tensor of image with original size(height and width)</param>
		/// <param name="dst">tensor of image with new size</param>
		/// <param name="dst_height">new height</param>
		/// <param name="dst_width">new width</param>
		/// <param name="type">interpolationType: Nearest / Bilinear</param>
		template<typename Dtype>
		void tensor_operation_gpu::resize_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst,
			int dst_height, int dst_width, interpolationType type)
		{
			if (src.device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			if (dst_height * dst_width <= 0)
			{
				LOG(ERROR) << "Illegal input size.";
				return;
			}

			int num = src.num();
			int channels = src.channels();
			int height = src.height();
			int width = src.width();

			if (dst_width == width && dst_height == height)
			{
				dst = src.clone();
				return;
			}

			tensor<Dtype> dst_temp;
			if (src.order() == NCHW)
			{
				dst_temp = tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src.device(), src.order());
			}
			else if (src.order() == NHWC)
			{
				dst_temp = tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src.device(), src.order());
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			Dtype* dst_data = dst_temp.mutable_gpu_data();
			const Dtype* src_data = src.gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(dst_width, dst_height, num);

			
			kernel_resize << <grid_size, block_size >> > (channels, src_data, height, width, dst_data, dst_height, dst_width, type, src.order());
			dst = dst_temp.clone();
		}



		/// <summary>
		/// rotate around center point, height and width will be changed
		/// </summary>
		/// <param name="channels">image channel</param>
		/// <param name="src_data">original image data</param>
		/// <param name="height">original height</param>
		/// <param name="width">original width</param>
		/// <param name="dst_data">rotated image data</param>
		/// <param name="dst_height">new height</param>
		/// <param name="dst_width">new width</param>
		/// <param name="sina">sin(theta)</param>
		/// <param name="cosa">cos(theta) </param>
		/// <param name="varX">X offset</param>
		/// <param name="varY">Y offset</param>
		/// <param name="fill_pixel_value">pixel value to fill in the blank area</param>
		/// <param name="type">interpolationType: Nearest / Bilinear</param>
		/// <param name="order">orderType: NHWC / NCHW</param>
		template<typename Dtype>
		__global__
			void kernel_rotate_with_center(int channels, const Dtype* src_data, int height, int width,
				Dtype* dst_data, int dst_height, int dst_width,
				float sina, float cosa, float varX, float varY, int fill_pixel_value, interpolationType type, orderType order)
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int channelID = totalID % channels;
			int dst_offset = channels * dst_height * dst_width;
			int numID = totalID / dst_offset;
			int remainID = totalID % dst_offset;
			remainID /= channels;
			int rowID = remainID / dst_width;
			int colID = remainID % dst_width;
			unsigned maxIndex = height * width * channels - 1;
			int src_num_offset = numID * channels * height * width;
			int dst_num_offset = numID * channels * dst_height * dst_width;

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

				int src_channel_offset = channelID * src_offset;
				int dst_channel_offset = channelID * dst_offset;
				int src_index = src_channel_offset + y * width + x;
				int dst_index = dst_channel_offset + rowID * dst_width + colID;

				if (x >= width || x < 0 || y >= height || y < 0)
				{
					dst_data[dst_num_offset + dst_index] = (Dtype)fill_pixel_value;
				}
				else
				{
					unsigned indexA = min(unsigned(src_index), maxIndex);
					unsigned indexB = min(unsigned(src_index + 1), maxIndex);
					unsigned indexC = min(unsigned(src_index + width), maxIndex);
					unsigned indexD = min(unsigned(src_index + width + 1), maxIndex);
					Dtype A = src_data[src_num_offset + indexA];
					Dtype B = src_data[src_num_offset + indexB];
					Dtype C = src_data[src_num_offset + indexC];
					Dtype D = src_data[src_num_offset + indexD];

					dst_data[dst_num_offset + dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
						static_cast<float>(B) * xdiff * (1 - ydiff) +
						static_cast<float>(C) * ydiff * (1 - xdiff) +
						static_cast<float>(D) * xdiff * ydiff);
				}
			}
			else if (order == NHWC)
			{
				int src_pos1 = (y * width + x) * channels;
				int dst_pos1 = (rowID * dst_width + colID) * channels;

				int src_pos2 = src_pos1 + channelID;
				int dst_pos2 = dst_pos1 + channelID;

				if (x >= width || x < 0 || y >= height || y < 0)
				{
					dst_data[dst_num_offset + dst_pos2] = (Dtype)fill_pixel_value;
				}
				else
				{
					unsigned indexA = min(unsigned(src_pos2), maxIndex);
					unsigned indexB = min(unsigned(src_pos2 + channels), maxIndex);
					unsigned indexC = min(unsigned(src_pos2 + width * channels), maxIndex);
					unsigned indexD = min(unsigned(src_pos2 + (width + 1) * channels), maxIndex);
					Dtype A = src_data[src_num_offset + indexA];
					Dtype B = src_data[src_num_offset + indexB];
					Dtype C = src_data[src_num_offset + indexC];
					Dtype D = src_data[src_num_offset + indexD];

					dst_data[dst_num_offset + dst_pos2] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
						static_cast<float>(B) * xdiff * (1 - ydiff) +
						static_cast<float>(C) * ydiff * (1 - xdiff) +
						static_cast<float>(D) * xdiff * ydiff);
				}
			}
			else
			{
				return;
			}
		}



		/// <summary>
		/// rotate around center point, height and width will be changed
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		/// <param name="theta">rotation angle, anti-clockwise is positive</param>
		/// <param name="dst_height">new height after rotate</param>
		/// <param name="dst_width">new width after rotate</param>
		/// <param name="fill_pixel_value">pixel value to fill in the blank area, zero by default</param>
		/// <param name="type">interpolationType: Nearest / Bilinear</param>
		template<typename Dtype>
		void tensor_operation_gpu::rotate_with_center_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst,
			float theta, int &dst_height, int &dst_width, int fill_pixel_value, interpolationType type)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			if (fabs(theta) <= 1e-6)
			{
				dst = std::make_shared<excalibur::tensor<Dtype>>(src->clone());
			}

			int num = src->num();
			int channels = src->channels();
			int height = src->height();
			int width = src->width();

			float rad = -1 * theta*(PI / 180);//anti-clockwise is positive
			float cosa = cos(rad);
			float sina = sin(rad);

			dst_width = (int)(width * abs(cosa) + height * abs(sina));
			dst_height = (int)(width * abs(sina) + height * abs(cosa));

			float VarX = (float)(-dst_width * cosa / 2.0f - dst_height * sina / 2.0f + width / 2.0f);
			float VarY = (float)(dst_width * sina / 2.0f - dst_height * cosa / 2.0f + height / 2.0f);

			std::shared_ptr<tensor<Dtype>> dst_temp;
			if (src->order() == NCHW)
			{
				dst_temp.reset(new excalibur::tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src->device(), src->order()));
			}
			else if (src->order() == NHWC)
			{
				dst_temp.reset(new excalibur::tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src->device(), src->order()));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			Dtype* dst_data = dst_temp->mutable_gpu_data();
			const Dtype* src_data = src->gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(dst_width, dst_height, num);

			
			kernel_rotate_with_center << <grid_size, block_size >> > (channels, src_data, height, width, dst_data, dst_height, dst_width, sina, cosa, VarX, VarY, fill_pixel_value, type, src->order());
			dst = std::make_shared<tensor<Dtype>>(dst_temp->clone());
		}



		/// <summary>
		/// rotate around center point, height and width will be changed
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		/// <param name="theta">rotation angle, anti-clockwise is positive</param>
		/// <param name="dst_height">new height after rotate</param>
		/// <param name="dst_width">new width after rotate</param>
		/// <param name="fill_pixel_value">pixel value to fill in the blank area, zero by default</param>
		/// <param name="type">interpolationType: Nearest / Bilinear</param>
		template<typename Dtype>
		void tensor_operation_gpu::rotate_with_center_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst,
			float theta, int &dst_height, int &dst_width, int fill_pixel_value, interpolationType type)
		{
			if (src.device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			if (fabs(theta) <= 1e-6)
			{
				dst = src.clone();
			}

			int num = src.num();
			int channels = src.channels();
			int height = src.height();
			int width = src.width();

			float rad = -1 * theta*(PI / 180);//anti-clockwise is positive
			float cosa = cos(rad);
			float sina = sin(rad);

			dst_width = (int)(width * abs(cosa) + height * abs(sina));
			dst_height = (int)(width * abs(sina) + height * abs(cosa));

			float VarX = (float)(-dst_width * cosa / 2.0f - dst_height * sina / 2.0f + width / 2.0f);
			float VarY = (float)(dst_width * sina / 2.0f - dst_height * cosa / 2.0f + height / 2.0f);

			tensor<Dtype> dst_temp;
			if (src.order() == NCHW)
			{
				dst_temp = tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src.device(), src.order());
			}
			else if (src.order() == NHWC)
			{
				dst_temp = tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src.device(), src.order());
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			Dtype* dst_data = dst_temp.mutable_gpu_data();
			const Dtype* src_data = src.gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(dst_width, dst_height, num);

			
			kernel_rotate_with_center << <grid_size, block_size >> > (channels, src_data, height, width, dst_data, dst_height, dst_width, sina, cosa, VarX, VarY, fill_pixel_value, type, src.order());
			dst = dst_temp.clone();
		}






		/// <summary>
		/// rotate around any point, height and width will be constant
		/// </summary>
		/// <param name="src_data">original image data</param>
		/// <param name="dst_data">rotated image data</param>
		/// <param name="channels">image channel</param>
		/// <param name="height">image height</param>
		/// <param name="width">image width</param>
		/// <param name="M_data">transfomation matrix</param>
		/// <param name="fill_pixel_value">pixel value to fill in the blank area</param>
		/// <param name="type">interpolationType: Nearest / Bilinear</param>
		/// <param name="order">orderType: NHWC / NCHW</param>
		template<typename Dtype>
		__global__
			void kernel_rotate_with_points(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, 
				double* M_data, int fill_pixel_value, interpolationType type, orderType order)
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int channelID = totalID % channels;
			int offset = channels * height * width;
			int numID = totalID / offset;
			int remainID = totalID % offset;
			remainID /= channels;
			int rowID = remainID / width;
			int colID = remainID % width;
			unsigned maxIndex = height * width * channels - 1;
			int num_offset = numID * channels * height * width;

			double xf = M_data[0] * colID + M_data[1] * rowID + M_data[2];
			double yf = M_data[3] * colID + M_data[4] * rowID + M_data[5];
			int x = (int)xf;
			int y = (int)yf;
			float xdiff = xf - x;
			float ydiff = yf - y;

			if (order == NCHW)
			{
				int channel_offset = channelID * height * width;
				int src_index = channel_offset + y * width + x;
				int dst_index = channel_offset + rowID * width + colID;

				if (x < 0 || x >= width || y < 0 || y >= height)
				{
					dst_data[num_offset + dst_index] = (Dtype)fill_pixel_value;
				}
				else
				{
					if (type == excalibur::Nearest)
					{
						dst_data[num_offset + dst_index] = src_data[num_offset + src_index];
					}
					else if (type == excalibur::Bilinear)
					{
						unsigned indexA = min(unsigned(src_index), maxIndex);
						unsigned indexB = min(unsigned(src_index + 1), maxIndex);
						unsigned indexC = min(unsigned(src_index + width), maxIndex);
						unsigned indexD = min(unsigned(src_index + width + 1), maxIndex);
						Dtype A = src_data[num_offset + indexA];
						Dtype B = src_data[num_offset + indexB];
						Dtype C = src_data[num_offset + indexC];
						Dtype D = src_data[num_offset + indexD];

						dst_data[num_offset + dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
							static_cast<float>(B) * xdiff * (1 - ydiff) +
							static_cast<float>(C) * ydiff * (1 - xdiff) +
							static_cast<float>(D) * xdiff * ydiff);
					}
				}
			}
			else if (order == NHWC)
			{
				int src_index = (y * width + x) * channels + channelID;
				int dst_index = (rowID * width + colID) * channels + channelID;

				if (x < 0 || x >= width || y < 0 || y >= height)
				{
					dst_data[num_offset + dst_index] = (Dtype)fill_pixel_value;
				}
				else
				{
					if (type == excalibur::Nearest)
					{
						dst_data[num_offset + dst_index] = src_data[num_offset + src_index];
					}
					else if (type == excalibur::Bilinear)
					{
						unsigned indexA = min(unsigned(src_index), maxIndex);
						unsigned indexB = min(unsigned(src_index + channels), maxIndex);
						unsigned indexC = min(unsigned(src_index + width * channels), maxIndex);
						unsigned indexD = min(unsigned(src_index + (width + 1) * channels), maxIndex);
						Dtype A = src_data[num_offset + indexA];
						Dtype B = src_data[num_offset + indexB];
						Dtype C = src_data[num_offset + indexC];
						Dtype D = src_data[num_offset + indexD];

						dst_data[num_offset + dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
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
		/// rotate around any point, height and width will be constant
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		/// <param name="center">rotation center</param>
		/// <param name="theta">rotation angle, anti-clockwise is positive</param>
		/// <param name="scale">ratio of scale</param>
		/// <param name="fill_pixel_value">pixel value to fill in the blank area</param>
		/// <param name="type">interpolationType: Nearest / Bilinear</param>
		template<typename Dtype, typename Ptype>
		void tensor_operation_gpu::rotate_with_points_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst,
			const point<Ptype> &center, float theta, float scale, int fill_pixel_value, interpolationType type)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			if (fabs(theta) <= 1e-6)
			{
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				return;
			}

			int num = src->num();
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
			CUDA_CHECK(cudaMalloc(&reverse_M_data, 9 * sizeof(double)));
			CUDA_CHECK(cudaMemcpy(reverse_M_data, &(reverse_M[0][0]), 3 * sizeof(double), cudaMemcpyDefault));
			CUDA_CHECK(cudaMemcpy(reverse_M_data + 3, &(reverse_M[1][0]), 3 * sizeof(double), cudaMemcpyDefault));
			CUDA_CHECK(cudaMemcpy(reverse_M_data + 6, &(reverse_M[2][0]), 3 * sizeof(double), cudaMemcpyDefault));

			std::shared_ptr<tensor<Dtype>> dst_temp;
			dst_temp.reset(new tensor<Dtype>(src->data_shape(), src->device(), src->order()));
			Dtype* dst_data = dst_temp->mutable_gpu_data();
			const Dtype* src_data = src->gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_rotate_with_points << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, reverse_M_data, fill_pixel_value, type, src->order());
			dst = std::make_shared<tensor<Dtype>>(dst_temp->clone());
		}



		/// <summary>
		/// rotate around any point, height and width will be constant
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		/// <param name="center">rotation center</param>
		/// <param name="theta">rotation angle, anti-clockwise is positive</param>
		/// <param name="scale">ratio of scale</param>
		/// <param name="fill_pixel_value">pixel value to fill in the blank area</param>
		/// <param name="type">interpolationType: Nearest / Bilinear</param>
		template<typename Dtype, typename Ptype>
		void tensor_operation_gpu::rotate_with_points_gpu(const tensor<Dtype> &src, tensor<Dtype> &dst,
			const point<Ptype> &center, float theta, float scale, int fill_pixel_value, interpolationType type)
		{
			if (src.device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			if (fabs(theta) <= 1e-6)
			{
				dst = src.clone();
				return;
			}

			int num = src.num();
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
			CUDA_CHECK(cudaMalloc(&reverse_M_data, 9 * sizeof(double)));
			CUDA_CHECK(cudaMemcpy(reverse_M_data, &(reverse_M[0][0]), 3 * sizeof(double), cudaMemcpyDefault));
			CUDA_CHECK(cudaMemcpy(reverse_M_data + 3, &(reverse_M[1][0]), 3 * sizeof(double), cudaMemcpyDefault));
			CUDA_CHECK(cudaMemcpy(reverse_M_data + 6, &(reverse_M[2][0]), 3 * sizeof(double), cudaMemcpyDefault));

			tensor<Dtype> dst_temp = tensor<Dtype>(src.data_shape(), src.device(), src.order());
			Dtype* dst_data = dst_temp.mutable_gpu_data();
			const Dtype* src_data = src.gpu_data();
			
			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_rotate_with_points << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, reverse_M_data, fill_pixel_value, type, src.order());
			dst = dst_temp.clone();
		}






		/// <summary>
		/// flip image
		/// </summary>
		/// <param name="src_data">original image data</param>
		/// <param name="dst_data">fliped image data</param>
		/// <param name="channels">image channel</param>
		/// <param name="height">image height</param>
		/// <param name="width">image width</param>
		/// <param name="axis">flip axis: Width_Wise / Height_Wise / Center_Wise(flip both height and width) / Channel_Wise</param>
		/// <param name="order">orderType: NHWC / NCHW</param>
		template<typename Dtype>
		__global__
			void kernel_flip(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, flipType axis, orderType order)
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int channelID = totalID % channels;
			int offset = channels * height * width;
			int numID = totalID / offset;
			int remainID = totalID % offset;
			remainID /= channels;
			int rowID = remainID / width;
			int colID = remainID % width;
			int num_offset = numID * channels * height * width;

			if (order == NCHW)
			{
				if (axis == Width_Wise)
				{
					int channel_offset = channelID * height * width;
					int index = channel_offset + rowID * width;
					dst_data[num_offset + index + colID] = src_data[num_offset + index + (width - colID - 1)];
				}
				else if (axis == Height_Wise)
				{
					int channel_offset = channelID * height * width;
					int dst_index = channel_offset + rowID * width;
					int src_index = channel_offset + (height - rowID - 1) * width;
					dst_data[num_offset + dst_index + colID] = src_data[num_offset + src_index + colID];
				}
				else if (axis == Center_Wise)
				{
					int channel_offset = channelID * height * width;
					int dst_index = channel_offset + rowID * width;
					int src_index = channel_offset + (height - rowID - 1) * width;
					dst_data[num_offset + dst_index + colID] = src_data[num_offset + src_index + (width - colID - 1)];
				}
				else if (axis == Channel_Wise)
				{
					int dst_channel_offset = channelID * height * width;
					int src_channel_offset = (channels - 1 - channelID) * height * width;
					int dst_index = dst_channel_offset + rowID * width;
					int src_index = src_channel_offset + rowID * width;
					dst_data[num_offset + dst_index + colID] = src_data[num_offset + src_index + colID];
				}
			}
			else if (order == NHWC)
			{
				if (axis == Width_Wise)
				{
					int dst_index = (rowID * width + colID) * channels + channelID;
					int src_index = (rowID * width + (width - 1 - colID)) * channels + channelID;
					dst_data[num_offset + dst_index] = src_data[num_offset + src_index];
				}
				else if (axis == Height_Wise)
				{
					int dst_index = (rowID * width + colID) * channels + channelID;
					int src_index = ((height - 1 - rowID) * width + colID) * channels + channelID;
					dst_data[num_offset + dst_index] = src_data[num_offset + src_index];
				}
				else if (axis == Center_Wise)
				{
					int dst_index = (rowID * width + colID) * channels + channelID;
					int src_index = ((height - 1 - rowID) * width + (width - 1 - colID)) * channels + channelID;
					dst_data[num_offset + dst_index] = src_data[num_offset + src_index];
				}
				else if (axis == Channel_Wise)
				{
					int dst_index = (rowID * width + colID) * channels + channelID;
					int src_index = (rowID * width + colID) * channels + channels - 1 - channelID;
					dst_data[num_offset + dst_index] = src_data[num_offset + src_index];
				}
			}
			else
			{
				return;
			}
		}



		/// <summary>
		/// flip image
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		/// <param name="axis">flip axis: Width_Wise / Height_Wise / Center_Wise(flip both height and width) / Channel_Wise</param>
		template<typename Dtype>
		void tensor_operation_gpu::flip_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst, flipType axis)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src->num();
			int channels = src->channels();
			int height = src->height();
			int width = src->width();

			std::shared_ptr<tensor<Dtype>> dst_temp;
			dst_temp.reset(new tensor<Dtype>(src->data_shape(), src->device(), src->order()));
			Dtype* dst_data = dst_temp->mutable_gpu_data();
			const Dtype* src_data = src->gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_flip << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, axis, src->order());
			dst = std::make_shared<tensor<Dtype>>(dst_temp->clone());
		}



		/// <summary>
		/// flip image
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		/// <param name="axis">flip axis: Width_Wise / Height_Wise / Center_Wise(flip both height and width) / Channel_Wise</param>
		template<typename Dtype>
		void tensor_operation_gpu::flip_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst, flipType axis)
		{
			if (src.device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src.num();
			int channels = src.channels();
			int height = src.height();
			int width = src.width();

			tensor<Dtype> dst_temp = tensor<Dtype>(src.data_shape(), src.device(), src.order());
			Dtype* dst_data = dst_temp.mutable_gpu_data();
			const Dtype* src_data = src.gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_flip << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, axis, src.order());
			dst = dst_temp.clone();
		}






		/// <summary>
		/// convert 3 channels image to 1 channel
		/// </summary>
		/// <param name="src_data">original 3 channels image data</param>
		/// <param name="dst_data">new 1 channel image data</param>
		/// <param name="height">image height</param>
		/// <param name="width">image width</param>
		/// <param name="order">orderType: NHWC / NCHW</param>
		template<typename Dtype>
		__global__
			void kernel_rgb2gray(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, orderType order)
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int offset = height * width;
			int numID = totalID / offset;
			int remainID = totalID % offset;
			int rowID = remainID / width;
			int colID = remainID % width;
			int src_num_offset = numID * channels * height * width;
			int dst_num_offset = numID * 1 * height * width;

			int index = rowID * width + colID;

			//pixel order in opencv: B / G / R
			//convert formula: gray = 0.114 * B + 0.587 * G + 0.299 * R
			if (order == NCHW)
			{
				dst_data[dst_num_offset + index] = Dtype(static_cast<float>(src_data[src_num_offset + index]) * 0.114f +
					static_cast<float>(src_data[src_num_offset + offset * 1 + index]) * 0.587f +
					static_cast<float>(src_data[src_num_offset + offset * 2 + index]) * 0.299f);
			}
			else if (order == NHWC)
			{
				dst_data[dst_num_offset + index] = Dtype(static_cast<float>(src_data[src_num_offset + 3 * index]) * 0.114f +
					static_cast<float>(src_data[src_num_offset + 3 * index + 1]) * 0.587f +
					static_cast<float>(src_data[src_num_offset + 3 * index + 2]) * 0.299f);
			}
			else
			{
				return;
			}
		}



		/// <summary>
		/// convert 3 channels image to 1 channel
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		template<typename Dtype>
		void tensor_operation_gpu::rgb2gray_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src->num();
			int channels = src->channels();
			int height = src->height();
			int width = src->width();

			if (!(channels == 3 || channels == 4))
			{
				LOG(ERROR) << "Incorrect input channel.";
				return;
			}

			if (src->order() == NCHW)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{num, 1, height, width}, src->device(), src->order()));
			}
			else if (src->order() == NHWC)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{num, height, width, 1}, src->device(), src->order()));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			const Dtype* src_data = src->gpu_data();
			Dtype* dst_data = dst->mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_rgb2gray << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, src->order());
		}



		/// <summary>
		/// convert 3 channels image to 1 channel
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		template<typename Dtype>
		void tensor_operation_gpu::rgb2gray_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst)
		{
			if (src.device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src.num();
			int channels = src.channels();
			int height = src.height();
			int width = src.width();

			if (!(channels == 3 || channels == 4))
			{
				LOG(ERROR) << "Incorrect input channel.";
				return;
			}

			if (src.order() == NCHW)
			{
				dst = tensor<Dtype>(std::vector<int>{num, 1, height, width}, src.device(), src.order());
			}
			else if (src.order() == NHWC)
			{
				dst = tensor<Dtype>(std::vector<int>{num, height, width, 1}, src.device(), src.order());
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			const Dtype* src_data = src.gpu_data();
			Dtype* dst_data = dst.mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_rgb2gray << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, src.order());
		}






		/// <summary>
		/// convert rgb image to hsv image
		/// </summary>
		/// <param name="src_data">original rgb image data</param>
		/// <param name="dst_data">new hsv image data</param>
		/// <param name="height">image height</param>
		/// <param name="width">image width</param>
		/// <param name="order">orderType: NHWC / NCHW</param>
		template<typename Dtype>
		__global__
			void kernel_rgb2hsv(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, orderType order)
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int offset = height * width;
			int numID = totalID / offset;
			int remainID = totalID % offset;
			int rowID = remainID / width;
			int colID = remainID % width;
			int src_num_offset = numID * channels * height * width;
			int dst_num_offset = numID * 3 * height * width;

			int index = rowID * width + colID;
			unsigned char B, G, R, minVal, maxVal, interval;

			if (order == NCHW)
			{
				B = src_data[src_num_offset + index];
				G = src_data[src_num_offset + offset * 1 + index];
				R = src_data[src_num_offset + offset * 2 + index];

				minVal = min(B, G);
				minVal = min(minVal, R);
				maxVal = max(B, G);
				maxVal = max(maxVal, R);
				interval = maxVal - minVal;
			}
			else if (order == NHWC)
			{
				B = src_data[src_num_offset + 3 * index];
				G = src_data[src_num_offset + 3 * index + 1];
				R = src_data[src_num_offset + 3 * index + 2];
				
				minVal = min(B, G);
				minVal = min(minVal, R);
				maxVal = max(B, G);
				maxVal = max(maxVal, R);
				interval = maxVal - minVal;
			}
			else
			{
				return;
			}

			//v
			dst_data[dst_num_offset + 2 * offset + index] = maxVal;

			//s
			if (maxVal == 0)
			{
				dst_data[dst_num_offset + offset + index] = 0;
			}
			else
			{
				dst_data[dst_num_offset + offset + index] = 255 * interval / maxVal;
			}

			//h
			int H;
			if (maxVal == minVal)
			{
				H = 0;
			}
			else if ((maxVal == R) && (G >= B))
			{
				H = 60 * (G - B) / interval;
			}
			else if ((maxVal == R) && (G < B))
			{
				H = 60 * (G - B) / interval + 360;
			}
			else if (maxVal == G)
			{
				H = 120 + 60 * (B - R) / interval;
			}
			else if (maxVal == B)
			{
				H = 240 + 60 * (R - G) / interval;
			}

			dst_data[dst_num_offset + index] = H / 2;
		}



		/// <summary>
		/// convert rgb image to hsv image
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		template<typename Dtype>
		void tensor_operation_gpu::rgb2hsv_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src->num();
			int channels = src->channels();
			int height = src->height();
			int width = src->width();

			if (!(channels == 3 || channels == 4))
			{
				LOG(ERROR) << "Incorrect input channel.";
				return;
			}

			std::shared_ptr<tensor<Dtype>> dst_temp;
			dst_temp.reset(new tensor<Dtype>(std::vector<int>{num, 3, height, width}, src->device(), NCHW));

			const Dtype* src_data = src->gpu_data();
			Dtype* dst_data = dst_temp->mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, num);

			kernel_rgb2hsv << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, src->order());
			dst = std::make_shared<tensor<Dtype>>(dst_temp->clone());
		}




		/// <summary>
		/// convert rgb image to hsv image
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		template<typename Dtype>
		void tensor_operation_gpu::rgb2hsv_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst)
		{
			if (src.device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src.num();
			int channels = src.channels();
			int height = src.height();
			int width = src.width();

			if (!(channels == 3 || channels == 4))
			{
				LOG(ERROR) << "Incorrect input channel.";
				return;
			}

			tensor<Dtype> dst_temp = tensor<Dtype>(std::vector<int>{num, 3, height, width}, src.device(), NCHW);

			const Dtype* src_data = src.gpu_data();
			Dtype* dst_data = dst_temp.mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, num);

			kernel_rgb2hsv << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, src.order());
			dst = dst_temp.clone();
		}




		/// <summary>
		/// matrix transpose
		/// </summary>
		/// <param name="src_data">original image data</param>
		/// <param name="dst_data">new image data</param>
		/// <param name="channels">image channel</param>
		/// <param name="height">image height</param>
		/// <param name="width">image width</param>
		/// <param name="order">orderType: NHWC / NCHW</param>
		template<typename Dtype>
		__global__
			void kernel_matrix_transpose(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, orderType order)
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int channelID = totalID % channels;
			int offset = channels * height * width;
			int numID = totalID / offset;
			int remainID = totalID % offset;
			remainID /= channels;
			int rowID = remainID / width;
			int colID = remainID % width;
			int num_offset = numID * channels * height * width;

			if (order == NCHW)
			{
				int dst_index = rowID * width + colID;
				int src_index = colID * height + rowID;
				int channel_offset = channelID * height * width;
				dst_data[num_offset + dst_index + channel_offset] = src_data[num_offset + src_index + channel_offset];
			}
			else if (order == NHWC)
			{
				int dst_index = (rowID * width + colID) * channels + channelID;
				int src_index = (colID * height + rowID) * channels + channelID;
				dst_data[num_offset + dst_index] = src_data[num_offset + src_index];
			}
			else
			{
				return;
			}
		}



		/// <summary>
		/// matrix transpose
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		template<typename Dtype>
		void tensor_operation_gpu::matrix_transpose_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src->num();
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			int offset = height * width;

			std::shared_ptr<tensor<Dtype>> dst_temp;
			if (src->order() == NCHW)
			{
				dst_temp.reset(new tensor<Dtype>(std::vector<int>{num, channels, width, height}, src->device(), src->order()));
			}
			else if (src->order() == NHWC)
			{
				dst_temp.reset(new tensor<Dtype>(std::vector<int>{num, width, height, channels}, src->device(), src->order()));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			const Dtype* src_data = src->gpu_data();
			Dtype* dst_data = dst_temp->mutable_gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(height, width, num);

			
			kernel_matrix_transpose << <grid_size, block_size >> > (src_data, dst_data, channels, width, height, src->order());
			dst = std::make_shared<tensor<Dtype>>(dst_temp->clone());
		}



		/// <summary>
		/// matrix transpose
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		template<typename Dtype>
		void tensor_operation_gpu::matrix_transpose_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst)
		{
			if (src.device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src.num();
			int channels = src.channels();
			int height = src.height();
			int width = src.width();
			int offset = height * width;

			tensor<Dtype> dst_temp;
			if (src.order() == NCHW)
			{
				dst_temp = tensor<Dtype>(std::vector<int>{num, channels, width, height}, src.device(), src.order());
			}
			else if (src.order() == NHWC)
			{
				dst_temp = tensor<Dtype>(std::vector<int>{num, width, height, channels}, src.device(), src.order());
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			const Dtype* src_data = src.gpu_data();
			Dtype* dst_data = dst_temp.mutable_gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(height, width, num);

			
			kernel_matrix_transpose << <grid_size, block_size >> > (src_data, dst_data, channels, width, height, src.order());
			dst = dst_temp.clone();
		}






		/// <summary>
		/// get ROI(region of interest) from image
		/// </summary>
		/// <param name="src_data">original image data</param>
		/// <param name="channels">image channel</param>
		/// <param name="src_height">original height</param>
		/// <param name="src_width">original width</param>
		/// <param name="dst_data">ROI image data</param>
		/// <param name="rect">ROI rectangle</param>
		/// <param name="order">orderType: NHWC / NCHW</param>
		template<typename Dtype, typename Rtype>
		__global__
			void kernel_ROI(const Dtype* src_data, int channels, int src_height, int src_width,
				Dtype* dst_data, excalibur::rectangle<Rtype> rect, orderType order)
		{
			int dst_height = (int)rect.h;
			int dst_width = (int)rect.w;

			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int channelID = totalID % channels;
			int dst_offset = channels * dst_height * dst_width;
			int numID = totalID / dst_offset;
			int remainID = totalID % dst_offset;
			remainID /= channels;
			int rowID = remainID / dst_width;
			int colID = remainID % dst_width;
			int dst_num_offset = numID * channels * dst_height * dst_width;
			int src_num_offset = numID * channels * src_height * src_width;

			if (order == NCHW)
			{
				int src_channel_offset = channelID * src_height * src_width;
				int dst_channel_offset = channelID * dst_height * dst_width;
				int src_index = src_channel_offset + (rowID + rect.y) * src_width + (colID + rect.x);
				int dst_index = dst_channel_offset + rowID * dst_width + colID;
				dst_data[dst_num_offset + dst_index] = src_data[src_num_offset + src_index];
			}
			else if (order == NHWC)
			{
				int src_index = (rowID + rect.y) * src_width * channels + (colID + rect.x) * channels + channelID;
				int dst_index = (rowID * dst_width + colID) * channels + channelID;
				dst_data[dst_num_offset + dst_index] = src_data[src_num_offset + src_index];
			}
			else
			{
				return;
			}
		}



		/// <summary>
		/// get ROI(region of interest) from image
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">ROI tensor</param>
		/// <param name="rect">ROI rectangle</param>
		template<typename Dtype, typename Rtype>
		void tensor_operation_gpu::roi_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst, excalibur::rectangle<Rtype> rect)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src->num();
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
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				return;
			}

			if (src->order() == NCHW)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src->device(), src->order()));
			}
			else if (src->order() == NHWC)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src->device(), src->order()));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			Dtype* dst_data = dst->mutable_gpu_data();
			const Dtype* src_data = src->gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(dst_width, dst_height, num);

			
			kernel_ROI << <grid_size, block_size >> > (src_data, channels, height, width, dst_data, rect, src->order());
		}



		/// <summary>
		/// get ROI(region of interest) from image
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">ROI tensor</param>
		/// <param name="rect">ROI rectangle</param>
		template<typename Dtype, typename Rtype>
		void tensor_operation_gpu::roi_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst, excalibur::rectangle<Rtype> rect)
		{
			if (src.device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src.num();
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
				dst = src.clone();
				return;
			}

			if (src.order() == NCHW)
			{
				dst = tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src.device(), src.order());
			}
			else if (src.order() == NHWC)
			{
				dst = tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src.device(), src.order());
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			Dtype* dst_data = dst.mutable_gpu_data();
			const Dtype* src_data = src.gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(dst_width, dst_height, num);

			
			kernel_ROI << <grid_size, block_size >> > (src_data, channels, height, width, dst_data, rect, src.order());
		}






		/// <summary>
		/// threshold image data, gray image required
		/// </summary>
		/// <param name="src_data">original image data</param>
		/// <param name="dst_data">new image data</param>
		/// <param name="thresh">threshold value</param>
		/// <param name="maxval">max value</param>
		/// <param name="type">thresholdType: binary(default, set pixel value as maxval when pixel value is greater than thresh, otherwise set pixel value as 0)
		///                               binary_inv(set pixel value as 0 when pixel value is greater than thresh, otherwise set pixel value as maxval)
		///                              small_trunc(set pixel value as thresh when pixel value is smaller than thresh, otherwise remain unchanged)
		///                                big_trunc(set pixel value as thresh when pixel value is greater than thresh, otherwise remain unchanged)
		///                            small_to_zero(set pixel value as 0 when pixel value is smaller than thresh, otherwise remain unchanged)
		///                              big_to_zero(set pixel value as 0 when pixel value is greater than thresh, otherwise remain unchanged)</param>
		template<typename Dtype>
		__global__
			void kernel_threshold(const Dtype* src_data, Dtype* dst_data, int thresh, int maxval, thresholdType type)
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

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
		/// threshold image data, gray image required
		/// </summary>
		/// <param name="src">original tensor, gray image</param>
		/// <param name="dst">new tensor</param>
		/// <param name="thresh">threshold value</param>
		/// <param name="maxval">max value</param>
		/// <param name="type">thresholdType: binary(default, set pixel value as maxval when pixel value is greater than thresh, otherwise set pixel value as 0)
		///                               binary_inv(set pixel value as 0 when pixel value is greater than thresh, otherwise set pixel value as maxval)
		///                              small_trunc(set pixel value as thresh when pixel value is smaller than thresh, otherwise remain unchanged)
		///                                big_trunc(set pixel value as thresh when pixel value is greater than thresh, otherwise remain unchanged)
		///                            small_to_zero(set pixel value as 0 when pixel value is smaller than thresh, otherwise remain unchanged)
		///                              big_to_zero(set pixel value as 0 when pixel value is greater than thresh, otherwise remain unchanged)</param>
		template<typename Dtype>
		void tensor_operation_gpu::threshold_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst,
			int thresh, int maxval, thresholdType type)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src->num();
			CHECK_EQ(src->channels(), 1);
			int height = src->height();
			int width = src->width();
			int src_offset = height * width;

			std::shared_ptr<tensor<Dtype>> dst_temp;
			dst_temp.reset(new tensor<Dtype>(src->data_shape(), src->device(), src->order()));
			Dtype* dst_data = dst_temp->mutable_gpu_data();
			const Dtype *src_data = src->gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_threshold<Dtype> << <grid_size, block_size >> > (src_data, dst_data, thresh, maxval, type);
			dst = std::make_shared<tensor<Dtype>>(dst_temp->clone());
		}



		/// <summary>
		/// threshold image data, gray image required
		/// </summary>
		/// <param name="src">original tensor, gray image</param>
		/// <param name="dst">new tensor</param>
		/// <param name="thresh">threshold value</param>
		/// <param name="maxval">max value</param>
		/// <param name="type">thresholdType: binary(default, set pixel value as maxval when pixel value is greater than thresh, otherwise set pixel value as 0)
		///                               binary_inv(set pixel value as 0 when pixel value is greater than thresh, otherwise set pixel value as maxval)
		///                              small_trunc(set pixel value as thresh when pixel value is smaller than thresh, otherwise remain unchanged)
		///                                big_trunc(set pixel value as thresh when pixel value is greater than thresh, otherwise remain unchanged)
		///                            small_to_zero(set pixel value as 0 when pixel value is smaller than thresh, otherwise remain unchanged)
		///                              big_to_zero(set pixel value as 0 when pixel value is greater than thresh, otherwise remain unchanged)</param>
		template<typename Dtype>
		void tensor_operation_gpu::threshold_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst, 
			int thresh, int maxval, thresholdType type)
		{
			if (src.device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src.num();
			CHECK_EQ(src.channels(), 1);
			int height = src.height();
			int width = src.width();
			int src_offset = height * width;

			tensor<Dtype> dst_temp = tensor<Dtype>(src.data_shape(), src.device(), src.order());
			Dtype* dst_data = dst_temp.mutable_gpu_data();
			const Dtype *src_data = src.gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_threshold<Dtype> << <grid_size, block_size >> > (src_data, dst_data, thresh, maxval, type);
			dst = dst_temp.clone();
		}



		/// <summary>
		/// warp affine image
		/// </summary>
		/// <param name="src_data">original image data</param>
		/// <param name="dst_data">new image data</param>
		/// <param name="channels">image channel</param>
		/// <param name="height">image height</param>
		/// <param name="width">image width</param>
		/// <param name="M_data">transformation matrix</param>
		/// <param name="fill_pixel_value">fill blank area with fill_pixel_value</param>
		/// <param name="type">interpolationType: Nearest / Bilinear</param>
		/// <param name="order">orderType: NHWC / NCHW</param>
		template<typename Dtype>
		__global__
			void kernel_warp_affine(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, float* X, int fill_pixel_value, interpolationType type, orderType order)
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int channelID = totalID % channels;
			int offset = channels * height * width;
			int numID = totalID / offset;
			int remainID = totalID % offset;
			remainID /= channels;
			int rowID = remainID / width;
			int colID = remainID % width;
			int num_offset = numID * channels * height * width;
			unsigned maxIndex = height * width * channels - 1;

			double xf = X[0] * colID + X[1] * rowID + X[2];
			double yf = X[3] * colID + X[4] * rowID + X[5];
			int x = (int)xf;
			int y = (int)yf;
			float xdiff = xf - x;
			float ydiff = yf - y;

			if (order == NCHW)
			{
				int channel_offset = channelID * height * width;
				int src_index = channel_offset + y * width + x;
				int dst_index = channel_offset + rowID * width + colID;

				if (x < 0 || x >= width || y < 0 || y >= height)
				{
					dst_data[num_offset + dst_index] = (Dtype)fill_pixel_value;
				}
				else
				{
					if (type == Nearest)
					{
						dst_data[num_offset + dst_index] = src_data[num_offset + src_index];
					}
					else if (type == Bilinear)
					{
						unsigned indexA = min(unsigned(src_index), maxIndex);
						unsigned indexB = min(unsigned(src_index + 1), maxIndex);
						unsigned indexC = min(unsigned(src_index + width), maxIndex);
						unsigned indexD = min(unsigned(src_index + width + 1), maxIndex);
						Dtype A = src_data[num_offset + indexA];
						Dtype B = src_data[num_offset + indexB];
						Dtype C = src_data[num_offset + indexC];
						Dtype D = src_data[num_offset + indexD];

						dst_data[num_offset + dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
							static_cast<float>(B) * xdiff * (1 - ydiff) +
							static_cast<float>(C) * ydiff * (1 - xdiff) +
							static_cast<float>(D) * xdiff * ydiff);
					}
				}
			}
			else if (order == NHWC)
			{
				int src_index = (y * width + x) * channels + channelID;
				int dst_index = (rowID * width + colID) * channels + channelID;

				if (x < 0 || x >= width || y < 0 || y >= height)
				{
					dst_data[num_offset + dst_index] = (Dtype)fill_pixel_value;
				}
				else
				{
					if (type == Nearest)
					{
						dst_data[num_offset + dst_index] = src_data[num_offset + src_index];
					}
					else if (type == Bilinear)
					{
						unsigned indexA = min(unsigned(src_index), maxIndex);
						unsigned indexB = min(unsigned(src_index + channels), maxIndex);
						unsigned indexC = min(unsigned(src_index + width * channels), maxIndex);
						unsigned indexD = min(unsigned(src_index + (width + 1) * channels), maxIndex);
						Dtype A = src_data[num_offset + indexA];
						Dtype B = src_data[num_offset + indexB];
						Dtype C = src_data[num_offset + indexC];
						Dtype D = src_data[num_offset + indexD];

						dst_data[num_offset + dst_index] = Dtype(static_cast<float>(A) * (1 - xdiff) * (1 - ydiff) +
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
		/// warp affine image
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		/// <param name="src_point">3 points before warp affine</param>
		/// <param name="dst_point">3 points after warp affine, get transformation matrix using src_point and dst_point</param>
		/// <param name="fill_pixel_value">fill blank area with fill_pixel_value</param>
		/// <param name="type">interpolationType: Nearest / Bilinear</param>
		template<typename Dtype, typename Ptype>
		void tensor_operation_gpu::warp_affine_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst,
			const std::vector<point<Ptype>> &src_point, const std::vector<point<Ptype>> &dst_point, int fill_pixel_value, interpolationType type)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			if (src_point.size() != 3 || dst_point.size() != 3)
			{
				LOG(WARNING) << "please use 3 src_points and 3 dst_points.";
				return;
			}

			int num = src->num();
			int channels = src->channels();
			int height = src->height();
			int width = src->width();

			//AX=B, expand A to 6*6, expand B to 6*1
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
			CUDA_CHECK(cudaMalloc(&X_data, 6 * sizeof(float)));
			CUDA_CHECK(cudaMemcpy(X_data, X.data(), 6 * sizeof(float), cudaMemcpyDefault));

			std::shared_ptr<tensor<Dtype>> dst_temp;
			dst_temp.reset(new tensor<Dtype>(src->data_shape(), src->device(), src->order()));
			Dtype* dst_data = dst_temp->mutable_gpu_data();
			const Dtype* src_data = src->gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_warp_affine << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, X_data, fill_pixel_value, type, src->order());
			dst = std::make_shared<tensor<Dtype>>(dst_temp->clone());
		}



		/// <summary>
		/// warp affine image
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		/// <param name="src_point">3 points before warp affine</param>
		/// <param name="dst_point">3 points after warp affine, get transformation matrix using src_point and dst_point</param>
		/// <param name="fill_pixel_value">fill blank area with fill_pixel_value</param>
		/// <param name="type">interpolationType: Nearest / Bilinear</param>
		template<typename Dtype, typename Ptype>
		void tensor_operation_gpu::warp_affine_gpu(const tensor<Dtype> &src, tensor<Dtype>& dst,
			const std::vector<point<Ptype>> &src_point, const std::vector<point<Ptype>> &dst_point, int fill_pixel_value, excalibur::interpolationType type)
		{
			if (src.device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			if (src_point.size() != 3 || dst_point.size() != 3)
			{
				LOG(WARNING) << "please use 3 src_points and 3 dst_points.";
				return;
			}

			int num = src.num();
			int channels = src.channels();
			int height = src.height();
			int width = src.width();

			//AX=B, expand A to 6*6, expand B to 6*1
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
			CUDA_CHECK(cudaMalloc(&X_data, 6 * sizeof(float)));
			CUDA_CHECK(cudaMemcpy(X_data, X.data(), 6 * sizeof(float), cudaMemcpyDefault));

			tensor<Dtype> dst_temp = tensor<Dtype>(src.data_shape(), src.device(), src.order());
			Dtype* dst_data = dst_temp.mutable_gpu_data();
			const Dtype* src_data = src.gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_warp_affine << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, X_data, fill_pixel_value, type, src.order());
			dst = dst_temp.clone();
		}






		/// <summary>
		/// blur filter(gaussian_blur mean_value_blur)
		/// </summary>
		/// <param name="src_data">original image data</param>
		/// <param name="dst_data">new image data</param>
		/// <param name="channels">image channel</param>
		/// <param name="height">image height</param>
		/// <param name="width">image width</param>
		/// <param name="ksize">size of kernel, odd value required</param>
		/// <param name="paras">weights of kernel</param>
		/// <param name="order">orderType: NHWC / NCHW</param>
		template<typename Dtype>
		__global__
			void kernel_blur(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, int ksize, double *paras, orderType order)
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int channelID = totalID % channels;
			int offset = channels * height * width;
			int numID = totalID / offset;
			int remainID = totalID % offset;
			remainID /= channels;
			int rowID = remainID / width;
			int colID = remainID % width;
			int num_offset = numID * channels * height * width;

			int half = (ksize - 1) * 0.5;

			if (order == NCHW)
			{
				int channel_offset = channelID * height * width;
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

						sum += paras[(kernel_row + half) * ksize + (kernel_col + half)] * src_data[num_offset + index + kernel_row * width + kernel_col];
					}
				}
				dst_data[num_offset + index] = (Dtype)sum;
			}
			else if (order == NHWC)
			{
				int index = (rowID * width + colID) * channels + channelID;
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

						sum += paras[(kernel_row + half) * ksize + (kernel_col + half)] * src_data[num_offset + index + (kernel_row * width + kernel_col) * channels];
					}
				}
				dst_data[num_offset + index] = (Dtype)sum;
			}
			else
			{
				return;
			}
		}



		/// <summary>
		/// gaussian blur
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		/// <param name="ksize">size of gaussian kernel, odd value required</param>
		template<typename Dtype>
		void tensor_operation_gpu::gaussian_blur_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, int ksize)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			if (ksize % 2 != 1)
			{
				LOG(WARNING) << "convolution kernel: width and height should be odd.";
				return;
			}

			if (ksize == 1)
			{
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				return;
			}

			int num = src->num();
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
			CUDA_CHECK(cudaMalloc(&paras, ksize * ksize * sizeof(double)));
			CUDA_CHECK(cudaMemcpy(paras, convolution_kernel, ksize * ksize * sizeof(double), cudaMemcpyDefault));

			std::shared_ptr<tensor<Dtype>> dst_temp;
			dst_temp.reset(new tensor<Dtype>(src->data_shape(), src->device(), src->order()));
			Dtype* dst_data = dst_temp->mutable_gpu_data();
			const Dtype* src_data = src->gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_blur << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, ksize, paras, src->order());
			dst = std::make_shared<tensor<Dtype>>(dst_temp->clone());
		}



		/// <summary>
		/// gaussian blur
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		/// <param name="ksize">size of gaussian kernel, odd value required</param>
		template<typename Dtype>
		void tensor_operation_gpu::gaussian_blur_gpu(const tensor<Dtype> &src, tensor<Dtype> &dst, int ksize)
		{
			if (src.device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			if (ksize % 2 != 1)
			{
				LOG(WARNING) << "convolution kernel: width and height should be odd.";
				return;
			}

			if (ksize == 1)
			{
				dst = src.clone();
				return;
			}

			int num = src.num();
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
			CUDA_CHECK(cudaMalloc(&paras, ksize * ksize * sizeof(double)));
			CUDA_CHECK(cudaMemcpy(paras, convolution_kernel, ksize * ksize * sizeof(double), cudaMemcpyDefault));

			tensor<Dtype> dst_temp = tensor<Dtype>(src.data_shape(), src.device(), src.order());
			Dtype* dst_data = dst_temp.mutable_gpu_data();
			const Dtype* src_data = src.gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_blur << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, ksize, paras, src.order());
			dst = dst_temp.clone();
		}





		/// <summary>
		/// mean value blur
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		/// <param name="ksize">size of kernel, odd value required</param>
		template<typename Dtype>
		void tensor_operation_gpu::mean_value_blur_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, int ksize)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			if (ksize % 2 != 1)
			{
				LOG(WARNING) << "convolution kernel: width and height should be odd.";
				return;
			}

			if (ksize == 1)
			{
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				return;
			}

			int num = src->num();
			int channels = src->channels();
			int height = src->height();
			int width = src->width();

			int half = (ksize - 1) * 0.5;
			double *convolution_kernel = (double*)malloc(ksize * ksize * sizeof(double));
			for (int row = 0; row < ksize; ++row)
			{
				for (int col = 0; col < ksize; ++col)
				{
					convolution_kernel[row * ksize + col] = (double)1/(ksize * ksize);
				}
			}

			double *paras = nullptr;
			CUDA_CHECK(cudaMalloc(&paras, ksize * ksize * sizeof(double)));
			CUDA_CHECK(cudaMemcpy(paras, convolution_kernel, ksize * ksize * sizeof(double), cudaMemcpyDefault));

			std::shared_ptr<tensor<Dtype>> dst_temp;
			dst_temp.reset(new tensor<Dtype>(src->data_shape(), src->device(), src->order()));
			Dtype* dst_data = dst_temp->mutable_gpu_data();
			const Dtype* src_data = src->gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);


			kernel_blur << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, ksize, paras, src->order());
			dst = std::make_shared<tensor<Dtype>>(dst_temp->clone());
		}




		/// <summary>
		/// mean value blur
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		/// <param name="ksize">size of kernel, odd value required</param>
		template<typename Dtype>
		void tensor_operation_gpu::mean_value_blur_gpu(const tensor<Dtype> &src, tensor<Dtype> &dst, int ksize)
		{
			if (src.device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			if (ksize % 2 != 1)
			{
				LOG(WARNING) << "convolution kernel: width and height should be odd.";
				return;
			}

			if (ksize == 1)
			{
				dst = src.clone();
				return;
			}

			int num = src.num();
			int channels = src.channels();
			int height = src.height();
			int width = src.width();

			int half = (ksize - 1) * 0.5;
			double *convolution_kernel = (double*)malloc(ksize * ksize * sizeof(double));
			for (int row = 0; row < ksize; ++row)
			{
				for (int col = 0; col < ksize; ++col)
				{
					convolution_kernel[row * ksize + col] = (double)1/(ksize * ksize);
				}
			}

			double *paras = nullptr;
			CUDA_CHECK(cudaMalloc(&paras, ksize * ksize * sizeof(double)));
			CUDA_CHECK(cudaMemcpy(paras, convolution_kernel, ksize * ksize * sizeof(double), cudaMemcpyDefault));

			tensor<Dtype> dst_temp = tensor<Dtype>(src.data_shape(), src.device(), src.order());
			Dtype* dst_data = dst_temp.mutable_gpu_data();
			const Dtype* src_data = src.gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);


			kernel_blur << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, ksize, paras, src.order());
			dst = dst_temp.clone();
		}




		/// <summary>
		/// calculate gradient in horizontal / vertical direction
		/// </summary>
		/// <param name="src_data">original image data</param>
		/// <param name="dst_data">new image data</param>
		/// <param name="channels">image channel</param>
		/// <param name="height">image height</param>
		/// <param name="width">image width</param>
		/// <param name="dx">calculate horizontal gradient when dx is greater than 0, otherwise do nothing</param>
		/// <param name="dy">calculate vertical gradient when dy is greater than 0, otherwise do nothing</param>
		/// <param name="order">orderType: NHWC / NCHW</param>
		template<typename Dtype>
		__global__
			void kernel_sobel(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, int dx, int dy, orderType order)
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int channelID = totalID % channels;
			int offset = channels * height * width;
			int numID = totalID / offset;
			int remainID = totalID % offset;
			remainID /= channels;
			int rowID = remainID / width;
			int colID = remainID % width;
			int num_offset = numID * channels * height * width;

			//do not deal with first row, first col, last row, last col
			if (rowID == 0 || rowID == height - 2 || colID == 0 || colID == width - 2)
			{
				return;
			}

			if (order == NCHW)
			{
				int sumx = 0, sumy = 0, total = 0;
				int channel_offset = channelID * height * width;
				int pos = channel_offset + rowID * width + colID;
				int posAdd = pos + width;
				int posSub = pos - width;

				if (dx > 0)
				{
					sumx = src_data[num_offset + posSub + 1] + 2 * src_data[num_offset + pos + 1] + src_data[num_offset + posAdd + 1]
						- src_data[num_offset + posSub - 1] - 2 * src_data[num_offset + pos - 1] - src_data[num_offset + posAdd - 1];
				}

				if (dy > 0)
				{
					sumy = src_data[num_offset + posSub - 1] + 2 * src_data[num_offset + posSub] + src_data[num_offset + posSub + 1]
						- src_data[num_offset + posAdd - 1] - 2 * src_data[num_offset + posAdd] - src_data[num_offset + posAdd + 1];
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

				dst_data[num_offset + pos] = (Dtype)total;
			}
			else if (order == NHWC)
			{
				int sumx = 0, sumy = 0, total = 0;
				int pos = (rowID * width + colID) * channels + channelID;
				int posAdd = pos + width * channels;
				int posSub = pos - width * channels;

				if (dx > 0)
				{
					sumx = src_data[num_offset + posSub + channels] + 2 * src_data[num_offset + pos + channels] + src_data[num_offset + posAdd + channels]
						- src_data[num_offset + posSub - channels] - 2 * src_data[num_offset + pos - channels] - src_data[num_offset + posAdd - channels];
				}

				if (dy > 0)
				{
					sumy = src_data[num_offset + posSub - channels] + 2 * src_data[num_offset + posSub] + src_data[num_offset + posSub + channels]
						- src_data[num_offset + posAdd - channels] - 2 * src_data[num_offset + posAdd] - src_data[num_offset + posAdd + channels];
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

				dst_data[num_offset + pos] = (Dtype)total;
			}
			else
			{
				return;
			}
		}



		/// <summary>
		/// calculate gradient in horizontal / vertical direction
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		/// <param name="dx">calculate horizontal gradient when dx is greater than 0, otherwise do nothing</param>
		/// <param name="dy">calculate vertical gradient when dy is greater than 0, otherwise do nothing</param>
		template<typename Dtype>
		void tensor_operation_gpu::sobel_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, int dx, int dy)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			if (dx == 0 && dy == 0)
			{
				LOG(WARNING) << "dx, dy cannot be zero simultaneously.";
				return;
			}

			int num = src->num();
			int channels = src->channels();
			int height = src->height();
			int width = src->width();

			std::shared_ptr<tensor<Dtype>> dst_temp;
			dst_temp.reset(new tensor<Dtype>(src->data_shape(), src->device(), src->order()));
			Dtype* dst_data = dst_temp->mutable_gpu_data();
			const Dtype* src_data = src->gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_sobel << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, dx, dy, src->order());
			dst = std::make_shared<tensor<Dtype>>(dst_temp->clone());
		}



		/// <summary>
		/// calculate gradient in horizontal / vertical direction
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		/// <param name="dx">calculate horizontal gradient when dx is greater than 0, otherwise do nothing</param>
		/// <param name="dy">calculate vertical gradient when dy is greater than 0, otherwise do nothing</param>
		template<typename Dtype>
		void tensor_operation_gpu::sobel_gpu(const tensor<Dtype> &src, tensor<Dtype> &dst, int dx, int dy)
		{
			if (src.device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			if (dx == 0 && dy == 0)
			{
				LOG(WARNING) << "dx, dy cannot be zero simultaneously.";
				return;
			}

			int num = src.num();
			int channels = src.channels();
			int height = src.height();
			int width = src.width();

			tensor<Dtype> dst_temp = tensor<Dtype>(src.data_shape(), src.device(), src.order());
			Dtype* dst_data = dst_temp.mutable_gpu_data();
			const Dtype* src_data = src.gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_sobel << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, dx, dy, src.order());
			dst = dst_temp.clone();
		}






		/// <summary>
		/// dilate or erode image data
		/// </summary>
		/// <param name="src_data">original image data</param>
		/// <param name="dst_data">new image data</param>
		/// <param name="channels">image channel</param>
		/// <param name="height">image height</param>
		/// <param name="width">image width</param>
		/// <param name="type">morphType: Dilate / Erode</param>
		/// <param name="ksize">size of square kernel, odd value required</param>
		/// <param name="order">orderType: NHWC / NCHW</param>
		template<typename Dtype>
		__global__
			void kernel_morph(const Dtype* src_data, Dtype* dst_data, int channels, int height, int width, excalibur::morphType type, int ksize, orderType order)
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int channelID = totalID % channels;
			int offset = channels * height * width;
			int numID = totalID / offset;
			int remainID = totalID % offset;
			remainID /= channels;
			int rowID = remainID / width;
			int colID = remainID % width;
			int num_offset = numID * channels * height * width;

			int half = (ksize - 1) * 0.5;

			if (order == NCHW)
			{
				if (type == Dilate)
				{
					int channel_offset = channelID * height * width;
					int index = channel_offset + rowID * width + colID;
					int max = INT_MIN;
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
							if (src_data[num_offset + pos] > max)
							{
								max = src_data[num_offset + pos];
							}
						}
					}
					dst_data[num_offset + index] = (Dtype)max;
				}
				else if (type == Erode)
				{
					int channel_offset = channelID * height * width;
					int index = channel_offset + rowID * width + colID;
					int min = INT_MAX;
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
							if (src_data[num_offset + pos] < min)
							{
								min = src_data[num_offset + pos];
							}
						}
					}
					dst_data[num_offset + index] = (Dtype)min;
				}
			}
			else if (order == NHWC)
			{
				if (type == Dilate)
				{
					int index = (rowID * width + colID) * channels + channelID;
					int max = INT_MIN;
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
							if (src_data[num_offset + pos] > max)
							{
								max = src_data[num_offset + pos];
							}
						}
					}
					dst_data[num_offset + index] = (Dtype)max;
				}
				else if (type == Erode)
				{
					int index = (rowID * width + colID) * channels + channelID;
					int min = INT_MAX;
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
							if (src_data[num_offset + pos] < min)
							{
								min = src_data[num_offset + pos];
							}
						}
					}
					dst_data[num_offset + index] = (Dtype)min;
				}
			}
			else
			{
				return;
			}
		}



		/// <summary>
		/// dilate or erode image data
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		/// <param name="type">morphType: Dilate(default) / Erode</param>
		/// <param name="ksize">size of square kernel, odd value required</param>
		template<typename Dtype>
		void tensor_operation_gpu::morph_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, 
			excalibur::morphType type, int ksize)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			if (ksize % 2 != 1)
			{
				LOG(WARNING) << "ksize should be odd.";
				return;
			}

			if (ksize == 1)
			{
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				return;
			}

			int num = src->num();
			int channels = src->channels();
			int height = src->height();
			int width = src->width();

			std::shared_ptr<tensor<Dtype>> dst_temp;
			dst_temp.reset(new tensor<Dtype>(src->data_shape(), src->device(), src->order()));
			Dtype* dst_data = dst_temp->mutable_gpu_data();
			const Dtype* src_data = src->gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_morph << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, type, ksize, src->order());
			dst = std::make_shared<tensor<Dtype>>(dst_temp->clone());
		}



		/// <summary>
		/// dilate or erode image data
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		/// <param name="type">morphType: Dilate(default) / Erode</param>
		/// <param name="ksize">size of square kernel, odd value required</param>
		template<typename Dtype>
		void tensor_operation_gpu::morph_gpu(const tensor<Dtype> &src, tensor<Dtype> &dst, excalibur::morphType type, int ksize)
		{
			if (src.device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			if (ksize % 2 != 1)
			{
				LOG(WARNING) << "ksize should be odd.";
				return;
			}

			if (ksize == 1)
			{
				dst = src.clone();
				return;
			}

			int num = src.num();
			int channels = src.channels();
			int height = src.height();
			int width = src.width();

			tensor<Dtype> dst_temp = tensor<Dtype>(src.data_shape(), src.device(), src.order());
			Dtype* dst_data = dst_temp.mutable_gpu_data();
			const Dtype* src_data = src.gpu_data();

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);

			
			kernel_morph << <grid_size, block_size >> > (src_data, dst_data, channels, height, width, type, ksize, src.order());
			dst = dst_temp.clone();
		}



		/// <summary>
		/// convert between different datatype of tensor
		/// </summary>
		/// <param name="src_data">original image data</param>
		/// <param name="dst_data">new image data</param>
		template <typename DtypeSRC, typename DtypeDST>
		__global__
			void kernel_type_converter(const DtypeSRC* src_data, DtypeDST* dst_data)
		{
			int index = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			dst_data[index] = DtypeDST(src_data[index]);
		}



		/// <summary>
		/// convert between different datatype of tensor
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		template <typename DtypeSRC, typename DtypeDST>
		void tensor_operation_gpu::type_converter_gpu(const std::shared_ptr<tensor<DtypeSRC>> &src, std::shared_ptr<tensor<DtypeDST>> &dst)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			const DtypeSRC* src_data = src->gpu_data();
			std::shared_ptr<tensor<DtypeDST>> dst_temp;
			dst_temp.reset(new tensor<DtypeDST>(src->data_shape(), src->device(), src->order()));
			DtypeDST* dst_data = dst_temp->mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(src->count(), 1, 1);

			
			kernel_type_converter << <grid_size, block_size >> > (src_data, dst_data);
			dst = std::make_shared<tensor<DtypeDST>>(dst_temp->clone());
		}



		/// <summary>
		/// convert between different datatype of tensor
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		template <typename DtypeSRC, typename DtypeDST>
		void tensor_operation_gpu::type_converter_gpu(const tensor<DtypeSRC> &src, tensor<DtypeDST> &dst)
		{
			if (src.device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			const DtypeSRC* src_data = src.gpu_data();
			tensor<DtypeDST> dst_temp = tensor<DtypeDST>(src.data_shape(), src.device(), src.order());
			DtypeDST* dst_data = dst_temp.mutable_gpu_data();

			const dim3 block_size(1, 1, 1);
			const dim3 grid_size(src.count(), 1, 1);

			
			kernel_type_converter << <grid_size, block_size >> > (src_data, dst_data);
			dst = dst_temp.clone();
		}



		/// <summary>
		/// preprocess tensor
		/// </summary>
		/// <param name="src_data">original image data</param>
		/// <param name="dst_data">new image data</param>
		/// <param name="channels">image channel</param>
		/// <param name="height">image height</param>
		/// <param name="width">image width</param>
		/// <param name="order">orderType: NHWC / NCHW</param>
		template <typename DtypeSRC, typename DtypeDST>
		__global__
			void kernel_preprocess_tensors(const DtypeSRC* src_data, DtypeDST* dst_data, int num, int channels, int height, int width, orderType order, float *means, float var)
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int channelID = totalID % channels;
			int offset = channels * height * width;
			int numID = totalID / offset;
			int remainID = totalID % offset;
			remainID /= channels;
			int rowID = remainID / width;
			int colID = remainID % width;
			int num_offset = numID * channels * height * width;

			if (channels == 1)
			{
				int index = rowID * width + colID;
				dst_data[num_offset + index] = DtypeDST((src_data[num_offset + index] - means[0]) * var);
			}
			else if (channels == 3)
			{
				if (order == NCHW)
				{
					int channel_offset = channelID * height * width;
					int index = channel_offset + rowID * width + colID;
					dst_data[num_offset + index] = DtypeDST((src_data[num_offset + index] - means[channelID]) * var);
				}
				else if (order == NHWC)
				{
					int index = (rowID * width + colID) * channels + channelID;
					dst_data[num_offset + index] = DtypeDST((src_data[num_offset + index] - means[channelID]) * var);
				}
				else
				{
					return;
				}
			}
		}



		/// <summary>
		/// preprocess tensor
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		template <typename DtypeSRC, typename DtypeDST>
		void tensor_operation_gpu::preprocess_tensors_gpu(const std::shared_ptr<tensor<DtypeSRC>> &src, std::shared_ptr<tensor<DtypeDST>> &dst, float means[3], float var)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src->num();
			int channels = src->channels();
			int height = src->height();
			int width = src->width();

			const DtypeSRC* src_data = src->gpu_data();
			std::shared_ptr<tensor<DtypeDST>> dst_temp;
			dst_temp.reset(new tensor<DtypeDST>(src->data_shape(), src->device(), src->order()));
			DtypeDST* dst_data = dst_temp->mutable_gpu_data();

			std::shared_ptr<tensor<float>> means_tensor;
			means_tensor.reset(new tensor<float>(std::vector<int>{3}, src->device(), src->order()));
			float *means_data = means_tensor->mutable_gpu_data();
			cudaMemcpy(means_data, means, 3 * sizeof(float), cudaMemcpyDefault);

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);

			kernel_preprocess_tensors << <grid_size, block_size >> > (src_data, dst_data, num, channels, height, width, src->order(), means_data, var);
			dst = std::make_shared<tensor<DtypeDST>>(dst_temp->clone());
		}



		/// <summary>
		/// preprocess tensor
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		template <typename DtypeSRC, typename DtypeDST>
		void tensor_operation_gpu::preprocess_tensors_gpu(const tensor<DtypeSRC> &src, tensor<DtypeDST> &dst, float means[3], float var)
		{
			if (src.device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src.num();
			int channels = src.channels();
			int height = src.height();
			int width = src.width();

			const DtypeSRC* src_data = src.gpu_data();
			tensor<DtypeDST> dst_temp = tensor<DtypeDST>(src.data_shape(), src.device(), src.order());
			DtypeDST* dst_data = dst_temp.mutable_gpu_data();

			std::shared_ptr<tensor<float>> means_tensor;
			means_tensor.reset(new tensor<float>(std::vector<int>{3}, src.device(), src.order()));
			float *means_data = means_tensor->mutable_gpu_data();
			cudaMemcpy(means_data, means, 3 * sizeof(float), cudaMemcpyDefault);

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width, height, num);
			
			kernel_preprocess_tensors << <grid_size, block_size >> > (src_data, dst_data, num, channels, height, width, src.order(), means_data, var);
			dst = dst_temp.clone();
		}



		/// <summary>
		/// expand border
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		/// <param name="top">pixels to expand at top of image</param>
		/// <param name="bottom">pixels to expand at bottom of image</param>
		/// <param name="left">pixels to expand at left of image</param>
		/// <param name="right">pixels to expand at right of image</param>
		/// <param name="type">borderType: Border_Constant(default, use constant pixel value(fill_pixel_value) to fill in new blank area) / Border_Replicate(replicate neighboring pixel to fill in new blank area)</param>
		/// <param name="fill_pixel_value">validate when borderType is Border_Constant, zero by default</param>
		template <typename Dtype>
		void tensor_operation_gpu::make_border_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst,
			int top, int bottom, int left, int right, borderType type, Dtype fill_pixel_value)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src->num();
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			int src_offset = height * width;
			int src_num_offset = channels * height * width;

			int dst_height = height + top + bottom;
			int dst_width = width + left + right;
			int dst_offset = dst_height * dst_width;
			int dst_num_offset = channels * dst_height * dst_width;

			if (top < 0 || bottom < 0 || left < 0 || right < 0)
			{
				LOG(ERROR) << "top, bottom, left, right: should all be non-negtive.";
				return;
			}

			if (dst_height == height && dst_width == width)
			{
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				return;
			}

			std::shared_ptr<tensor<Dtype>> dst_temp;
			if (src->order() == NCHW)
			{
				dst_temp.reset(new tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src->device(), src->order()));
				Dtype* dst_data = dst_temp->mutable_gpu_data();
				const Dtype* src_data = src->gpu_data();

				if (type == Border_Constant)
				{
					std::vector<Dtype> top_data(top * dst_width, fill_pixel_value);
					std::vector<Dtype> center_left_data(left, fill_pixel_value);
					std::vector<Dtype> center_right_data(right, fill_pixel_value);
					std::vector<Dtype> bottom_data(bottom * dst_width, fill_pixel_value);

					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;

						for (int ch = 0; ch < channels; ++ch)
						{
							int src_channel_offset = ch * src_offset;
							int dst_channel_offset = ch * dst_offset;

							//top
							CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_channel_offset, top_data.data(), top * dst_width * sizeof(Dtype), cudaMemcpyDefault));

							//center
							for (int row = top; row < top + height; ++row)
							{
								int src_index = src_channel_offset + (row - top) * width;
								int dst_index = dst_channel_offset + row * dst_width;

								CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index, center_left_data.data(), left * sizeof(Dtype), cudaMemcpyDefault));

								CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_index, width * sizeof(Dtype), cudaMemcpyDefault));

								CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index + left + width, center_right_data.data(), right * sizeof(Dtype), cudaMemcpyDefault));
							}

							//bottom
							CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_channel_offset + (top + height) * dst_width, bottom_data.data(), bottom * dst_width * sizeof(Dtype), cudaMemcpyDefault));
						}
					}
				}
				else if (type == Border_Replicate)
				{
					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;

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
									dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_channel_offset];
								}

								//center
								CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_channel_offset, width * sizeof(Dtype), cudaMemcpyDefault));

								//right
								for (int col = left + width; col < dst_width; col++)
								{
									dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_channel_offset + width - 1];
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
									dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_index];
								}

								//center
								CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_index, width * sizeof(Dtype), cudaMemcpyDefault));

								//right
								for (int col = left + width; col < dst_width; col++)
								{
									dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_index + width - 1];
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
									dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_index];
								}

								//center
								CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index + left, src_data + src_n_offset + src_index, width * sizeof(Dtype), cudaMemcpyDefault));

								//right
								for (int col = left + width; col < dst_width; col++)
								{
									dst_data[dst_n_offset + dst_index + col] = src_data[src_n_offset + src_index + width - 1];
								}
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
				dst_temp.reset(new tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src->device(), src->order()));
				Dtype* dst_data = dst_temp->mutable_gpu_data();
				const Dtype* src_data = src->gpu_data();

				if (type == Border_Constant)
				{
					std::vector<Dtype> top_data(channels * top * dst_width, fill_pixel_value);
					std::vector<Dtype> center_left_data(channels * left, fill_pixel_value);
					std::vector<Dtype> center_right_data(channels * right, fill_pixel_value);
					std::vector<Dtype> bottom_data(channels * bottom * dst_width, fill_pixel_value);

					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;

						//top
						CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset, top_data.data(), channels * top * dst_width * sizeof(Dtype), cudaMemcpyDefault));

						//center
						for (int row = top; row < top + height; ++row)
						{
							int src_index = (row - top) * width * channels;

							//left
							CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + row * dst_width * channels, center_left_data.data(), channels * left * sizeof(Dtype), cudaMemcpyDefault));

							//center
							CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + (row * dst_width + left) * channels, src_data + src_n_offset + src_index, width * channels * sizeof(Dtype), cudaMemcpyDefault));

							//right
							CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + (row * dst_width + left + width) * channels, center_right_data.data(), channels * right * sizeof(Dtype), cudaMemcpyDefault));
						}

						//bottom
						CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + (top + height) * dst_width * channels, bottom_data.data(), channels * bottom * dst_width * sizeof(Dtype), cudaMemcpyDefault));
					}
				}
				else if (type == Border_Replicate)
				{
					for (int n = 0; n < num; n++)
					{
						int src_n_offset = n * src_num_offset;
						int dst_n_offset = n * dst_num_offset;

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
									dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + ch];
								}
							}

							//center
							CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index1 + left * channels, src_data + src_n_offset, width * channels * sizeof(Dtype), cudaMemcpyDefault));

							//right
							for (int col = left + width; col < dst_width; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								int src_index = (width - 1) * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index + ch];
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
									dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index1 + ch];
								}
							}

							//center
							CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index1 + left * channels, src_data + src_n_offset + src_index1, width * channels * sizeof(Dtype), cudaMemcpyDefault));

							//right
							int src_index2 = src_index1 + (width - 1) * channels;
							for (int col = left + width; col < dst_width; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index2 + ch];
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
									dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index1 + ch];
								}
							}

							//center
							CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index1 + left * channels, src_data + src_n_offset + src_index1, width * channels * sizeof(Dtype), cudaMemcpyDefault));

							//right
							int src_index2 = src_index1 + (width - 1) * channels;
							for (int col = left + width; col < dst_width; ++col)
							{
								int dst_index2 = dst_index1 + col * channels;
								for (int ch = 0; ch < channels; ++ch)
								{
									dst_data[dst_n_offset + dst_index2 + ch] = src_data[src_n_offset + src_index2 + ch];
								}
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

			dst = std::make_shared<tensor<Dtype>>(dst_temp->clone());
		}


		/// <summary>
        /// cut border
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		/// <param name="top">pixels to cut at top of image</param>
		/// <param name="bottom">pixels to cut at bottom of image</param>
		/// <param name="left">pixels to cut at left of image</param>
		/// <param name="right">pixels to cut at right of image</param>
		template <typename Dtype>
		void tensor_operation_gpu::cut_border_gpu(const std::shared_ptr<tensor<Dtype>> &src,
			std::shared_ptr<tensor<Dtype>>& dst, int top, int bottom, int left, int right)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			if (top < 0 || bottom < 0 || left < 0 || right < 0)
			{
				LOG(ERROR) << "top, bottom, left, right: should all be non-negtive.";
				return;
			}

			int num = src->num();
			int channels = src->channels();
			int height = src->height();
			int width = src->width();
			int src_offset = height * width;
			int src_num_offset = channels * height * width;

			int dst_height = height - top - bottom;
			int dst_width = width - left - right;
			int dst_offset = dst_height * dst_width;
			int dst_num_offset = channels * dst_height * dst_width;

			if (dst_height == height && dst_width == width)
			{
				dst = std::make_shared<tensor<Dtype>>(src->clone());
				return;
			}

			if (dst_height <= 0 || dst_width <= 0)
			{
				LOG(ERROR) << "Illegal input size.";
				return;
			}

			std::shared_ptr<tensor<Dtype>> dst_temp;
			if (src->order() == NCHW)
			{
				dst_temp.reset(new tensor<Dtype>(std::vector<int>{num, channels, dst_height, dst_width}, src->device(), src->order()));
				Dtype* dst_data = dst_temp->mutable_gpu_data();
				const Dtype* src_data = src->gpu_data();

				for (int n = 0; n < num; n++)
				{
					int src_n_offset = n * src_num_offset;
					int dst_n_offset = n * dst_num_offset;

					for (int ch = 0; ch < channels; ++ch)
					{
						int src_channel_offset = ch * src_offset;
						int dst_channel_offset = ch * dst_offset;

						for (int row = 0; row < dst_height; ++row)
						{
							int src_index = src_channel_offset + (row + top) * width + left;
							int dst_index = dst_channel_offset + row * dst_width;
							CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index, src_data + src_n_offset + src_index, dst_width * sizeof(Dtype), cudaMemcpyDefault));
						}
					}
				}
			}
			else if (src->order() == NHWC)
			{
				dst_temp.reset(new tensor<Dtype>(std::vector<int>{num, dst_height, dst_width, channels}, src->device(), src->order()));
				Dtype* dst_data = dst_temp->mutable_gpu_data();
				const Dtype* src_data = src->gpu_data();

				for (int n = 0; n < num; n++)
				{
					int src_n_offset = n * src_num_offset;
					int dst_n_offset = n * dst_num_offset;

					for (int row = 0; row < dst_height; ++row)
					{
						int src_index = ((row + top) * width + left) * channels;
						int dst_index = row * dst_width * channels;
						CUDA_CHECK(cudaMemcpy(dst_data + dst_n_offset + dst_index, src_data + src_n_offset + src_index, dst_width * channels * sizeof(Dtype), cudaMemcpyDefault));
					}
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			dst = std::make_shared<tensor<Dtype>>(dst_temp->clone());
		}


		/// <summary>
		/// get ROI(region of interest) from image. Similar to roi_cpu, but more safe, if rectangle exceeds border, fill 0 instead. 
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">ROI tensor</param>
		/// <param name="rect">region of interest</param>
		template <typename Dtype, typename Rtype>
		void tensor_operation_gpu::safty_cut_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, rectangle<Rtype>* rect)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

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
		/// equalize histogram, gray image required
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">new tensor</param>
		template <typename Dtype>
		void tensor_operation_gpu::equalize_hist_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>>& dst)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src->num();
			CHECK_EQ(src->channels(), 1);
			int height = src->height();
			int width = src->width();
			int offset = height * width;
			const Dtype* src_data = src->gpu_data();
			const Dtype* src_cpu_data = src->cpu_data();
			std::shared_ptr<tensor<Dtype>> dst_temp;
			dst_temp.reset(new tensor<Dtype>(src->data_shape(), src->device(), src->order()));
			Dtype* dst_data = dst_temp->mutable_gpu_data();
			Dtype* temp_dst = new Dtype[offset];

			for (int n = 0; n < num; n++)
			{
				int n_offset = n * offset;
				int gray_value[256] = { 0 };
				float probability_distribution[256] = { 0 };
				float accumulate_probability_distribution[256] = { 0 };
				int normalized_gray_value[256] = { 0 };

				//Count the number of pixels in each grayscale
				for (size_t i = 0; i < offset; i++)
				{
					int value = static_cast<unsigned char>(src_cpu_data[n_offset + i]);
					gray_value[value]++;
				}

				for (int i = 0; i < 256; i++)
				{
					probability_distribution[i] = static_cast<float>(gray_value[i]) / offset;

					if (i > 0)
					{
						accumulate_probability_distribution[i] = accumulate_probability_distribution[i - 1] + probability_distribution[i];
					}
					else
					{
						accumulate_probability_distribution[0] = probability_distribution[0];
					}

					normalized_gray_value[i] = static_cast<unsigned char>(255 * accumulate_probability_distribution[i] + 0.5);
				}

				for (size_t i = 0; i < offset; i++)
				{
					temp_dst[i] = Dtype(normalized_gray_value[static_cast<unsigned char>(src_cpu_data[n_offset + i])]);
				}

				CUDA_CHECK(cudaMemcpy(dst_data + n_offset, temp_dst, offset * sizeof(Dtype), cudaMemcpyDefault));
			}
			
			dst = std::make_shared<tensor<Dtype>>(dst_temp->clone());
			delete temp_dst;
		}


		/// <summary>
		/// merge 3 single-channel images together, to create one 3-channel image 
		/// </summary>
		/// <param name="src_vector">array of tensors, each tensor should share the same height/width/device/order</param>
		/// <param name="dst">new tensor</param>
		template <typename Dtype>
		void tensor_operation_gpu::merge_channel_gpu(const std::vector<std::shared_ptr<tensor<Dtype>>> &src_vector, std::shared_ptr<tensor<Dtype>> &dst)
		{
			CHECK_EQ(src_vector.size(), 3);
			int height, width, device;
			orderType order;
			for (int i = 0; i < src_vector.size(); ++i)
			{
				CHECK_EQ(src_vector.at(i)->channels(), 1);
				if (i == 0)
				{
					height = src_vector.at(i)->height();
					width = src_vector.at(i)->width();
					device = src_vector.at(i)->device();
					order = src_vector.at(i)->order();

					if (device < 0)
					{
						LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
						return;
					}
				}
				else
				{
					if (height != src_vector.at(i)->height() ||
						width != src_vector.at(i)->width() ||
						device != src_vector.at(i)->device() ||
						order != src_vector.at(i)->order())
					{
						LOG(WARNING) << "the element of vector<mat> should have the exact same height/width/device/type.";
						return;
					}
				}
			}

			if (order == NCHW)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, 3, height, width}, device, order));
			}
			else if (order == NHWC)
			{
				dst.reset(new tensor<Dtype>(std::vector<int>{1, height, width, 3}, device, order));
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
				CUDA_CHECK(cudaMemcpy((void*)(dst_data + i * offset), (void*)(temp_data), offset * sizeof(Dtype), cudaMemcpyDefault));
			}
		}





		/// <summary>
        /// for image(w*h), calculate LBP feature((w-2)*(h-2))
        /// </summary>
        /// <param name="src_data">original 3 channels image data</param>
        /// <param name="dst_data">new 1 channel image data</param>
        /// <param name="height">image height</param>
        /// <param name="width">image width</param>
		template<typename Dtype>
		__global__
			void kernel_lbp_feature(const Dtype* src_data, Dtype* dst_data, unsigned char *gpu_LBPMAP_data, int height, int width, bool map_59)
		{
			int totalID = (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
			int offset = height * width;
			int numID = totalID / offset;
			int remainID = totalID % offset;
			int rowID = remainID / width;
			int colID = remainID % width;

			int src_height = height + 2;
			int src_width = width + 2;

			int src_num_offset = numID * src_height * src_width;
			int dst_num_offset = numID * height * width;

			int src_index = src_num_offset + (rowID + 1) * src_width + colID + 1;
			int dst_index = dst_num_offset + rowID * width + colID;

			Dtype center = src_data[src_index];
			unsigned char code = 0;
			//code |= (src_data[src_index - src_width - 1] >= center) << 7;
			//code |= (src_data[src_index - src_width - 0] >= center) << 6;
			//code |= (src_data[src_index - src_width + 1] >= center) << 5;
			//code |= (src_data[src_index + 1] >= center) << 4;
			//code |= (src_data[src_index + src_width + 1] >= center) << 3;
			//code |= (src_data[src_index + src_width + 0] >= center) << 2;
			//code |= (src_data[src_index + src_width - 1] >= center) << 1;
			//code |= (src_data[src_index - 1] >= center) << 0;

			code |= (src_data[src_index - src_width - 1] > center) << 0;
			code |= (src_data[src_index - src_width - 0] > center) << 1;
			code |= (src_data[src_index - src_width + 1] > center) << 2;
			code |= (src_data[src_index + 1] > center) << 3;
			code |= (src_data[src_index + src_width + 1] > center) << 4;
			code |= (src_data[src_index + src_width + 0] > center) << 5;
			code |= (src_data[src_index + src_width - 1] > center) << 6;
			code |= (src_data[src_index - 1] > center) << 7;

			if (map_59)
			{
				dst_data[dst_index] = static_cast<Dtype>(gpu_LBPMAP_data[code]);
			}
			else
			{
				dst_data[dst_index] = static_cast<Dtype>(code);
			}			
		}

		template <typename Dtype>
		/// <summary>
		/// for image(w*h), calculate LBP feature((w-2)*(h-2))
		/// </summary>
		/// <param name="src">original tensor</param>
		/// <param name="dst">LBP feature tensor</param>
		/// <param name="type">lbpType: Native(calculate with neighboring 8 pixels)</param>
		static void tensor_operation_gpu::lbp_feature_gpu(const std::shared_ptr<tensor<Dtype>> &src, std::shared_ptr<tensor<Dtype>> &dst, bool map_59)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src->num();
			int channels = src->channels();
			CHECK_EQ(channels, 1);
			int height = src->height();
			int width = src->width();
			int src_num_offset = channels * height * width;
			int dst_num_offset = channels * (height - 2) * (width - 2);
			std::shared_ptr<tensor<Dtype>> dst_temp;

			if (src->order() == NCHW)
			{
				dst_temp.reset(new tensor<Dtype>(std::vector<int>{num, channels, height - 2, width - 2}, src->device(), src->order()));
			}
			else if (src->order() == NHWC)
			{
				dst_temp.reset(new tensor<Dtype>(std::vector<int>{num, height - 2, width - 2, channels}, src->device(), src->order()));
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			const Dtype* src_data = src->gpu_data();
			Dtype* dst_data = dst_temp->mutable_gpu_data();

			std::shared_ptr<tensor<unsigned char>> gpu_LBPMAP;
			gpu_LBPMAP.reset(new tensor<unsigned char>(256, 0));
			unsigned char *gpu_LBPMAP_data = gpu_LBPMAP->mutable_gpu_data();
			cudaMemcpy(gpu_LBPMAP_data, LBPMAP[0], 256 * sizeof(unsigned char), cudaMemcpyDefault);

			const dim3 block_size(channels, 1, 1);
			const dim3 grid_size(width - 2, height - 2, num);

			kernel_lbp_feature << <grid_size, block_size >> > (src_data, dst_data, gpu_LBPMAP_data, height - 2, width - 2, map_59);
			dst = std::make_shared<tensor<Dtype>>(dst_temp->clone());
		}


		/// <summary>
        /// calculate histogram, gray image required
        /// </summary>
        /// <param name="src">original tensor</param>
        /// <param name="dst">new tensor</param>
		void tensor_operation_gpu::calc_hist_gpu(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<float>>& dst, int dimension)
		{
			if (src->device() < 0)
			{
				LOG(ERROR) << "device wrong, invoke function xxx_cpu() instead!!!";
				return;
			}

			int num = src->num();
			CHECK_EQ(src->channels(), 1);
			int height = src->height();
			int width = src->width();
			int offset = height * width;

			std::shared_ptr<tensor<int>> dst_int;
			dst_int.reset(new tensor<int>(std::vector<int>{num, 1, 1, dimension}, -1, src->order()));
			int *dst_int_data = dst_int->mutable_cpu_data();

			const unsigned char* src_data = src->cpu_data();

			for (int n = 0; n < num; n++)
			{
				int *dst_int_data_num = dst_int_data + n * dimension;

				//Count the number of pixels in each grayscale
				for (int i = 0; i < offset; i++)
				{
					int value = static_cast<int>(src_data[n * offset + i]);
					dst_int_data_num[value]++;
				}
			}

			std::vector<int> max_vals(num), min_vals(num);
			for (int n = 0; n < num; n++)
			{
				max_vals[n] = INT_MIN;
				min_vals[n] = INT_MAX;
			}

			for (int n = 0; n < num; n++)
			{
				int *dst_int_data_num = dst_int_data + n * dimension;

				for (int i = 0; i < dimension; i++)
				{
					if (dst_int_data_num[i] > max_vals[n])
					{
						max_vals[n] = dst_int_data_num[i];
					}

					if (dst_int_data_num[i] < min_vals[n])
					{
						min_vals[n] = dst_int_data_num[i];
					}
				}
			}

			std::vector<float> temp_float(num * dimension);
			float *dst_float_data = temp_float.data();

			for (int n = 0; n < num; n++)
			{
				float *dst_float_data_num = temp_float.data() + n * dimension;
				int *dst_int_data_num = dst_int_data + n * dimension;

				for (int i = 0; i < dimension; i++)
				{
					dst_float_data_num[i] = float(dst_int_data_num[i] - min_vals[n]) / (max_vals[n] - min_vals[n]);
				}
			}

			dst.reset(new tensor<float>(std::vector<int>{num, 1, 1, dimension}, src->device(), src->order()));
			cudaMemcpy(dst->mutable_gpu_data(), temp_float.data(), num * dimension * sizeof(float), cudaMemcpyDefault);

			return;
		}



		template void tensor_operation_gpu::lbp_feature_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>> &dst, bool map_59);
		template void tensor_operation_gpu::lbp_feature_gpu<char>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>> &dst, bool map_59);
		template void tensor_operation_gpu::lbp_feature_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>> &dst, bool map_59);
		template void tensor_operation_gpu::lbp_feature_gpu<int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>> &dst, bool map_59);
		template void tensor_operation_gpu::lbp_feature_gpu<float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>> &dst, bool map_59);



		template void tensor_operation_gpu::make_border_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst,
			int top, int bottom, int left, int right, borderType type , unsigned char fill_pixel_value);
		template void tensor_operation_gpu::make_border_gpu<char>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>>& dst,
			int top, int bottom, int left, int right, borderType type , char fill_pixel_value);
		template void tensor_operation_gpu::make_border_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>>& dst,
			int top, int bottom, int left, int right, borderType type , unsigned int fill_pixel_value);
		template void tensor_operation_gpu::make_border_gpu<int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>>& dst,
			int top, int bottom, int left, int right, borderType type , int fill_pixel_value);
		template void tensor_operation_gpu::make_border_gpu<float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>>& dst,
			int top, int bottom, int left, int right, borderType type , float fill_pixel_value);



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



		template void tensor_operation_gpu::preprocess_tensors_gpu<unsigned char, float>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<float>> &dst, float means[3], float var);
		template void tensor_operation_gpu::preprocess_tensors_gpu<char, float>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<float>> &dst, float means[3], float var);
		template void tensor_operation_gpu::preprocess_tensors_gpu<unsigned int, float>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<float>> &dst, float means[3], float var);
		template void tensor_operation_gpu::preprocess_tensors_gpu<int, float>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<float>> &dst, float means[3], float var);
		template void tensor_operation_gpu::preprocess_tensors_gpu<float, int>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<int>> &dst, float means[3], float var);
		template void tensor_operation_gpu::preprocess_tensors_gpu<float, float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>> &dst, float means[3], float var);



		template void tensor_operation_gpu::preprocess_tensors_gpu<unsigned char, float>(const tensor<unsigned char> &src, tensor<float> &dst, float means[3], float var);
		template void tensor_operation_gpu::preprocess_tensors_gpu<char, float>(const tensor<char> &src, tensor<float> &dst, float means[3], float var);
		template void tensor_operation_gpu::preprocess_tensors_gpu<unsigned int, float>(const tensor<unsigned int> &src, tensor<float> &dst, float means[3], float var);
		template void tensor_operation_gpu::preprocess_tensors_gpu<int, float>(const tensor<int> &src, tensor<float> &dst, float means[3], float var);
		template void tensor_operation_gpu::preprocess_tensors_gpu<float, int>(const tensor<float> &src, tensor<int> &dst, float means[3], float var);
		template void tensor_operation_gpu::preprocess_tensors_gpu<float, float>(const tensor<float> &src, tensor<float> &dst, float means[3], float var);



		template void tensor_operation_gpu::type_converter_gpu<unsigned char, unsigned int>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned int>> &dst);
		template void tensor_operation_gpu::type_converter_gpu<unsigned char, int>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<int>> &dst);
		template void tensor_operation_gpu::type_converter_gpu<unsigned char, float>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<float>> &dst);
		template void tensor_operation_gpu::type_converter_gpu<float, unsigned char>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<unsigned char>> &dst);
		template void tensor_operation_gpu::type_converter_gpu<float, int>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<int>> &dst);
		template void tensor_operation_gpu::type_converter_gpu<float, unsigned int>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<unsigned int>> &dst);



		template void tensor_operation_gpu::type_converter_gpu<unsigned char, unsigned int>(const tensor<unsigned char> &src, tensor<unsigned int> &dst);
		template void tensor_operation_gpu::type_converter_gpu<unsigned char, int>(const tensor<unsigned char> &src, tensor<int> &dst);
		template void tensor_operation_gpu::type_converter_gpu<unsigned char, float>(const tensor<unsigned char> &src, tensor<float> &dst);
		template void tensor_operation_gpu::type_converter_gpu<float, unsigned char>(const tensor<float> &src, tensor<unsigned char> &dst);
		template void tensor_operation_gpu::type_converter_gpu<float, int>(const tensor<float> &src, tensor<int> &dst);
		template void tensor_operation_gpu::type_converter_gpu<float, unsigned int>(const tensor<float> &src, tensor<unsigned int> &dst);

#ifdef USE_OPENCV
		template void tensor_operation_gpu::tensor2mat_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src, std::vector<cv::Mat>& dst);
		template void tensor_operation_gpu::tensor2mat_gpu<char>(const std::shared_ptr<tensor<char>> &src, std::vector<cv::Mat>& dst);
		template void tensor_operation_gpu::tensor2mat_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src, std::vector<cv::Mat>& dst);
		template void tensor_operation_gpu::tensor2mat_gpu<int>(const std::shared_ptr<tensor<int>> &src, std::vector<cv::Mat>& dst);
		template void tensor_operation_gpu::tensor2mat_gpu<float>(const std::shared_ptr<tensor<float>> &src, std::vector<cv::Mat>& dst);


		template void tensor_operation_gpu::tensor2mat_gpu<unsigned char>(const tensor<unsigned char>& src, std::vector<cv::Mat>& dst);
		template void tensor_operation_gpu::tensor2mat_gpu<char>(const tensor<char>& src, std::vector<cv::Mat>& dst);
		template void tensor_operation_gpu::tensor2mat_gpu<unsigned int>(const tensor<unsigned int>& src, std::vector<cv::Mat>& dst);
		template void tensor_operation_gpu::tensor2mat_gpu<int>(const tensor<int>& src, std::vector<cv::Mat>& dst);
		template void tensor_operation_gpu::tensor2mat_gpu<float>(const tensor<float>& src, std::vector<cv::Mat>& dst);


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
			int dst_height, int dst_width, interpolationType type);
		template void tensor_operation_gpu::resize_gpu<char>(const tensor<char> &src, tensor<char>& dst,
			int dst_height, int dst_width, interpolationType type);
		template void tensor_operation_gpu::resize_gpu<unsigned int>(const tensor<unsigned int> &src, tensor<unsigned int>& dst,
			int dst_height, int dst_width, interpolationType type);
		template void tensor_operation_gpu::resize_gpu<int>(const tensor<int> &src, tensor<int>& dst,
			int dst_height, int dst_width, interpolationType type);
		template void tensor_operation_gpu::resize_gpu<float>(const tensor<float> &src, tensor<float>& dst,
			int dst_height, int dst_width, interpolationType type);



		template void tensor_operation_gpu::rotate_with_center_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst,
			float theta, int &dst_height, int &dst_width, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_center_gpu<char>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>>& dst,
			float theta, int &dst_height, int &dst_width, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_center_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>>& dst,
			float theta, int &dst_height, int &dst_width, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_center_gpu<int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>>& dst,
			float theta, int &dst_height, int &dst_width, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_center_gpu<float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>>& dst,
			float theta, int &dst_height, int &dst_width, int fill_pixel_value, interpolationType type);



		template void tensor_operation_gpu::rotate_with_center_gpu<unsigned char>(const tensor<unsigned char> &src, tensor<unsigned char>& dst,
			float theta, int &dst_height, int &dst_width, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_center_gpu<char>(const tensor<char> &src, tensor<char>& dst,
			float theta, int &dst_height, int &dst_width, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_center_gpu<unsigned int>(const tensor<unsigned int> &src, tensor<unsigned int>& dst,
			float theta, int &dst_height, int &dst_width, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_center_gpu<int>(const tensor<int> &src, tensor<int>& dst,
			float theta, int &dst_height, int &dst_width, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_center_gpu<float>(const tensor<float> &src, tensor<float>& dst,
			float theta, int &dst_height, int &dst_width, int fill_pixel_value, interpolationType type);



		template void tensor_operation_gpu::rotate_with_points_gpu<unsigned char, int>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>> &dst,
			const point<int> &center, float theta, float scale, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_points_gpu<char, int>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>> &dst,
			const point<int> &center, float theta, float scale, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_points_gpu<unsigned int, int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>> &dst,
			const point<int> &center, float theta, float scale, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_points_gpu<int, int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>> &dst,
			const point<int> &center, float theta, float scale, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_points_gpu<float, int>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>> &dst,
			const point<int> &center, float theta, float scale, int fill_pixel_value, interpolationType type);

		template void tensor_operation_gpu::rotate_with_points_gpu<unsigned char, float>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>> &dst,
			const point<float> &center, float theta, float scale, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_points_gpu<char, float>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>> &dst,
			const point<float> &center, float theta, float scale, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_points_gpu<unsigned int, float>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>> &dst,
			const point<float> &center, float theta, float scale, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_points_gpu<int, float>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>> &dst,
			const point<float> &center, float theta, float scale, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_points_gpu<float, float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>> &dst,
			const point<float> &center, float theta, float scale, int fill_pixel_value, interpolationType type);



		template void tensor_operation_gpu::rotate_with_points_gpu<unsigned char, int>(const tensor<unsigned char> &src, tensor<unsigned char> &dst,
			const point<int> &center, float theta, float scale, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_points_gpu<char, int>(const tensor<char> &src, tensor<char> &dst,
			const point<int> &center, float theta, float scale, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_points_gpu<unsigned int, int>(const tensor<unsigned int> &src, tensor<unsigned int> &dst,
			const point<int> &center, float theta, float scale, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_points_gpu<int, int>(const tensor<int> &src, tensor<int> &dst,
			const point<int> &center, float theta, float scale, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_points_gpu<float, int>(const tensor<float> &src, tensor<float> &dst,
			const point<int> &center, float theta, float scale, int fill_pixel_value, interpolationType type);

		template void tensor_operation_gpu::rotate_with_points_gpu<unsigned char, float>(const tensor<unsigned char> &src, tensor<unsigned char> &dst,
			const point<float> &center, float theta, float scale, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_points_gpu<char, float>(const tensor<char> &src, tensor<char> &dst,
			const point<float> &center, float theta, float scale, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_points_gpu<unsigned int, float>(const tensor<unsigned int> &src, tensor<unsigned int> &dst,
			const point<float> &center, float theta, float scale, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_points_gpu<int, float>(const tensor<int> &src, tensor<int> &dst,
			const point<float> &center, float theta, float scale, int fill_pixel_value, interpolationType type);
		template void tensor_operation_gpu::rotate_with_points_gpu<float, float>(const tensor<float> &src, tensor<float> &dst,
			const point<float> &center, float theta, float scale, int fill_pixel_value, interpolationType type);



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



		template void tensor_operation_gpu::rgb2hsv_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst);
		template void tensor_operation_gpu::rgb2hsv_gpu<char>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>>& dst);
		template void tensor_operation_gpu::rgb2hsv_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>>& dst);
		template void tensor_operation_gpu::rgb2hsv_gpu<int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>>& dst);
		template void tensor_operation_gpu::rgb2hsv_gpu<float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>>& dst);



		template void tensor_operation_gpu::rgb2hsv_gpu<unsigned char>(const tensor<unsigned char> &src, tensor<unsigned char>& dst);
		template void tensor_operation_gpu::rgb2hsv_gpu<char>(const tensor<char> &src, tensor<char>& dst);
		template void tensor_operation_gpu::rgb2hsv_gpu<unsigned int>(const tensor<unsigned int> &src, tensor<unsigned int>& dst);
		template void tensor_operation_gpu::rgb2hsv_gpu<int>(const tensor<int> &src, tensor<int>& dst);
		template void tensor_operation_gpu::rgb2hsv_gpu<float>(const tensor<float> &src, tensor<float>& dst);



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
			std::shared_ptr<tensor<unsigned char>>& dst, int thresh, int maxval, thresholdType type);
		template void tensor_operation_gpu::threshold_gpu<char>(const std::shared_ptr<tensor<char>> &src,
			std::shared_ptr<tensor<char>>& dst, int thresh, int maxval, thresholdType type);
		template void tensor_operation_gpu::threshold_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src,
			std::shared_ptr<tensor<unsigned int>>& dst, int thresh, int maxval, thresholdType type);
		template void tensor_operation_gpu::threshold_gpu<int>(const std::shared_ptr<tensor<int>> &src,
			std::shared_ptr<tensor<int>>& dst, int thresh, int maxval, thresholdType type);
		template void tensor_operation_gpu::threshold_gpu<float>(const std::shared_ptr<tensor<float>> &src,
			std::shared_ptr<tensor<float>>& dst, int thresh, int maxval, thresholdType type);



		template void tensor_operation_gpu::threshold_gpu<unsigned char>(const tensor<unsigned char> &src,
			tensor<unsigned char>& dst, int thresh, int maxval, thresholdType type);
		template void tensor_operation_gpu::threshold_gpu<char>(const tensor<char> &src,
			tensor<char>& dst, int thresh, int maxval, thresholdType type);
		template void tensor_operation_gpu::threshold_gpu<unsigned int>(const tensor<unsigned int> &src,
			tensor<unsigned int>& dst, int thresh, int maxval, thresholdType type);
		template void tensor_operation_gpu::threshold_gpu<int>(const tensor<int> &src,
			tensor<int>& dst, int thresh, int maxval, thresholdType type);
		template void tensor_operation_gpu::threshold_gpu<float>(const tensor<float> &src,
			tensor<float>& dst, int thresh, int maxval, thresholdType type);



		template void tensor_operation_gpu::warp_affine_gpu<unsigned char, int>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst,
			const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);
		template void tensor_operation_gpu::warp_affine_gpu<unsigned char, float>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>>& dst,
			const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);

		template void tensor_operation_gpu::warp_affine_gpu<char, int>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>>& dst,
			const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);
		template void tensor_operation_gpu::warp_affine_gpu<char, float>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>>& dst,
			const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);

		template void tensor_operation_gpu::warp_affine_gpu<unsigned int, int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>>& dst,
			const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);
		template void tensor_operation_gpu::warp_affine_gpu<unsigned int, float>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>>& dst,
			const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);

		template void tensor_operation_gpu::warp_affine_gpu<int, int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>>& dst,
			const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);
		template void tensor_operation_gpu::warp_affine_gpu<int, float>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>>& dst,
			const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);

		template void tensor_operation_gpu::warp_affine_gpu<float, int>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>>& dst,
			const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);
		template void tensor_operation_gpu::warp_affine_gpu<float, float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>>& dst,
			const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);



		template void tensor_operation_gpu::warp_affine_gpu<unsigned char, int>(const tensor<unsigned char> &src, tensor<unsigned char>& dst,
			const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);
		template void tensor_operation_gpu::warp_affine_gpu<unsigned char, float>(const tensor<unsigned char> &src, tensor<unsigned char>& dst,
			const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);

		template void tensor_operation_gpu::warp_affine_gpu<char, int>(const tensor<char> &src, tensor<char>& dst,
			const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);
		template void tensor_operation_gpu::warp_affine_gpu<char, float>(const tensor<char> &src, tensor<char>& dst,
			const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);

		template void tensor_operation_gpu::warp_affine_gpu<unsigned int, int>(const tensor<unsigned int> &src, tensor<unsigned int>& dst,
			const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);
		template void tensor_operation_gpu::warp_affine_gpu<unsigned int, float>(const tensor<unsigned int> &src, tensor<unsigned int>& dst,
			const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);

		template void tensor_operation_gpu::warp_affine_gpu<int, int>(const tensor<int> &src, tensor<int>& dst,
			const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);
		template void tensor_operation_gpu::warp_affine_gpu<int, float>(const tensor<int> &src, tensor<int>& dst,
			const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);

		template void tensor_operation_gpu::warp_affine_gpu<float, int>(const tensor<float> &src, tensor<float>& dst,
			const std::vector<point<int>> &src_point, const std::vector<point<int>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);
		template void tensor_operation_gpu::warp_affine_gpu<float, float>(const tensor<float> &src, tensor<float>& dst,
			const std::vector<point<float>> &src_point, const std::vector<point<float>> &dst_point, int fill_pixel_value, excalibur::interpolationType type);



		template void tensor_operation_gpu::gaussian_blur_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>> &dst, int ksize);
		template void tensor_operation_gpu::gaussian_blur_gpu<char>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>> &dst, int ksize);
		template void tensor_operation_gpu::gaussian_blur_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>> &dst, int ksize);
		template void tensor_operation_gpu::gaussian_blur_gpu<int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>> &dst, int ksize);
		template void tensor_operation_gpu::gaussian_blur_gpu<float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>> &dst, int ksize);



		template void tensor_operation_gpu::gaussian_blur_gpu<unsigned char>(const tensor<unsigned char> &src, tensor<unsigned char> &dst, int ksize);
		template void tensor_operation_gpu::gaussian_blur_gpu<char>(const tensor<char> &src, tensor<char> &dst, int ksize);
		template void tensor_operation_gpu::gaussian_blur_gpu<unsigned int>(const tensor<unsigned int> &src, tensor<unsigned int> &dst, int ksize);
		template void tensor_operation_gpu::gaussian_blur_gpu<int>(const tensor<int> &src, tensor<int> &dst, int ksize);
		template void tensor_operation_gpu::gaussian_blur_gpu<float>(const tensor<float> &src, tensor<float> &dst, int ksize);



		template void tensor_operation_gpu::mean_value_blur_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src, std::shared_ptr<tensor<unsigned char>> &dst, int ksize);
		template void tensor_operation_gpu::mean_value_blur_gpu<char>(const std::shared_ptr<tensor<char>> &src, std::shared_ptr<tensor<char>> &dst, int ksize);
		template void tensor_operation_gpu::mean_value_blur_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src, std::shared_ptr<tensor<unsigned int>> &dst, int ksize);
		template void tensor_operation_gpu::mean_value_blur_gpu<int>(const std::shared_ptr<tensor<int>> &src, std::shared_ptr<tensor<int>> &dst, int ksize);
		template void tensor_operation_gpu::mean_value_blur_gpu<float>(const std::shared_ptr<tensor<float>> &src, std::shared_ptr<tensor<float>> &dst, int ksize);



		template void tensor_operation_gpu::mean_value_blur_gpu<unsigned char>(const tensor<unsigned char> &src, tensor<unsigned char> &dst, int ksize);
		template void tensor_operation_gpu::mean_value_blur_gpu<char>(const tensor<char> &src, tensor<char> &dst, int ksize);
		template void tensor_operation_gpu::mean_value_blur_gpu<unsigned int>(const tensor<unsigned int> &src, tensor<unsigned int> &dst, int ksize);
		template void tensor_operation_gpu::mean_value_blur_gpu<int>(const tensor<int> &src, tensor<int> &dst, int ksize);
		template void tensor_operation_gpu::mean_value_blur_gpu<float>(const tensor<float> &src, tensor<float> &dst, int ksize);



		template void tensor_operation_gpu::sobel_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src,
			std::shared_ptr<tensor<unsigned char>> &dst, int dx, int dy);
		template void tensor_operation_gpu::sobel_gpu<char>(const std::shared_ptr<tensor<char>> &src,
			std::shared_ptr<tensor<char>> &dst, int dx, int dy);
		template void tensor_operation_gpu::sobel_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src,
			std::shared_ptr<tensor<unsigned int>> &dst, int dx, int dy);
		template void tensor_operation_gpu::sobel_gpu<int>(const std::shared_ptr<tensor<int>> &src,
			std::shared_ptr<tensor<int>> &dst, int dx, int dy);
		template void tensor_operation_gpu::sobel_gpu<float>(const std::shared_ptr<tensor<float>> &src,
			std::shared_ptr<tensor<float>> &dst, int dx, int dy);



		template void tensor_operation_gpu::sobel_gpu<unsigned char>(const tensor<unsigned char> &src, tensor<unsigned char> &dst, int dx, int dy);
		template void tensor_operation_gpu::sobel_gpu<char>(const tensor<char> &src, tensor<char> &dst, int dx, int dy);
		template void tensor_operation_gpu::sobel_gpu<unsigned int>(const tensor<unsigned int> &src, tensor<unsigned int> &dst, int dx, int dy);
		template void tensor_operation_gpu::sobel_gpu<int>(const tensor<int> &src, tensor<int> &dst, int dx, int dy);
		template void tensor_operation_gpu::sobel_gpu<float>(const tensor<float> &src, tensor<float> &dst, int dx, int dy);



		template void tensor_operation_gpu::morph_gpu<unsigned char>(const std::shared_ptr<tensor<unsigned char>> &src,
			std::shared_ptr<tensor<unsigned char>> &dst, excalibur::morphType type, int ksize);
		template void tensor_operation_gpu::morph_gpu<char>(const std::shared_ptr<tensor<char>> &src,
			std::shared_ptr<tensor<char>> &dst, excalibur::morphType type, int ksize);
		template void tensor_operation_gpu::morph_gpu<unsigned int>(const std::shared_ptr<tensor<unsigned int>> &src,
			std::shared_ptr<tensor<unsigned int>> &dst, excalibur::morphType type, int ksize);
		template void tensor_operation_gpu::morph_gpu<int>(const std::shared_ptr<tensor<int>> &src,
			std::shared_ptr<tensor<int>> &dst, excalibur::morphType type, int ksize);
		template void tensor_operation_gpu::morph_gpu<float>(const std::shared_ptr<tensor<float>> &src,
			std::shared_ptr<tensor<float>> &dst, excalibur::morphType type, int ksize);



		template void tensor_operation_gpu::morph_gpu<unsigned char>(const tensor<unsigned char> &src, tensor<unsigned char> &dst, excalibur::morphType type, int ksize);
		template void tensor_operation_gpu::morph_gpu<char>(const tensor<char> &src, tensor<char> &dst, excalibur::morphType type, int ksize);
		template void tensor_operation_gpu::morph_gpu<unsigned int>(const tensor<unsigned int> &src, tensor<unsigned int> &dst, excalibur::morphType type, int ksize);
		template void tensor_operation_gpu::morph_gpu<int>(const tensor<int> &src, tensor<int> &dst, excalibur::morphType type, int ksize);
		template void tensor_operation_gpu::morph_gpu<float>(const tensor<float> &src, tensor<float> &dst, excalibur::morphType type, int ksize);
	}
}
#endif // USE_CUDA