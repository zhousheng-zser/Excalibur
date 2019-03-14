#pragma once
#ifndef _BASE_CONV_HPP_
#define _BASE_CONV_HPP_
#include <glasssix\tensor.hpp>
#include "im2col.hpp"
#include "math_functions.hpp"
#include <memory>
#ifdef USE_CUDNN
#include "cudnn.hpp"
#endif

namespace glasssix
{
	namespace excalibur
	{
		class baseconv
		{
		public:
			
			std::shared_ptr<tensor<float>> weights_;
			std::shared_ptr<tensor<float>> bias_;
			std::shared_ptr<tensor<float>> col_buffer_;
			int device_;
			orderType order_;

			/// parameters
			int input_Channel_;
			int output_Channel_;
			int kernelSize_;
			int stride_;
			int pad_;

			///
			std::vector<int> intput_shape_;
			std::vector<int> output_shape_;
			int num_;
			int group_;
			int input_dim_h_;
			int input_dim_w_;
			int input_spatial_dim_;
			int bottom_dim_;
			int output_dim_h_;
			int output_dim_w_;
			int output_spatial_dim_;
			int top_dim_;
			bool isfirst;
			int last_height;
			int last_width;
			float* gpu_temp_col_buffer_;
			int kernel_dim_;
			int weight_offset_;
			int col_offset_;
			int output_offset_;
			bool bias_term_;
			std::shared_ptr<tensor<float>> bias_multiplier_;


			baseconv() {}

			baseconv(int input_Channel, int output_Channel, int group, int kernelSize, int stride, int pad, bool bias_term, int device)
			{
				CHECK_EQ(output_Channel % group, 0);
				CHECK_EQ(input_Channel % group, 0);
				input_Channel_ = input_Channel;
				output_Channel_ = output_Channel;
				kernelSize_ = kernelSize;
				stride_ = stride;
				pad_ = pad;
				bias_term_ = bias_term;
				device_ = device;
				group_ = group;
				weights_.reset(new tensor<float>(std::vector<int>{input_Channel_*output_Channel_*kernelSize_*kernelSize_ / group}, device_));
				bias_.reset(new tensor<float>(std::vector<int>{output_Channel_}, device_));
				kernel_dim_ = input_Channel_*kernelSize_*kernelSize_;
				weight_offset_ = kernelSize_*kernelSize_;
			}

			virtual ~baseconv() {};

			void set_bias(float* bias)
			{
				if (bias_term_)
				{
					bias_->set_cpu_data(bias);
				}
			}

			void set_weights(float* weights)
			{
				weights_->set_cpu_data(weights);
			}

			virtual void Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top) = 0;

		protected:
			virtual void forward_gemm(const float* input, const float* weights, float* output, bool skip_im2col = false) = 0;
			virtual void forward_bias(float* output, const float* bias) = 0;			

#ifdef USE_CUDA
		public:
			virtual void Forward(cublasHandle_t cublas_handle_, const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top) = 0;
		protected:
			virtual void forward_gemm(cublasHandle_t cublas_handle_, const float* input, const float* weights, float* output, bool skip_im2col = false) = 0; 
			virtual void forward_bias(cublasHandle_t cublas_handle_, float* output, const float* bias) = 0;			
#ifdef USE_CUDNN
		public:
			virtual void Forward(cudnnHandle_t cudnn_handle_, const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top) = 0;		
#endif//!USE_CUDNN
#endif//!USE_CUDA

		protected:
			void conv_im2col_cpu(const float* data, float* col_buff)
			{
				if (order_ == NCHW)
				{
					im2col_cpu(data, input_Channel_, intput_shape_[2], intput_shape_[3], kernelSize_,
						kernelSize_, pad_, pad_, stride_, stride_, 1, 1, col_buff, order_);
				}
				else if (order_ == NHWC)
				{
					im2col_cpu(data, input_Channel_, intput_shape_[1], intput_shape_[2], kernelSize_,
						kernelSize_, pad_, pad_, stride_, stride_, 1, 1, col_buff, order_);
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}

			void conv_col2im_cpu(const float* col_buff, float* data)
			{
				col2im_cpu(col_buff, input_Channel_, intput_shape_[2], intput_shape_[3], kernelSize_,
					kernelSize_, pad_, pad_, stride_, stride_, 1, 1, data);
			}

#ifdef USE_CUDA
			void conv_im2col_gpu(const float* data, float* col_buff)
			{
				if (order_ == NCHW)
				{
					im2col_gpu(data, input_Channel_, intput_shape_[2], intput_shape_[3], kernelSize_,
						kernelSize_, pad_, pad_, stride_, stride_, 1, 1, col_buff, order_);
				}
				else if (order_ == NHWC)
				{
					im2col_gpu(data, input_Channel_, intput_shape_[1], intput_shape_[2], kernelSize_,
						kernelSize_, pad_, pad_, stride_, stride_, 1, 1, col_buff, order_);
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}

			void conv_col2im_gpu(const float* col_buff, float* data)
			{
				col2im_gpu(col_buff, input_Channel_, intput_shape_[2], intput_shape_[3], kernelSize_,
					kernelSize_, pad_, pad_, stride_, stride_, 1, 1, data);
			}
#endif //!USE_CUDA

#ifdef USE_CUDA
#ifdef USE_CUDNN
			float one = 1.0, zero = 0.0;
			size_t size;
			cudnnTensorDescriptor_t xdesc;
			cudnnTensorDescriptor_t	ydesc;
			cudnnTensorDescriptor_t bdesc;
			cudnnFilterDescriptor_t wdesc;
			cudnnConvolutionDescriptor_t conv_desc;
			// algorithms for forward and backwards convolutions
			cudnnConvolutionFwdAlgo_t fwd_algo_;
			size_t workspace_limit_bytes = 8 * 1024 * 1024;
			float *extra = nullptr;
			size_t current_size;
#endif//!USE_CUDNN
#endif//!USE_CUDA
		};
	}
}
#endif //_BASE_CONV_HPP_