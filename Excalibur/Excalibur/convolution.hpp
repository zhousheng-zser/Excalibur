#pragma once
#ifndef _CONVOLUTION_HPP_
#define _CONVOLUTION_HPP_
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
		class convolution
		{
			std::shared_ptr<tensor<float>> weights_;
			std::shared_ptr<tensor<float>> bias_;
			std::shared_ptr<tensor<float>> col_buffer_;
			int device_;
			orderType order_;


			int tile_size_;
			int h_tile_num_;
			int w_tile_num_;
			int V_num_;//the quantity of V
			int U_num_;//the quantity of U
			float* U_;
			float* V_;

			////winograd F(4,3), error rate is 10%, abandon
			//int m_ = 4;
			//const float A_[24] = { 1.0f, 0.0f, 0.0f, 0.0f,
			//	                   1.0f, 1.0f, 1.0f, 1.0f,
			//	                   1.0f,-1.0f, 1.0f,-1.0f,
			//	                   1.0f, 2.0f, 4.0f, 8.0f,
			//	                   1.0f,-2.0f, 4.0f,-8.0f,
			//	                   0.0f, 0.0f, 0.0f, 1.0f };

			//const float AT_[24] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
			//	                    0.0f, 1.0f,-1.0f, 2.0f,-2.0f, 0.0f,
			//	                    0.0f, 1.0f, 1.0f, 4.0f, 4.0f, 0.0f,
			//	                    0.0f, 1.0f,-1.0f, 8.0f,-8.0f, 1.0f };

			//const float B_[36] = { 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
			//	                   0.0f,-4.0f, 4.0f,-2.0f, 2.0f, 4.0f,
			//	                  -5.0f,-4.0f,-4.0f,-1.0f,-1.0f, 0.0f,
			//	                   0.0f, 1.0f,-1.0f, 2.0f,-2.0f,-5.0f,
			//	                   1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
			//	                   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };

			//const float BT_[36] = { 4.0f, 0.0f,-5.0f, 0.0f, 1.0f, 0.0f,
			//	                   0.0f,-4.0f,-4.0f, 1.0f, 1.0f, 0.0f,
			//	                   0.0f, 4.0f,-4.0f,-1.0f, 1.0f, 0.0f,
			//	                   0.0f,-2.0f,-1.0f, 2.0f, 1.0f, 0.0f,
			//	                   0.0f, 2.0f,-1.0f,-2.0f, 1.0f, 0.0f,
			//	                   0.0f, 4.0f, 0.0f,-5.0f, 0.0f, 1.0f };

			//const float G_[18] = {  1.0f / 4,   0.0f,       0.0f,
			//	                  -1.0f / 6,  -1.0f / 6,  -1.0f / 6,
			//	                  -1.0f / 6,   1.0f / 6,  -1.0f / 6,
			//	                   1.0f / 24,  1.0f / 12,  1.0f / 6,
			//	                   1.0f / 24, -1.0f / 12,  1.0f / 6,
			//	                   0.0f,       0.0f,       1.0f };

			//const float GT_[18] = { 1.0f / 4, -1.0f / 6, -1.0f / 6, 1.0f / 24, 1.0f / 24, 0.0f,
			//	                    0.0f,     -1.0f / 6,  1.0f / 6, 1.0f / 12,-1.0f / 12, 0.0f,
			//	                    0.0f,     -1.0f / 6, -1.0f / 6, 1.0f / 6,  1.0f / 6,  1.0f };

			//winograd F(2,3)
			int m_ = 2;
			const float A_[8] = { 1.0f,  0.0f,
				                  1.0f,  1.0f,
				                  1.0f, -1.0f,
				                  0.0f, -1.0f };

			const float AT_[8] = { 1.0f, 1.0f,  1.0f,  0.0f,
				                   0.0f, 1.0f, -1.0f, -1.0f };

			const float B_[16] = { 1.0f, 0.0f,  0.0f,  0.0f,
				                   0.0f, 1.0f, -1.0f,  1.0f,
				                  -1.0f, 1.0f,  1.0f,  0.0f,
				                   0.0f, 0.0f,  0.0f, -1.0f };

			const float BT_[16] = { 1.0f,  0.0f, -1.0f,  0.0f,
				                    0.0f,  1.0f,  1.0f,  0.0f,
				                    0.0f, -1.0f,  1.0f,  0.0f,
				                    0.0f,  1.0f,  0.0f, -1.0f };

			const float G_[12] = { 1.0f,  0.0f,  0.0f,
				                   0.5f,  0.5f,  0.5f,
				                   0.5f, -0.5f,  0.5f,
				                   0.0f,  0.0f,  1.0f };

			const float GT_[12] = { 1.0f, 0.5f,  0.5f, 0.0f,
				                    0.0f, 0.5f, -0.5f, 0.0f,
				                    0.0f, 0.5f,  0.5f, 1.0f };

			/// parameters
			int input_Channel_;
			int output_Channel_;
			int kernelSize_;
			int stride_;
			int pad_;
			///
			std::vector<int> intput_shape_;
			std::vector<int> output_shape_;
			int group_;
			int output_dim_h_;
			int output_dim_w_;
			int out_spatial_dim_;
			int weight_offset_;
			bool bias_term_;
			bool isfirst;
			int last_height;
			int last_width;
			float* gpu_temp_col_buffer_;
			///
			int num_kernels_im2col_;
			int num_kernels_col2im_;
			int conv_out_channels_;
			int conv_in_channels_;
			int conv_out_spatial_dim_;
			int kernel_dim_;
			int col_offset_;
			int output_offset_;
			std::shared_ptr<tensor<float>> bias_multiplier_;
			///
			inline void conv_im2col_cpu(const float* data, float* col_buff, int num = 1);
			inline void conv_col2im_cpu(const float* col_buff, float* data);
#ifdef USE_CUDA
#ifdef USE_CUDNN
			float one = 1.0, zero = 0.0;
			size_t size;
			cudnnHandle_t cudnn_handle_ = nullptr;
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
#endif
			void conv_im2col_gpu(const float* data, float* col_buff);
			void conv_col2im_gpu(const float* col_buff, float* data);
#endif
			void setup_internal_params();
			void setup_internal_params(int group);
		public:
			convolution(int input_Channel, int output_Channel, int kernelSize,
				int stride, int pad, bool bias_term, int device);
			convolution(int input_Channel, int output_Channel, int kernelSize, int group,
				int stride, int pad, bool bias_term, int device);
			~convolution();
			void set_weights(float* weights);
			void set_bias(float* bias);
			void set_depthwise()
			{
				group_ = output_Channel_;
			}

			void Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
			void Forward_cpu_native(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
			void Forward_cpu_winograd(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);

#ifdef USE_CUDA
			void Forward_gpu_native(cublasHandle_t cublas_handle_, const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
#ifdef USE_CUDNN
			void Forward_gpu_cudnn(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
#endif
#endif
		private:
			void forward_cpu_gemm(const float* input, const float* weights, float* output, bool skip_im2col = false);
			void forward_cpu_bias(float* output, const float* bias);

#ifdef USE_MKL
			void Forward_cpu_batch(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
			void forward_cpu_gemm_batch(const float* input, const float* weights, float* output, int top_dim, int num, bool skip_im2col = false);
			void forward_cpu_bias_batch(float* output, const float* bias, int num);
#endif
#ifdef USE_CUDA
			void forward_gpu_gemm(cublasHandle_t cublas_handle_, const float* input, const float* weights, float* output, bool skip_im2col = false);
			void forward_gpu_bias(cublasHandle_t cublas_handle_, float* output, const float* bias);
			void forward_gpu_depthwise_native(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top);
#endif
		};
	}
}
#endif //_CONVOLUTION_HPP_