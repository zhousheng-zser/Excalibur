#pragma once
#ifndef _BASE_CONV_HPP_
#define _BASE_CONV_HPP_
#include <glasssix/tensor.hpp>
#include "im2col.hpp"
#include "math_functions.hpp"
#include <memory>
#ifdef USE_CUDNN
#include "cudnn.hpp"
#endif

#include "../../include/Julius/simd_helper.hpp"

namespace glasssix
{
	namespace excalibur
	{
		class baseconv
		{
		public:
			bool int8_quantization_;
			std::shared_ptr<tensor<signed char>> weights_int8_;
			std::shared_ptr<tensor<signed char>> col_buffer_int8_;
			std::shared_ptr<tensor<signed char>> bottom_int8_;
			std::shared_ptr<tensor<int>> top_int32_;
			std::shared_ptr<tensor<float>> scales_;
			signed char *col_buffer_int8_data, *bottom_int8_data;
			const signed char *weights_int8_data;
			int *top_int32_data;
			const float *scales_data;

#if SIMD_TYPE >= SIMDTYPE_SSE
			std::shared_ptr<tensor<float>> bottom_round_ = std::make_shared<tensor<float>>(std::vector<int>{mm_align_size});
			float* bottom_round_data_ = bottom_round_->mutable_cpu_data();
#endif // SIMD_TYPE >= SIMDTYPE_SSE

			//float32
			std::shared_ptr<tensor<float>> weights_;
			std::shared_ptr<tensor<float>> col_buffer_;
			std::shared_ptr<tensor<float>> bias_;
			std::shared_ptr<tensor<float>> bias_multiplier_;
			float *col_buffer_data, *bias_multiplier_data;
			const float *weights_data, *bias_data;

			float *top_data;
			const float *bottom_data;

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
			bool isfirst = true;
			int last_height;
			int last_width;
			float* gpu_temp_col_buffer_;
			int kernel_dim_;
			int weight_offset_;
			int col_offset_;
			int output_offset_;
			bool bias_term_;
			
			baseconv() {}

			baseconv(int input_Channel, int output_Channel, int group, int kernelSize, int stride, int pad, bool bias_term, int device = -1, bool int8_quantization = false)
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
				int8_quantization_ = int8_quantization;

				if (int8_quantization_)
				{
					scales_.reset(new tensor<float>(std::vector<int>{ 1 + group_ }, device_));
					weights_int8_.reset(new tensor<signed char>(std::vector<int>{input_Channel_*output_Channel_*kernelSize_*kernelSize_ / group}, device_));
			    }
				else
				{
					weights_.reset(new tensor<float>(std::vector<int>{input_Channel_*output_Channel_*kernelSize_*kernelSize_ / group}, device_));
				}

				bias_.reset(new tensor<float>(std::vector<int>{output_Channel_}, device_));
				kernel_dim_ = input_Channel_*kernelSize_*kernelSize_;
				weight_offset_ = kernelSize_*kernelSize_;

				//winograd
				if ((kernelSize_ == 3) && (stride_ == 1))
				{
					useWinograd23 = true;
					tile_size_ = m_ + kernelSize_ - 1;//m+r-1
					tile_length_ = tile_size_ * tile_size_;
					kernel_length_ = kernelSize_ * kernelSize_;
					m_length_ = m_ * m_;
					U_num_ = output_Channel_ * input_Channel_ / group_;

					//U=G*g*GT,so U has the same number as kernel g, there are tile_size_ * tile_size_ elements in single U
					if (int8_quantization)
					{
						U_int16.reset(new tensor<short>(std::vector<int>{U_num_ * tile_length_}));
						U_int16_data = U_int16->mutable_cpu_data();
					}
					else
					{
						U_.reset(new tensor<float>(std::vector<int>{U_num_ * tile_length_}));
						U_data = U_->mutable_cpu_data();
					}
				}
			}

			virtual ~baseconv() 
			{
#ifdef USE_CUDA
#ifdef USE_CUDNN
				if (device_ >= 0)
				{
					if (xdesc != nullptr)
					{
						CUDNN_CHECK(cudnnDestroyTensorDescriptor(xdesc));
					}

					if (ydesc != nullptr)
					{
						CUDNN_CHECK(cudnnDestroyTensorDescriptor(ydesc));
					}

					if (wdesc != nullptr)
					{
						CUDNN_CHECK(cudnnDestroyFilterDescriptor(wdesc));
					}

					if (conv_desc != nullptr)
					{
						CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
					}
				    
					if (bias_term_)
					{
						if (bdesc != nullptr)
						{
							CUDNN_CHECK(cudnnDestroyTensorDescriptor(bdesc));
						}
					}
					
					if (extra != nullptr)
					{
				        cudaFree(extra);
					}
				}
#endif
#endif // USE_CUDA

			};

			virtual void set_bias(float* bias)
			{
				if (bias_term_)
				{
					bias_->set_cpu_data(bias);

					if (device_ < 0)
					{
						bias_data = bias_->cpu_data();
					}
					else
					{
						bias_data = bias_->gpu_data();
					}
				}				
			}

			virtual void set_weights(float* weights)
			{
				weights_->set_cpu_data(weights);

				if (device_ < 0)
				{
					weights_data = weights_->cpu_data();

					if (useWinograd23)
					{
						//calculate U_
#ifdef _OPENMP
#pragma omp parallel for
#endif
						for (int n = 0; n < U_num_; ++n)
						{
							calculate_GgGT(weights_data + kernel_length_ * n, U_data + tile_length_ * n);//calculate U
						}
					}
				}
				else
				{
					weights_data = weights_->gpu_data();
				}
		    }

			void set_weights(signed char* weights_int8)
			{
				weights_int8_->set_cpu_data(weights_int8);
				if (device_ < 0)
				{
					weights_int8_data = weights_int8_->cpu_data();
				}
				else
				{
					weights_int8_data = weights_int8_->gpu_data();
				}
			}

			void set_scales(float* scales)
			{
				scales_->set_cpu_data(scales);
				if (device_ < 0)
				{
					scales_data = scales_->cpu_data();
				}
				else
				{
					scales_data = scales_->gpu_data();
				}
			}

			virtual void Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top) = 0;

			//winograd
			bool useWinograd23 = false;
			int m_ = 2;
			int m_length_;
			int kernel_length_;
			int tile_size_;
			int tile_length_;
			int h_tile_num_;
			int w_tile_num_;
			int V_num_;//the quantity of V
			int U_num_;//the quantity of U

			std::shared_ptr<tensor<float>> U_, V_;
			float *U_data, *V_data;

			std::shared_ptr<tensor<short>> U_int16, V_int16;
			short *U_int16_data, *V_int16_data;

			//fp32
			inline void calculate_GgGT(const float *weight_data, float *u_data)
			{
				u_data[0] = weight_data[0];
				u_data[1] = (weight_data[0] + weight_data[1] + weight_data[2]) / 2;
				u_data[2] = (weight_data[0] - weight_data[1] + weight_data[2]) / 2;
				u_data[3] = weight_data[2];
				u_data[4] = (weight_data[0] + weight_data[3] + weight_data[6]) / 2;
				u_data[5] = (weight_data[0] + weight_data[1] + weight_data[2] +
					weight_data[3] + weight_data[4] + weight_data[5] +
					weight_data[6] + weight_data[7] + weight_data[8]) / 4;
				u_data[6] = (weight_data[0] - weight_data[1] + weight_data[2] +
					weight_data[3] - weight_data[4] + weight_data[5] +
					weight_data[6] - weight_data[7] + weight_data[8]) / 4;
				u_data[7] = (weight_data[2] + weight_data[5] + weight_data[8]) / 2;
				u_data[8] = (weight_data[0] - weight_data[3] + weight_data[6]) / 2;
				u_data[9] = (weight_data[0] + weight_data[1] + weight_data[2] -
					weight_data[3] - weight_data[4] - weight_data[5] +
					weight_data[6] + weight_data[7] + weight_data[8]) / 4;
				u_data[10] = (weight_data[0] - weight_data[1] + weight_data[2] -
					weight_data[3] + weight_data[4] - weight_data[5] +
					weight_data[6] - weight_data[7] + weight_data[8]) / 4;
				u_data[11] = (weight_data[2] - weight_data[5] + weight_data[8]) / 2;
				u_data[12] = weight_data[6];
				u_data[13] = (weight_data[6] + weight_data[7] + weight_data[8]) / 2;
				u_data[14] = (weight_data[6] - weight_data[7] + weight_data[8]) / 2;
				u_data[15] = weight_data[8];
			}

			inline void calculate_BTdB(const float *tile_data, float *v_data)
			{
				v_data[0] = tile_data[0] - tile_data[2] - tile_data[8] + tile_data[10];
				v_data[1] = tile_data[1] + tile_data[2] - tile_data[9] - tile_data[10];
				v_data[2] = -tile_data[1] + tile_data[2] + tile_data[9] - tile_data[10];
				v_data[3] = tile_data[1] - tile_data[3] - tile_data[9] + tile_data[11];
				v_data[4] = tile_data[4] - tile_data[6] + tile_data[8] - tile_data[10];
				v_data[5] = tile_data[5] + tile_data[6] + tile_data[9] + tile_data[10];
				v_data[6] = -tile_data[5] + tile_data[6] - tile_data[9] + tile_data[10];
				v_data[7] = tile_data[5] - tile_data[7] + tile_data[9] - tile_data[11];
				v_data[8] = -tile_data[4] + tile_data[6] + tile_data[8] - tile_data[10];
				v_data[9] = -tile_data[5] - tile_data[6] + tile_data[9] + tile_data[10];
				v_data[10] = tile_data[5] - tile_data[6] - tile_data[9] + tile_data[10];
				v_data[11] = -tile_data[5] + tile_data[7] + tile_data[9] - tile_data[11];
				v_data[12] = tile_data[4] - tile_data[6] - tile_data[12] + tile_data[14];
				v_data[13] = tile_data[5] + tile_data[6] - tile_data[13] - tile_data[14];
				v_data[14] = -tile_data[5] + tile_data[6] + tile_data[13] - tile_data[14];
				v_data[15] = tile_data[5] - tile_data[7] - tile_data[13] + tile_data[15];
			}

			inline void calculate_ATmA(const float *m_data, float *result)
			{
				result[0] = m_data[0] + m_data[1] + m_data[2] + m_data[4] + m_data[5] + m_data[6] + m_data[8] + m_data[9] + m_data[10];
				result[1] = m_data[1] - m_data[2] - m_data[3] + m_data[5] - m_data[6] - m_data[7] + m_data[9] - m_data[10] - m_data[11];
				result[2] = m_data[4] + m_data[5] + m_data[6] - m_data[8] - m_data[9] - m_data[10] - m_data[12] - m_data[13] - m_data[14];
				result[3] = m_data[5] - m_data[6] - m_data[7] - m_data[9] + m_data[10] + m_data[11] - m_data[13] + m_data[14] + m_data[15];
			}

			//int8
			inline void calculate_GgGT(const signed char *weight_data, short *u_data)
			{
				//mutiply 4 to avoid accuracy loss
				u_data[0] = weight_data[0] * 4;
				u_data[1] = (short(weight_data[0]) + short(weight_data[1]) + short(weight_data[2])) * 2;
				u_data[2] = (short(weight_data[0]) - short(weight_data[1]) + short(weight_data[2])) * 2;
				u_data[3] = short(weight_data[2]) * 4;
				u_data[4] = (short(weight_data[0]) + short(weight_data[3]) + short(weight_data[6])) * 2;
				u_data[5] = (short(weight_data[0]) + short(weight_data[1]) + short(weight_data[2]) +
					short(weight_data[3]) + short(weight_data[4]) + short(weight_data[5]) +
					short(weight_data[6]) + short(weight_data[7]) + short(weight_data[8]));
				u_data[6] = (short(weight_data[0]) - short(weight_data[1]) + short(weight_data[2]) +
					short(weight_data[3]) - short(weight_data[4]) + short(weight_data[5]) +
					short(weight_data[6]) - short(weight_data[7]) + short(weight_data[8]));
				u_data[7] = (short(weight_data[2]) + short(weight_data[5]) + short(weight_data[8])) * 2;
				u_data[8] = (short(weight_data[0]) - short(weight_data[3]) + short(weight_data[6])) * 2;
				u_data[9] = (short(weight_data[0]) + short(weight_data[1]) + short(weight_data[2]) -
					short(weight_data[3]) - short(weight_data[4]) - short(weight_data[5]) +
					short(weight_data[6]) + short(weight_data[7]) + short(weight_data[8]));
				u_data[10] = (short(weight_data[0]) - short(weight_data[1]) + short(weight_data[2]) -
					short(weight_data[3]) + short(weight_data[4]) - short(weight_data[5]) +
					short(weight_data[6]) - short(weight_data[7]) + short(weight_data[8]));
				u_data[11] = (short(weight_data[2]) - short(weight_data[5]) + short(weight_data[8])) * 2;
				u_data[12] = weight_data[6] * 4;
				u_data[13] = (short(weight_data[6]) + short(weight_data[7]) + short(weight_data[8])) * 2;
				u_data[14] = (short(weight_data[6]) - short(weight_data[7]) + short(weight_data[8])) * 2;
				u_data[15] = weight_data[8] * 4;
			}

			inline void calculate_BTdB(const signed char *tile_data, short *v_data)
			{
				v_data[0] = short(tile_data[0]) - short(tile_data[2]) - short(tile_data[8]) + short(tile_data[10]);
				v_data[1] = short(tile_data[1]) + short(tile_data[2]) - short(tile_data[9]) - short(tile_data[10]);
				v_data[2] = -short(tile_data[1]) + short(tile_data[2]) + short(tile_data[9]) - short(tile_data[10]);
				v_data[3] = short(tile_data[1]) - short(tile_data[3]) - short(tile_data[9]) + short(tile_data[11]);
				v_data[4] = short(tile_data[4]) - short(tile_data[6]) + short(tile_data[8]) - short(tile_data[10]);
				v_data[5] = short(tile_data[5]) + short(tile_data[6]) + short(tile_data[9]) + short(tile_data[10]);
				v_data[6] = -short(tile_data[5]) + short(tile_data[6]) - short(tile_data[9]) + short(tile_data[10]);
				v_data[7] = short(tile_data[5]) - short(tile_data[7]) + short(tile_data[9]) - short(tile_data[11]);
				v_data[8] = -short(tile_data[4]) + short(tile_data[6]) + short(tile_data[8]) - short(tile_data[10]);
				v_data[9] = -short(tile_data[5]) - short(tile_data[6]) + short(tile_data[9]) + short(tile_data[10]);
				v_data[10] = short(tile_data[5]) - short(tile_data[6]) - short(tile_data[9]) + short(tile_data[10]);
				v_data[11] = -short(tile_data[5]) + short(tile_data[7]) + short(tile_data[9]) - short(tile_data[11]);
				v_data[12] = short(tile_data[4]) - short(tile_data[6]) - short(tile_data[12]) + short(tile_data[14]);
				v_data[13] = short(tile_data[5]) + short(tile_data[6]) - short(tile_data[13]) - short(tile_data[14]);
				v_data[14] = -short(tile_data[5]) + short(tile_data[6]) + short(tile_data[13]) - short(tile_data[14]);
				v_data[15] = short(tile_data[5]) - short(tile_data[7]) - short(tile_data[13]) + short(tile_data[15]);
			}

			inline void calculate_ATmA(const int *m_data, int *result)
			{
				result[0] = m_data[0] + m_data[1] + m_data[2] + m_data[4] + m_data[5] + m_data[6] + m_data[8] + m_data[9] + m_data[10];
				result[1] = m_data[1] - m_data[2] - m_data[3] + m_data[5] - m_data[6] - m_data[7] + m_data[9] - m_data[10] - m_data[11];
				result[2] = m_data[4] + m_data[5] + m_data[6] - m_data[8] - m_data[9] - m_data[10] - m_data[12] - m_data[13] - m_data[14];
				result[3] = m_data[5] - m_data[6] - m_data[7] - m_data[9] + m_data[10] + m_data[11] - m_data[13] + m_data[14] + m_data[15];
			}

		protected:

			virtual void forward_gemm(const float* input, const float* weights, float* output, bool skip_im2col = false) = 0;

			virtual void forward_gemm(const signed char* input, const signed char* weights, int* output, bool skip_im2col = false) = 0;

			virtual void forward_bias(float* output, const float* bias) = 0;			

#ifdef USE_CUDA
		public:
			virtual void Forward(cublasHandle_t cublas_handle_, const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top) = 0;
		protected:
			virtual void forward_gemm(cublasHandle_t cublas_handle_, const float* input, const float* weights, float* output, bool skip_im2col = false) = 0; 
			virtual void forward_gemm(cublasHandle_t cublas_handle_, const signed char* input, const signed char* weights, int* output, bool skip_im2col = false) = 0;
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

			void conv_im2col_cpu(const signed char* data, signed char* col_buff)
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

			void conv_im2col_gpu(const signed char* data, signed char* col_buff)
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
			cudnnTensorDescriptor_t xdesc = nullptr;
			cudnnTensorDescriptor_t	ydesc = nullptr;
			cudnnTensorDescriptor_t bdesc = nullptr;
			cudnnFilterDescriptor_t wdesc = nullptr;
			cudnnConvolutionDescriptor_t conv_desc = nullptr;
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
