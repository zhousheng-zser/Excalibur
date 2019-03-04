#include "convolution.hpp"
//#include "cblas.h"
#include <iostream>
#include <filesystem>

namespace glasssix
{
	namespace excalibur
	{
		convolution::convolution(int input_Channel, int output_Channel, int kernelSize, int stride, int pad, bool bias_term, int device)
		{
			input_Channel_ = input_Channel;
			output_Channel_ = output_Channel;
			kernelSize_ = kernelSize;
			stride_ = stride;
			pad_ = pad;
			bias_term_ = bias_term;
			device_ = device;
			weights_.reset(new tensor<float>(std::vector<int>{input_Channel_*output_Channel_*kernelSize_*kernelSize_}, device_));
			bias_.reset(new tensor<float>(std::vector<int>{output_Channel_}, device_));
			setup_internal_params();
		}

		convolution::convolution(int input_Channel, int output_Channel, int kernelSize, int group, int stride, int pad, bool bias_term, int device)
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
			weights_.reset(new tensor<float>(std::vector<int>{input_Channel_*output_Channel_*kernelSize_*kernelSize_ / group}, device_));
			bias_.reset(new tensor<float>(std::vector<int>{output_Channel_}, device_));
			setup_internal_params(group);
		}

		convolution::~convolution()
		{
			delete U_;
			delete V_;
#ifdef USE_CUDNN
			if (cudnn_handle_)
			{
				CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
			}
			CUDNN_CHECK(cudnnDestroyTensorDescriptor(xdesc));
			CUDNN_CHECK(cudnnDestroyTensorDescriptor(ydesc));
			CUDNN_CHECK(cudnnDestroyFilterDescriptor(wdesc));
			CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
			if (bias_term_)
			{
				CUDNN_CHECK(cudnnDestroyTensorDescriptor(bdesc));
			}
			if (extra != nullptr)
			{
				cudaFree(extra);
			}
#endif
		}

		void convolution::set_bias(float* bias)
		{
			if (bias_term_)
			{
				bias_->set_cpu_data(bias);
			}
		}

		void convolution::set_weights(float* weights)
		{
			weights_->set_cpu_data(weights);
		}

		void convolution::conv_im2col_cpu(const float* data, float* col_buff, int num)
		{
			if (order_ == NCHW)
			{
				im2col_cpu(data, input_Channel_, intput_shape_[2], intput_shape_[3], kernelSize_,
					kernelSize_, pad_, pad_, stride_, stride_, 1, 1, col_buff, order_, num);
			}
			else if (order_ == NHWC)
			{
				im2col_cpu(data, input_Channel_, intput_shape_[1], intput_shape_[2], kernelSize_,
					kernelSize_, pad_, pad_, stride_, stride_, 1, 1, col_buff, order_, num);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void convolution::conv_col2im_cpu(const float* col_buff, float* data)
		{
			col2im_cpu(col_buff, input_Channel_, intput_shape_[2], intput_shape_[3], kernelSize_,
				kernelSize_, pad_, pad_, stride_, stride_, 1, 1, data);
		}

#ifdef USE_CUDA
		void convolution::conv_im2col_gpu(const float* data, float* col_buff)
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

		void convolution::conv_col2im_gpu(const float* col_buff, float* data)
		{
			col2im_gpu(col_buff, input_Channel_, intput_shape_[2], intput_shape_[3], kernelSize_,
				kernelSize_, pad_, pad_, stride_, stride_, 1, 1, data);
		}

#endif

		void convolution::setup_internal_params()
		{
			kernel_dim_ = input_Channel_*kernelSize_*kernelSize_;
			group_ = 1;
			weight_offset_ = kernelSize_ * kernelSize_;
			isfirst = true;
			tile_size_ = m_ + kernelSize_ - 1;//m+r-1
			U_num_ = output_Channel_ * input_Channel_ / group_;
			//U=G*g*GT,so U has the same number as kernel g, there are tile_size_ * tile_size_ elements in single U
			U_ = (float*)malloc(U_num_ * tile_size_ * tile_size_ * sizeof(float));
		}

		void convolution::setup_internal_params(int group)
		{
			kernel_dim_ = input_Channel_*kernelSize_*kernelSize_;
			group_ = group;
			weight_offset_ = kernelSize_*kernelSize_;
			isfirst = true;
			tile_size_ = m_ + kernelSize_ - 1;//m+r-1
			U_num_ = output_Channel_ * input_Channel_ / group_;
			//U=G*g*GT,so U has the same number as kernel g, there are tile_size_ * tile_size_ elements in single U
			U_ = (float*)malloc(U_num_ * tile_size_ * tile_size_ * sizeof(float));
		}


#ifdef USE_CUDA
		void convolution::forward_gpu_gemm(cublasHandle_t cublas_handle_, const float* input, const float* weights, float* output, bool skip_im2col)
		{
			const float* col_buff = input;
			if ((kernelSize_ != 1) || (order_ == NHWC))
			{
				conv_im2col_gpu(input, col_buffer_->mutable_gpu_data());
				col_buff = col_buffer_->gpu_data();
			}

			if (order_ == NCHW)
			{
				if (group_ == 1)
				{
					math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, output_Channel_,
						output_spatial_dim_, kernel_dim_, 1.0f, weights, col_buff, 0.0f, output);
				}
			}
			else if (order_ == NHWC)
			{
				if (group_ == 1)
				{
					math_functions::gpu_sgemm(cublas_handle_, CblasTrans, CblasTrans, output_spatial_dim_,
						output_Channel_, kernel_dim_, 1.0f, col_buff, weights, 0.0f, output);
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void convolution::forward_gpu_bias(cublasHandle_t cublas_handle_, float* output, const float* bias)
		{
			if (order_ == NCHW)
			{
				math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, output_Channel_,
					output_spatial_dim_, 1, 1.0f, bias, bias_multiplier_->gpu_data(), 1.0f, output);
			}
			else if (order_ == NHWC)
			{
				math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, output_spatial_dim_,
					output_Channel_, 1, 1.0f, bias_multiplier_->gpu_data(), bias, 1.0f, output);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

#endif
		//native
		void convolution::forward_cpu_gemm(const float* input, const float* weights, float* output, bool skip_im2col)
		{
			const float* col_buff = input;

			if ((kernelSize_ != 1) || (order_ == NHWC))
			{
				conv_im2col_cpu(input, col_buffer_->mutable_cpu_data());
				col_buff = col_buffer_->cpu_data();
			}

			if (order_ == NCHW)
			{
				if (group_ == 1)
				{
					math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_,
						output_spatial_dim_, kernel_dim_, 1.0f,
						weights, col_buff, 0.0f, output);
				}
				else
				{
					for (int g = 0; g < group_; ++g)
					{
						math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_ / group_,
							output_spatial_dim_, kernelSize_ * kernelSize_, 1.0f,
							weights + kernelSize_ * kernelSize_ * g, col_buff + output_spatial_dim_ * kernelSize_ * kernelSize_ * g, 0.0f, output + output_spatial_dim_ * g);
					}
				}
			}
			else if (order_ == NHWC)
			{
				if (group_ == 1)
				{
					math_functions::cpu_sgemm(CblasTrans, CblasTrans, output_spatial_dim_, output_Channel_,
						kernel_dim_, 1.0f,
						col_buff, weights, 0.0f, output);
				}
				else
				{
					for (int g = 0; g < group_; ++g)
					{
						math_functions::cpu_sgemm(CblasTrans, CblasTrans, output_spatial_dim_, output_Channel_ / group_,
							kernelSize_ * kernelSize_, 1.0f,
							col_buff + output_spatial_dim_ * kernelSize_ * kernelSize_ * g, weights + kernelSize_ * kernelSize_ * g, 0.0f, output + output_spatial_dim_ * g);
					}

					float* temp_data = (float*)malloc(output_spatial_dim_ * output_Channel_ * sizeof(float));
					for (int ch = 0; ch < output_Channel_; ++ch)
					{
						int channel_offset = ch * output_dim_h_ * output_dim_w_;
						for (int row = 0; row < output_dim_h_; ++row)
						{
							int row_offset = row * output_dim_w_;
							for (int col = 0; col < output_dim_w_; ++col)
							{
								temp_data[(row_offset + col) * output_Channel_ + ch] = output[channel_offset + row_offset + col];
							}
						}
					}
					memcpy(output, temp_data, output_spatial_dim_ * output_Channel_ * sizeof(float));
					delete temp_data;
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void convolution::forward_cpu_bias(float* output, const float* bias)
		{
			if (order_ == NCHW)
			{
				math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_,
					output_spatial_dim_, 1, 1.0f, bias, bias_multiplier_->cpu_data(),
					1.0f, output);
			}
			else if (order_ == NHWC)
			{
				math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_spatial_dim_,
					output_Channel_, 1, 1.0f, bias_multiplier_->cpu_data(), bias,
					1.0f, output);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void convolution::Forward_cpu_native(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			const int num = bottom->data_shape()[0];
			const float* bottom_data = bottom->cpu_data();
			const float* weights = weights_->cpu_data();
			const float* bias = bias_->cpu_data();
			order_ = bottom->order();

			//
			intput_shape_.clear();
			intput_shape_ = bottom->data_shape();


			if (order_ == NCHW)
			{
				output_dim_h_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_dim_w_ = (bottom->data_shape()[3] + 2 * pad_ - kernelSize_) / stride_ + 1;

				top.reset(new tensor<float>(std::vector<int>{num, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
				//

				float* top_data = (top)->mutable_cpu_data();
				col_buffer_.reset(new tensor<float>(std::vector<int>{kernel_dim_/**group_*/, output_dim_h_, output_dim_w_}, device_));
				bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_dim_w_*output_dim_h_}, device_));
				output_spatial_dim_ = output_dim_w_*output_dim_h_;
				col_offset_ = kernel_dim_ * output_spatial_dim_;
				output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
				math_functions::cpu_set(output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
				//
				int bottom_dim_ = bottom->data_shape()[1] * bottom->data_shape()[2] * bottom->data_shape()[3];
				int top_dim = (top)->count(1, 4);
				for (int n = 0; n < num; n++)
				{
					forward_cpu_gemm(bottom_data + n * bottom_dim_, weights, top_data + n * top_dim);
					//std::cout << "top_data_before_bias:" << std::endl;
					//for (int i = 0; i < 10; i++)
					//{
					//	std::cout << top_data[i] << " ";
					//}
					//std::cout << std::endl;
					if (bias_term_)
					{
						forward_cpu_bias(top_data + n * top_dim, bias);
					}
					//std::cout << "top_data_after_bias:" << std::endl;
					//for (int i = 0; i < 10; i++)
					//{
					//	std::cout << top_data[i] << " ";
					//}
					//std::cout << std::endl;
				}
			}
			else if (order_ == NHWC)
			{
				output_dim_h_ = (bottom->data_shape()[1] + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_dim_w_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;

				top.reset(new tensor<float>(std::vector<int>{num, output_dim_h_, output_dim_w_, output_Channel_}, device_, order_));
				//

				float* top_data = (top)->mutable_cpu_data();
				col_buffer_.reset(new tensor<float>(std::vector<int>{kernel_dim_*group_, output_dim_h_, output_dim_w_}, device_));
				bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_dim_w_*output_dim_h_}, device_));
				output_spatial_dim_ = output_dim_w_*output_dim_h_;
				output_spatial_dim_ = output_dim_w_*output_dim_h_;
				col_offset_ = kernel_dim_ * output_spatial_dim_;
				output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
				math_functions::cpu_set(output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
				//
				int bottom_dim_ = bottom->data_shape()[1] * bottom->data_shape()[2] * bottom->data_shape()[3];
				int top_dim = (top)->count(1, 4);
				for (int n = 0; n < num; n++)
				{
					forward_cpu_gemm(bottom_data + n * bottom_dim_, weights, top_data + n * top_dim);
					if (bias_term_)
					{
						forward_cpu_bias(top_data + n * top_dim, bias);
					}
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}


		//winograd
		inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
			return static_cast<unsigned>(a) < static_cast<unsigned>(b);
		}

		void convolution::Forward_cpu_winograd(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			CHECK_EQ(kernelSize_, 3);
			CHECK_EQ(stride_, 1);

			const int num = bottom->data_shape()[0];
			const float* bottom_data = bottom->cpu_data();
			const float* weights = weights_->cpu_data();
			const float* bias = bias_->cpu_data();
			order_ = bottom->order();
			int bottom_dim_ = bottom->count(1, 4);
			intput_shape_.clear();
			intput_shape_ = bottom->data_shape();
			
			float *tile_data, *temp_v, *v_data, *temp_m, *temp_ATmA, *temp_result;
			tile_data = (float*)malloc(tile_size_*tile_size_ * sizeof(float));
			temp_v = (float*)malloc(tile_size_ * tile_size_ * sizeof(float));
			v_data = (float*)malloc(tile_size_ * tile_size_ * sizeof(float));
			temp_m = (float*)malloc(tile_size_ * tile_size_ * sizeof(float));
			temp_ATmA = (float*)malloc(m_ * tile_size_ * sizeof(float));
			temp_result = (float*)malloc(m_ * m_ * sizeof(float));

			bool isGEMM = false;

			//calculate U_
#pragma omp parallel for
			for (int n = 0; n < U_num_; ++n)
			{
				if (isGEMM)
				{
					float *temp_u = (float*)malloc(tile_size_ * kernelSize_ * sizeof(float));
					math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, tile_size_,
						kernelSize_, kernelSize_, 1.0f,
						G_, weights + kernelSize_ * kernelSize_ * n, 0.0f, temp_u);

					math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, tile_size_,
						tile_size_, kernelSize_, 1.0f,
						temp_u, GT_, 0.0f, U_ + tile_size_ * tile_size_ * n);
					delete temp_u;
				}
				else
				{
					float* u_data = U_ + tile_size_ * tile_size_ * n;
					const float* weight_data = weights + kernelSize_ * kernelSize_ * n;
					
					*(u_data + 0) = *(weight_data + 0);
					*(u_data + 1) = (*(weight_data + 0) + *(weight_data + 1) + *(weight_data + 2)) / 2;
					*(u_data + 2) = (*(weight_data + 0) - *(weight_data + 1) + *(weight_data + 2)) / 2;
					*(u_data + 3) = *(weight_data + 2);
					*(u_data + 4) = (*(weight_data + 0) + *(weight_data + 3) + *(weight_data + 6)) / 2;
					*(u_data + 5) = (*(weight_data + 0) + *(weight_data + 1) + *(weight_data + 2) + 
						             *(weight_data + 3) + *(weight_data + 4) + *(weight_data + 5) +
						             *(weight_data + 6) + *(weight_data + 7) + *(weight_data + 8)) / 4;
					*(u_data + 6) = (*(weight_data + 0) + *(weight_data + 3) + *(weight_data + 6) -
						             *(weight_data + 1) - *(weight_data + 4) - *(weight_data + 7) +
						             *(weight_data + 2) + *(weight_data + 5) + *(weight_data + 8)) / 4;
					*(u_data + 7) = (*(weight_data + 2) + *(weight_data + 5) + *(weight_data + 8)) / 2;
					*(u_data + 8) = (*(weight_data + 0) - *(weight_data + 3) + *(weight_data + 6)) / 2;
					*(u_data + 9) = (*(weight_data + 0) - *(weight_data + 3) + *(weight_data + 6) +
						             *(weight_data + 1) - *(weight_data + 4) + *(weight_data + 7) +
						             *(weight_data + 2) - *(weight_data + 5) + *(weight_data + 8)) / 4;
					*(u_data + 10) = (*(weight_data + 0) - *(weight_data + 3) + *(weight_data + 6) -
						              *(weight_data + 1) + *(weight_data + 4) - *(weight_data + 7) +
						              *(weight_data + 2) - *(weight_data + 5) + *(weight_data + 8)) / 4;
					*(u_data + 11) = (*(weight_data + 2) - *(weight_data + 5) + *(weight_data + 8)) / 2;
					*(u_data + 12) = *(weight_data + 6);
					*(u_data + 13) = (*(weight_data + 6) + *(weight_data + 7) + *(weight_data + 8)) / 2;
					*(u_data + 14) = (*(weight_data + 6) - *(weight_data + 7) + *(weight_data + 8)) / 2;
					*(u_data + 15) = *(weight_data + 8);
				}
			}

			if (order_ == NCHW)
			{
				int input_dim_h = bottom->data_shape()[2];
				int input_dim_w = bottom->data_shape()[3];
				int input_spatial_dim = input_dim_h * input_dim_w;
				output_dim_h_ = (input_dim_h + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_dim_w_ = (input_dim_w + 2 * pad_ - kernelSize_) / stride_ + 1;
				top.reset(new tensor<float>(std::vector<int>{num, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
				float* top_data = top->mutable_cpu_data();
				int top_dim = (top)->count(1, 4);
				output_spatial_dim_ = output_dim_w_*output_dim_h_;
				output_spatial_dim_ = output_dim_w_*output_dim_h_;

				int h_subtract_tilesize = input_dim_h + 2 * pad_ - tile_size_;
				int w_subtract_tilesize = input_dim_w + 2 * pad_ - tile_size_;
				h_tile_num_ = int(h_subtract_tilesize / m_ + 0.5f) + 1;//h_tile_num_ = ceil((H-(m+r-1))/m) + 1, H is height after padding
				w_tile_num_ = int(w_subtract_tilesize / m_ + 0.5f) + 1;//w_tile_num_ = ceil((W-(m+r-1))/m) + 1, W is width after padding
				int h_aligned = (h_subtract_tilesize + m_ - 1) / m_ * m_;
				int w_aligned = (w_subtract_tilesize + m_ - 1) / m_ * m_;
				int add_h = h_aligned - h_subtract_tilesize;
				int add_w = w_aligned - w_subtract_tilesize;
				int tile_length = tile_size_ * tile_size_;

				if (group_ > 1)
				{
					for (int n = 0; n < num; n++)
					{
						int num_offset_bottom = n * bottom_dim_;
						int num_offset_top = n * top_dim;

						for (int och = 0; och < output_Channel_; och++)
						{
							int U_och_offset = och * tile_length;
							int channel_offset_top = och * output_spatial_dim_;
							for (int num_h = 0; num_h < h_tile_num_; num_h++)
							{
								int row_in_output_data = num_h * m_;
								int row_in_input_data = row_in_output_data - pad_;
								for (int num_w = 0; num_w < w_tile_num_; num_w++)
								{
									int col_in_output_data = num_w * m_;
									int col_in_input_data = col_in_output_data - pad_;
									memset(temp_m, 0, tile_length * sizeof(float));

									for (int row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
									{
										int row_offset = row_in_tile * tile_size_;
										int real_row = row_in_input_data + row_in_tile;
										if (!is_a_ge_zero_and_a_lt_b(real_row, input_dim_h))
										{
											for (int col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
											{
												tile_data[row_offset + col_in_tile] = 0;
											}
										}
										else
										{
											for (int col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
											{
												int real_col = col_in_input_data + col_in_tile;
												if (!is_a_ge_zero_and_a_lt_b(real_col, input_dim_w))
												{
													tile_data[row_offset + col_in_tile] = 0;
												}
												else
												{
													tile_data[row_offset + col_in_tile] = *(bottom_data + num_offset_bottom + och * input_spatial_dim + real_row * input_dim_w + real_col);
												}
											}
										}
									}

									if (isGEMM)
									{
										math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, tile_size_,
											tile_size_, tile_size_, 1.0f,
											BT_, tile_data, 0.0f, temp_v);

										math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, tile_size_,
											tile_size_, tile_size_, 1.0f,
											temp_v, B_, 0.0f, v_data);
									}
									else
									{
										*(v_data + 0) = *(tile_data + 0) - *(tile_data + 8) + *(tile_data + 10) - *(tile_data + 2);
										*(v_data + 1) = *(tile_data + 1) - *(tile_data + 9) + *(tile_data + 2) - *(tile_data + 10);
										*(v_data + 2) = *(tile_data + 9) - *(tile_data + 1) + *(tile_data + 2) - *(tile_data + 10);
										*(v_data + 3) = *(tile_data + 1) - *(tile_data + 9) + *(tile_data + 11) - *(tile_data + 3);
										*(v_data + 4) = *(tile_data + 4) + *(tile_data + 8) - *(tile_data + 6) - *(tile_data + 10);
										*(v_data + 5) = *(tile_data + 5) + *(tile_data + 9) + *(tile_data + 6) + *(tile_data + 10);
										*(v_data + 6) = *(tile_data + 6) + *(tile_data + 10) - *(tile_data + 5) - *(tile_data + 9);
										*(v_data + 7) = *(tile_data + 5) + *(tile_data + 9) - *(tile_data + 7) - *(tile_data + 11);
										*(v_data + 8) = *(tile_data + 8) - *(tile_data + 4) + *(tile_data + 6) - *(tile_data + 10);
										*(v_data + 9) = *(tile_data + 9) - *(tile_data + 5) + *(tile_data + 10) - *(tile_data + 6);
										*(v_data + 10) = *(tile_data + 5) - *(tile_data + 9) + *(tile_data + 10) - *(tile_data + 6);
										*(v_data + 11) = *(tile_data + 9) - *(tile_data + 5) + *(tile_data + 7) - *(tile_data + 11);
										*(v_data + 12) = *(tile_data + 4) - *(tile_data + 12) + *(tile_data + 14) - *(tile_data + 6);
										*(v_data + 13) = *(tile_data + 5) - *(tile_data + 13) + *(tile_data + 6) - *(tile_data + 14);
										*(v_data + 14) = *(tile_data + 13) - *(tile_data + 5) + *(tile_data + 6) - *(tile_data + 14);
										*(v_data + 15) = *(tile_data + 5) - *(tile_data + 13) + *(tile_data + 15) - *(tile_data + 7);
									}

									for (int row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
									{
										int row_offset = row_in_tile * tile_size_;
										for (int col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
										{
											int offset = row_offset + col_in_tile;
											temp_m[offset] = v_data[offset] * *(U_ + U_och_offset + offset);
										}
									}

									if (isGEMM)
									{
										math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, m_,
											tile_size_, tile_size_, 1.0f,
											AT_, temp_m, 0.0f, temp_ATmA);

										math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, m_,
											m_, tile_size_, 1.0f,
											temp_ATmA, A_, 0.0f, temp_result);
									}
									else
									{
										*(temp_result + 0) = *(temp_m + 0) + *(temp_m + 4) + *(temp_m + 8) + *(temp_m + 1) + *(temp_m + 5) + *(temp_m + 9) + *(temp_m + 2) + *(temp_m + 6) + *(temp_m + 10);
										*(temp_result + 1) = *(temp_m + 1) + *(temp_m + 5) + *(temp_m + 9) - *(temp_m + 2) - *(temp_m + 6) - *(temp_m + 10) - *(temp_m + 3) - *(temp_m + 7) - *(temp_m + 11);
										*(temp_result + 2) = *(temp_m + 4) - *(temp_m + 8) - *(temp_m + 12) + *(temp_m + 5) - *(temp_m + 9) - *(temp_m + 13) + *(temp_m + 6) - *(temp_m + 10) - *(temp_m + 14);
										*(temp_result + 3) = *(temp_m + 5) - *(temp_m + 9) - *(temp_m + 13) - *(temp_m + 6) + *(temp_m + 10) + *(temp_m + 14) - *(temp_m + 7) + *(temp_m + 11) + *(temp_m + 15);
									}

									if (num_h == h_tile_num_ - 1)
									{
										for (size_t row = 0; row < m_ - add_h; row++)
										{
											int row_offset = (row_in_output_data + row) * output_dim_w_;
											if (num_w == w_tile_num_ - 1)
											{
												for (size_t col = 0; col < m_ - add_w; col++)
												{
													*(top_data + num_offset_top + channel_offset_top + row_offset + col_in_output_data + col) = temp_result[row * m_ + col];
												}
											}
											else
											{
												for (size_t col = 0; col < m_; col++)
												{
													*(top_data + num_offset_top + channel_offset_top + row_offset + col_in_output_data + col) = temp_result[row * m_ + col];
												}
											}
										}
									}
									else
									{
										for (int row = 0; row < m_; row++)
										{
											int row_offset = (row_in_output_data + row) * output_dim_w_;
											if (num_w == w_tile_num_ - 1)
											{
												for (int col = 0; col < m_ - add_w; col++)
												{
													*(top_data + num_offset_top + channel_offset_top + row_offset + col_in_output_data + col) = temp_result[row * m_ + col];
												}
											}
											else
											{
												for (int col = 0; col < m_; col++)
												{
													*(top_data + num_offset_top + channel_offset_top + row_offset + col_in_output_data + col) = temp_result[row * m_ + col];
												}
											}
										}
									}
								}
							}
						}

						bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_spatial_dim_}, device_));
						math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_->mutable_cpu_data());

						if (bias_term_)
						{
							forward_cpu_bias(top_data + n * top_dim, bias);
						}
					}
				}
				else if (group_ == 1)
				{
					//V=BT*d*B,so V has the same number as data tile, there are tile_length elements in single V
					V_num_ = input_Channel_ * h_tile_num_ * w_tile_num_;
					V_ = (float*)malloc(V_num_ * tile_length * sizeof(float));

					for (int n = 0; n < num; n++)
					{
						int num_offset_bottom = n * bottom_dim_;
						int num_offset_top = n * top_dim;
						bool is_first = true;//only calculate V when is_first==true

						//calculate final output
						for (int och = 0; och < output_Channel_; och++)
						{
							int U_och_offset = och * input_Channel_ * tile_length;
							int channel_offset_top = och * output_spatial_dim_;
							for (int num_h = 0; num_h < h_tile_num_; num_h++)
							{
								int V_h_offset = num_h * w_tile_num_ * tile_length;
								int row_in_output_data = num_h * m_;
								int row_in_input_data = row_in_output_data - pad_;
								for (int num_w = 0; num_w < w_tile_num_; num_w++)
								{
									int V_w_offset = num_w * tile_length;
									int col_in_output_data = num_w * m_;
									int col_in_input_data = col_in_output_data - pad_;
									memset(temp_m, 0, tile_length * sizeof(float));

									for (int ich = 0; ich < input_Channel_; ich++)
									{
										int V_ich_offset = ich * h_tile_num_ * w_tile_num_ * tile_length;
										int U_ich_offset = ich * tile_length;

										//calculate V when is_first==true
										if (is_first)
										{
											int V_sequence = (ich * h_tile_num_ + num_h) * w_tile_num_ + num_w;

											for (int row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
											{
												int real_row = row_in_input_data + row_in_tile;
												if (!is_a_ge_zero_and_a_lt_b(real_row, input_dim_h))
												{
													for (int col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
													{
														tile_data[row_in_tile * tile_size_ + col_in_tile] = 0;
													}
												}
												else
												{
													for (int col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
													{
														int real_col = col_in_input_data + col_in_tile;
														if (!is_a_ge_zero_and_a_lt_b(real_col, input_dim_w))
														{
															tile_data[row_in_tile * tile_size_ + col_in_tile] = 0;
														}
														else
														{
															tile_data[row_in_tile * tile_size_ + col_in_tile] = *(bottom_data + num_offset_bottom + ich * input_spatial_dim + real_row * input_dim_w + real_col);
														}
													}
												}
											}

											if (isGEMM)
											{
												math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, tile_size_,
													tile_size_, tile_size_, 1.0f,
													BT_, tile_data, 0.0f, temp_v);

												math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, tile_size_,
													tile_size_, tile_size_, 1.0f,
													temp_v, B_, 0.0f, V_ + tile_length * V_sequence);
											}
											else
											{
												float* v_data = V_ + tile_length * V_sequence;
												*(v_data + 0) = *(tile_data + 0) - *(tile_data + 8) + *(tile_data + 10) - *(tile_data + 2);
												*(v_data + 1) = *(tile_data + 1) - *(tile_data + 9) + *(tile_data + 2) - *(tile_data + 10);
												*(v_data + 2) = *(tile_data + 9) - *(tile_data + 1) + *(tile_data + 2) - *(tile_data + 10);
												*(v_data + 3) = *(tile_data + 1) - *(tile_data + 9) + *(tile_data + 11) - *(tile_data + 3);
												*(v_data + 4) = *(tile_data + 4) + *(tile_data + 8) - *(tile_data + 6) - *(tile_data + 10);
												*(v_data + 5) = *(tile_data + 5) + *(tile_data + 9) + *(tile_data + 6) + *(tile_data + 10);
												*(v_data + 6) = *(tile_data + 6) + *(tile_data + 10) - *(tile_data + 5) - *(tile_data + 9);
												*(v_data + 7) = *(tile_data + 5) + *(tile_data + 9) - *(tile_data + 7) - *(tile_data + 11);
												*(v_data + 8) = *(tile_data + 8) - *(tile_data + 4) + *(tile_data + 6) - *(tile_data + 10);
												*(v_data + 9) = *(tile_data + 9) - *(tile_data + 5) + *(tile_data + 10) - *(tile_data + 6);
												*(v_data + 10) = *(tile_data + 5) - *(tile_data + 9) + *(tile_data + 10) - *(tile_data + 6);
												*(v_data + 11) = *(tile_data + 9) - *(tile_data + 5) + *(tile_data + 7) - *(tile_data + 11);
												*(v_data + 12) = *(tile_data + 4) - *(tile_data + 12) + *(tile_data + 14) - *(tile_data + 6);
												*(v_data + 13) = *(tile_data + 5) - *(tile_data + 13) + *(tile_data + 6) - *(tile_data + 14);
												*(v_data + 14) = *(tile_data + 13) - *(tile_data + 5) + *(tile_data + 6) - *(tile_data + 14);
												*(v_data + 15) = *(tile_data + 5) - *(tile_data + 13) + *(tile_data + 15) - *(tile_data + 7);
											}
										}

										for (int row = 0; row < tile_size_; row++)
										{
											int row_offset = row * tile_size_;
											for (int col = 0; col < tile_size_; col++)
											{
												temp_m[row_offset + col] += *(V_ + V_ich_offset + V_h_offset + V_w_offset + row_offset + col) * *(U_ + U_och_offset + U_ich_offset + row_offset + col);
											}
										}
									}

									if (isGEMM)
									{
										math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, m_,
											tile_size_, tile_size_, 1.0f,
											AT_, temp_m, 0.0f, temp_ATmA);

										math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, m_,
											m_, tile_size_, 1.0f,
											temp_ATmA, A_, 0.0f, temp_result);
									}
									else
									{
										*(temp_result + 0) = *(temp_m + 0) + *(temp_m + 4) + *(temp_m + 8) + *(temp_m + 1) + *(temp_m + 5) + *(temp_m + 9) + *(temp_m + 2) + *(temp_m + 6) + *(temp_m + 10);
										*(temp_result + 1) = *(temp_m + 1) + *(temp_m + 5) + *(temp_m + 9) - *(temp_m + 2) - *(temp_m + 6) - *(temp_m + 10) - *(temp_m + 3) - *(temp_m + 7) - *(temp_m + 11);
										*(temp_result + 2) = *(temp_m + 4) - *(temp_m + 8) - *(temp_m + 12) + *(temp_m + 5) - *(temp_m + 9) - *(temp_m + 13) + *(temp_m + 6) - *(temp_m + 10) - *(temp_m + 14);
										*(temp_result + 3) = *(temp_m + 5) - *(temp_m + 9) - *(temp_m + 13) - *(temp_m + 6) + *(temp_m + 10) + *(temp_m + 14) - *(temp_m + 7) + *(temp_m + 11) + *(temp_m + 15);
									}

									if (num_h == h_tile_num_ - 1)
									{
										for (size_t row = 0; row < m_ - add_h; row++)
										{
											int row_offset = (row_in_output_data + row) * output_dim_w_;
											if (num_w == w_tile_num_ - 1)
											{
												for (size_t col = 0; col < m_ - add_w; col++)
												{
													*(top_data + num_offset_top + channel_offset_top + row_offset + col_in_output_data + col) = temp_result[row * m_ + col];
												}
											}
											else
											{
												for (size_t col = 0; col < m_; col++)
												{
													*(top_data + num_offset_top + channel_offset_top + row_offset + col_in_output_data + col) = temp_result[row * m_ + col];
												}
											}
										}
									}
									else
									{
										for (int row = 0; row < m_; row++)
										{
											int row_offset = (row_in_output_data + row) * output_dim_w_;
											if (num_w == w_tile_num_ - 1)
											{
												for (int col = 0; col < m_ - add_w; col++)
												{
													*(top_data + num_offset_top + channel_offset_top + row_offset + col_in_output_data + col) = temp_result[row * m_ + col];
												}
											}
											else
											{
												for (int col = 0; col < m_; col++)
												{
													*(top_data + num_offset_top + channel_offset_top + row_offset + col_in_output_data + col) = temp_result[row * m_ + col];
												}
											}
										}
									}
								}
							}

							is_first = false;
						}

						bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_spatial_dim_}, device_));
						math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_->mutable_cpu_data());

						if (bias_term_)
						{
							forward_cpu_bias(top_data + n * top_dim, bias);
						}
					}
				}
				else 
				{
					LOG(FATAL) << "group wrong!!!";
				}
			}
			else if (order_ == NHWC)
			{
				int input_dim_h = bottom->data_shape()[1];
				int input_dim_w = bottom->data_shape()[2];
				int input_spatial_dim = input_dim_h * input_dim_w;
				output_dim_h_ = (input_dim_h + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_dim_w_ = (input_dim_w + 2 * pad_ - kernelSize_) / stride_ + 1;
				top.reset(new tensor<float>(std::vector<int>{num, output_dim_h_, output_dim_w_, output_Channel_}, device_, order_));
				float* top_data = top->mutable_cpu_data();
				int top_dim = (top)->count(1, 4);
				output_spatial_dim_ = output_dim_w_*output_dim_h_;
				output_spatial_dim_ = output_dim_w_*output_dim_h_;

				int h_subtract_tilesize = input_dim_h + 2 * pad_ - tile_size_;
				int w_subtract_tilesize = input_dim_w + 2 * pad_ - tile_size_;
				h_tile_num_ = int(h_subtract_tilesize / m_ + 0.5f) + 1;//h_tile_num_ = ceil((H-(m+r-1))/m) + 1, H is height after padding
				w_tile_num_ = int(w_subtract_tilesize / m_ + 0.5f) + 1;//w_tile_num_ = ceil((W-(m+r-1))/m) + 1, W is width after padding
				int h_aligned = (h_subtract_tilesize + m_ - 1) / m_ * m_;
				int w_aligned = (w_subtract_tilesize + m_ - 1) / m_ * m_;
				int add_h = h_aligned - h_subtract_tilesize;
				int add_w = w_aligned - w_subtract_tilesize;
				int tile_length = tile_size_ * tile_size_;

				if (group_ > 1)
				{
					for (int n = 0; n < num; n++)
					{
						int num_offset_bottom = n * bottom_dim_;
						int num_offset_top = n * top_dim;

						for (int och = 0; och < output_Channel_; och++)
						{
							int U_och_offset = och * tile_length;
							for (int num_h = 0; num_h < h_tile_num_; num_h++)
							{
								int row_in_output_data = num_h * m_;
								int row_in_input_data = row_in_output_data - pad_;
								for (int num_w = 0; num_w < w_tile_num_; num_w++)
								{
									int col_in_output_data = num_w * m_;
									int col_in_input_data = col_in_output_data - pad_;
									memset(temp_m, 0, tile_length * sizeof(float));

									for (int row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
									{
										int row_offset = row_in_tile * tile_size_;
										int real_row = row_in_input_data + row_in_tile;
										if (!is_a_ge_zero_and_a_lt_b(real_row, input_dim_h))
										{
											for (int col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
											{
												tile_data[row_offset + col_in_tile] = 0;
											}
										}
										else
										{
											for (int col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
											{
												int real_col = col_in_input_data + col_in_tile;
												if (!is_a_ge_zero_and_a_lt_b(real_col, input_dim_w))
												{
													tile_data[row_offset + col_in_tile] = 0;
												}
												else
												{
													//output_Channel_ and input_Channel_ has the same value, so we use output_Channel_ instead
													tile_data[row_offset + col_in_tile] = *(bottom_data + num_offset_bottom + (real_row * input_dim_w + real_col) * output_Channel_ + och);
												}
											}
										}
									}

									if (isGEMM)
									{
										math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, tile_size_,
											tile_size_, tile_size_, 1.0f,
											BT_, tile_data, 0.0f, temp_v);

										math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, tile_size_,
											tile_size_, tile_size_, 1.0f,
											temp_v, B_, 0.0f, v_data);
									}
									else
									{
										*(v_data + 0) = *(tile_data + 0) - *(tile_data + 8) + *(tile_data + 10) - *(tile_data + 2);
										*(v_data + 1) = *(tile_data + 1) - *(tile_data + 9) + *(tile_data + 2) - *(tile_data + 10);
										*(v_data + 2) = *(tile_data + 9) - *(tile_data + 1) + *(tile_data + 2) - *(tile_data + 10);
										*(v_data + 3) = *(tile_data + 1) - *(tile_data + 9) + *(tile_data + 11) - *(tile_data + 3);
										*(v_data + 4) = *(tile_data + 4) + *(tile_data + 8) - *(tile_data + 6) - *(tile_data + 10);
										*(v_data + 5) = *(tile_data + 5) + *(tile_data + 9) + *(tile_data + 6) + *(tile_data + 10);
										*(v_data + 6) = *(tile_data + 6) + *(tile_data + 10) - *(tile_data + 5) - *(tile_data + 9);
										*(v_data + 7) = *(tile_data + 5) + *(tile_data + 9) - *(tile_data + 7) - *(tile_data + 11);
										*(v_data + 8) = *(tile_data + 8) - *(tile_data + 4) + *(tile_data + 6) - *(tile_data + 10);
										*(v_data + 9) = *(tile_data + 9) - *(tile_data + 5) + *(tile_data + 10) - *(tile_data + 6);
										*(v_data + 10) = *(tile_data + 5) - *(tile_data + 9) + *(tile_data + 10) - *(tile_data + 6);
										*(v_data + 11) = *(tile_data + 9) - *(tile_data + 5) + *(tile_data + 7) - *(tile_data + 11);
										*(v_data + 12) = *(tile_data + 4) - *(tile_data + 12) + *(tile_data + 14) - *(tile_data + 6);
										*(v_data + 13) = *(tile_data + 5) - *(tile_data + 13) + *(tile_data + 6) - *(tile_data + 14);
										*(v_data + 14) = *(tile_data + 13) - *(tile_data + 5) + *(tile_data + 6) - *(tile_data + 14);
										*(v_data + 15) = *(tile_data + 5) - *(tile_data + 13) + *(tile_data + 15) - *(tile_data + 7);
									}

									for (int row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
									{
										int row_offset = row_in_tile * tile_size_;
										for (int col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
										{
											int offset = row_offset + col_in_tile;
											temp_m[offset] = v_data[offset] * *(U_ + U_och_offset + offset);
										}
									}

									if (isGEMM)
									{
										math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, m_,
											tile_size_, tile_size_, 1.0f,
											AT_, temp_m, 0.0f, temp_ATmA);

										math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, m_,
											m_, tile_size_, 1.0f,
											temp_ATmA, A_, 0.0f, temp_result);
									}
									else
									{
										*(temp_result + 0) = *(temp_m + 0) + *(temp_m + 4) + *(temp_m + 8) + *(temp_m + 1) + *(temp_m + 5) + *(temp_m + 9) + *(temp_m + 2) + *(temp_m + 6) + *(temp_m + 10);
										*(temp_result + 1) = *(temp_m + 1) + *(temp_m + 5) + *(temp_m + 9) - *(temp_m + 2) - *(temp_m + 6) - *(temp_m + 10) - *(temp_m + 3) - *(temp_m + 7) - *(temp_m + 11);
										*(temp_result + 2) = *(temp_m + 4) - *(temp_m + 8) - *(temp_m + 12) + *(temp_m + 5) - *(temp_m + 9) - *(temp_m + 13) + *(temp_m + 6) - *(temp_m + 10) - *(temp_m + 14);
										*(temp_result + 3) = *(temp_m + 5) - *(temp_m + 9) - *(temp_m + 13) - *(temp_m + 6) + *(temp_m + 10) + *(temp_m + 14) - *(temp_m + 7) + *(temp_m + 11) + *(temp_m + 15);
									}

									if (num_h == h_tile_num_ - 1)
									{
										for (size_t row = 0; row < m_ - add_h; row++)
										{
											int row_offset = (row_in_output_data + row) * output_dim_w_ * output_Channel_;
											if (num_w == w_tile_num_ - 1)
											{
												for (size_t col = 0; col < m_ - add_w; col++)
												{
													*(top_data + num_offset_top + row_offset + (col_in_output_data + col) * output_Channel_ + och) = temp_result[row * m_ + col];
												}
											}
											else
											{
												for (size_t col = 0; col < m_; col++)
												{
													*(top_data + num_offset_top + row_offset + (col_in_output_data + col) * output_Channel_ + och) = temp_result[row * m_ + col];
												}
											}
										}
									}
									else
									{
										for (int row = 0; row < m_; row++)
										{
											int row_offset = (row_in_output_data + row) * output_dim_w_ * output_Channel_;
											if (num_w == w_tile_num_ - 1)
											{
												for (int col = 0; col < m_ - add_w; col++)
												{
													*(top_data + num_offset_top + row_offset + (col_in_output_data + col) * output_Channel_ + och) = temp_result[row * m_ + col];
												}
											}
											else
											{
												for (int col = 0; col < m_; col++)
												{
													*(top_data + num_offset_top + row_offset + (col_in_output_data + col) * output_Channel_ + och) = temp_result[row * m_ + col];
												}
											}
										}
									}
								}
							}
						}

						bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_spatial_dim_}, device_));
						math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_->mutable_cpu_data());

						if (bias_term_)
						{
							forward_cpu_bias(top_data + n * top_dim, bias);
						}
					}
				}
				else if (group_ == 1)
				{
					//V=BT*d*B,so V has the same number as data tile, there are tile_length elements in single V
					V_num_ = input_Channel_ * h_tile_num_ * w_tile_num_;
					V_ = (float*)malloc(V_num_ * tile_length * sizeof(float));

					for (int n = 0; n < num; n++)
					{
						int num_offset_bottom = n * bottom_dim_;
						int num_offset_top = n * top_dim;
						bool is_first = true;//only calculate V when is_first==true

						for (int och = 0; och < output_Channel_; och++)
						{
							int U_och_offset = och * input_Channel_ * tile_length;
							for (int num_h = 0; num_h < h_tile_num_; num_h++)
							{
								int V_h_offset = num_h * w_tile_num_ * tile_length;
								int row_in_output_data = num_h * m_;
								int row_in_input_data = row_in_output_data - pad_;
								for (int num_w = 0; num_w < w_tile_num_; num_w++)
								{
									int V_w_offset = num_w * tile_length;
									int col_in_output_data = num_w * m_;
									int col_in_input_data = col_in_output_data - pad_;
									memset(temp_m, 0, tile_length * sizeof(float));

									for (int ich = 0; ich < input_Channel_; ich++)
									{
										int V_ich_offset = ich * h_tile_num_ * w_tile_num_ * tile_length;
										int U_ich_offset = ich * tile_length;

										//calculate V when is_first==true
										if (is_first)
										{
											int V_sequence = (ich * h_tile_num_ + num_h) * w_tile_num_ + num_w;

											for (int row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
											{
												int real_row = row_in_input_data + row_in_tile;
												if (!is_a_ge_zero_and_a_lt_b(real_row, input_dim_h))
												{
													for (int col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
													{
														tile_data[row_in_tile * tile_size_ + col_in_tile] = 0;
													}
												}
												else
												{
													for (int col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
													{
														int real_col = col_in_input_data + col_in_tile;
														if (!is_a_ge_zero_and_a_lt_b(real_col, input_dim_w))
														{
															tile_data[row_in_tile * tile_size_ + col_in_tile] = 0;
														}
														else
														{
															tile_data[row_in_tile * tile_size_ + col_in_tile] = *(bottom_data + num_offset_bottom + (real_row * input_dim_w + real_col) * input_Channel_ + ich);
														}
													}
												}
											}

											if (isGEMM)
											{
												math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, tile_size_,
													tile_size_, tile_size_, 1.0f,
													BT_, tile_data, 0.0f, temp_v);

												math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, tile_size_,
													tile_size_, tile_size_, 1.0f,
													temp_v, B_, 0.0f, V_ + tile_length * V_sequence);
											}
											else
											{
												float* v_data = V_ + tile_length * V_sequence;
												*(v_data + 0) = *(tile_data + 0) - *(tile_data + 8) + *(tile_data + 10) - *(tile_data + 2);
												*(v_data + 1) = *(tile_data + 1) - *(tile_data + 9) + *(tile_data + 2) - *(tile_data + 10);
												*(v_data + 2) = *(tile_data + 9) - *(tile_data + 1) + *(tile_data + 2) - *(tile_data + 10);
												*(v_data + 3) = *(tile_data + 1) - *(tile_data + 9) + *(tile_data + 11) - *(tile_data + 3);
												*(v_data + 4) = *(tile_data + 4) + *(tile_data + 8) - *(tile_data + 6) - *(tile_data + 10);
												*(v_data + 5) = *(tile_data + 5) + *(tile_data + 9) + *(tile_data + 6) + *(tile_data + 10);
												*(v_data + 6) = *(tile_data + 6) + *(tile_data + 10) - *(tile_data + 5) - *(tile_data + 9);
												*(v_data + 7) = *(tile_data + 5) + *(tile_data + 9) - *(tile_data + 7) - *(tile_data + 11);
												*(v_data + 8) = *(tile_data + 8) - *(tile_data + 4) + *(tile_data + 6) - *(tile_data + 10);
												*(v_data + 9) = *(tile_data + 9) - *(tile_data + 5) + *(tile_data + 10) - *(tile_data + 6);
												*(v_data + 10) = *(tile_data + 5) - *(tile_data + 9) + *(tile_data + 10) - *(tile_data + 6);
												*(v_data + 11) = *(tile_data + 9) - *(tile_data + 5) + *(tile_data + 7) - *(tile_data + 11);
												*(v_data + 12) = *(tile_data + 4) - *(tile_data + 12) + *(tile_data + 14) - *(tile_data + 6);
												*(v_data + 13) = *(tile_data + 5) - *(tile_data + 13) + *(tile_data + 6) - *(tile_data + 14);
												*(v_data + 14) = *(tile_data + 13) - *(tile_data + 5) + *(tile_data + 6) - *(tile_data + 14);
												*(v_data + 15) = *(tile_data + 5) - *(tile_data + 13) + *(tile_data + 15) - *(tile_data + 7);
											}
										}

										for (int row = 0; row < tile_size_; row++)
										{
											int row_offset = row * tile_size_;
											for (int col = 0; col < tile_size_; col++)
											{
												temp_m[row_offset + col] += *(V_ + V_ich_offset + V_h_offset + V_w_offset + row_offset + col) * *(U_ + U_och_offset + U_ich_offset + row_offset + col);
											}
										}
									}

									if (isGEMM)
									{
										math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, m_,
											tile_size_, tile_size_, 1.0f,
											AT_, temp_m, 0.0f, temp_ATmA);

										math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, m_,
											m_, tile_size_, 1.0f,
											temp_ATmA, A_, 0.0f, temp_result);
									}
									else
									{
										*(temp_result + 0) = *(temp_m + 0) + *(temp_m + 4) + *(temp_m + 8) + *(temp_m + 1) + *(temp_m + 5) + *(temp_m + 9) + *(temp_m + 2) + *(temp_m + 6) + *(temp_m + 10);
										*(temp_result + 1) = *(temp_m + 1) + *(temp_m + 5) + *(temp_m + 9) - *(temp_m + 2) - *(temp_m + 6) - *(temp_m + 10) - *(temp_m + 3) - *(temp_m + 7) - *(temp_m + 11);
										*(temp_result + 2) = *(temp_m + 4) - *(temp_m + 8) - *(temp_m + 12) + *(temp_m + 5) - *(temp_m + 9) - *(temp_m + 13) + *(temp_m + 6) - *(temp_m + 10) - *(temp_m + 14);
										*(temp_result + 3) = *(temp_m + 5) - *(temp_m + 9) - *(temp_m + 13) - *(temp_m + 6) + *(temp_m + 10) + *(temp_m + 14) - *(temp_m + 7) + *(temp_m + 11) + *(temp_m + 15);
									}

									if (num_h == h_tile_num_ - 1)
									{
										for (size_t row = 0; row < m_ - add_h; row++)
										{
											int row_offset = (row_in_output_data + row) * output_dim_w_ * output_Channel_;
											if (num_w == w_tile_num_ - 1)
											{
												for (size_t col = 0; col < m_ - add_w; col++)
												{
													*(top_data + num_offset_top + row_offset + (col_in_output_data + col) * output_Channel_ + och) = temp_result[row * m_ + col];
												}
											}
											else
											{
												for (size_t col = 0; col < m_; col++)
												{
													*(top_data + num_offset_top + row_offset + (col_in_output_data + col) * output_Channel_ + och) = temp_result[row * m_ + col];
												}
											}
										}
									}
									else
									{
										for (int row = 0; row < m_; row++)
										{
											int row_offset = (row_in_output_data + row) * output_dim_w_ * output_Channel_;
											if (num_w == w_tile_num_ - 1)
											{
												for (int col = 0; col < m_ - add_w; col++)
												{
													*(top_data + num_offset_top + row_offset + (col_in_output_data + col) * output_Channel_ + och) = temp_result[row * m_ + col];
												}
											}
											else
											{
												for (int col = 0; col < m_; col++)
												{
													*(top_data + num_offset_top + row_offset + (col_in_output_data + col) * output_Channel_ + och) = temp_result[row * m_ + col];
												}
											}
										}
									}
								}
							}

							is_first = false;
						}

						bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_spatial_dim_}, device_));
						math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_->mutable_cpu_data());

						if (bias_term_)
						{
							forward_cpu_bias(top_data + n * top_dim, bias);
						}
					}
				}
				else
				{
					LOG(FATAL) << "group wrong!!!";
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}


        //batch
#ifdef USE_MKL

		void convolution::forward_cpu_gemm_batch(const float* input, const float* weights, float* output, int top_dim, int num, bool skip_im2col)
		{
			const float* col_buff = input;

			if ((kernelSize_ != 1) || (order_ == NHWC))
			{
				conv_im2col_cpu(input, col_buffer_->mutable_cpu_data(), num);
				col_buff = col_buffer_->cpu_data();
			}

			if (order_ == NCHW)
			{
				for (int g = 0; g < group_; ++g)
				{
					math_functions::cpu_batch_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_ / group_,
						output_spatial_dim_, kernel_dim_ / group_, 1.0f, 
						weights + g * kernel_dim_ / group_, 0,
						col_buff + g * output_spatial_dim_ * kernel_dim_ / group_, kernel_dim_*output_dim_h_*output_dim_w_, 0.0f, 
						output + output_spatial_dim_ * g, top_dim, num);
				}
			}
			else if (order_ == NHWC)
			{
				for (int g = 0; g < group_; ++g)
				{
					math_functions::cpu_batch_sgemm(CblasTrans, CblasTrans, output_spatial_dim_,
						output_Channel_ / group_, kernel_dim_ / group_, 1.0f, 
						col_buff + g * output_spatial_dim_ * kernel_dim_ / group_, kernel_dim_*output_dim_h_*output_dim_w_,
						weights + g * kernel_dim_ / group_, 0, 0.0f, 
						output + output_spatial_dim_ * g, top_dim, num);
				}

				if (group_ > 1)
				{
					float* temp_data = (float*)malloc(output_spatial_dim_ * output_Channel_ * sizeof(float));
					for (int n = 0; n < num; n++)
					{
						int n_offset = n * output_spatial_dim_ * output_Channel_;
						for (int ch = 0; ch < output_Channel_; ++ch)
						{
							int channel_offset = ch * output_dim_h_ * output_dim_w_;
							for (int row = 0; row < output_dim_h_; ++row)
							{
								int row_offset = row * output_dim_w_;
								for (int col = 0; col < output_dim_w_; ++col)
								{
									temp_data[(row_offset + col) * output_Channel_ + ch] = output[n_offset + channel_offset + row_offset + col];
								}
							}
						}
						memcpy(output + n_offset, temp_data, output_spatial_dim_ * output_Channel_ * sizeof(float));
					}

					delete temp_data;
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void convolution::forward_cpu_bias_batch(float* output, const float* bias, int num)
		{
			if (order_ == NCHW)
			{
				math_functions::cpu_batch_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_,
					output_spatial_dim_, 1, 1.0f, bias, 0,
					bias_multiplier_->cpu_data(), output_spatial_dim_, 1.0f, output, output_spatial_dim_ * output_Channel_, num);
			}
			else if (order_ == NHWC)
			{
				math_functions::cpu_batch_sgemm(CblasNoTrans, CblasNoTrans, output_spatial_dim_,
					output_Channel_, 1, 1.0f, bias_multiplier_->cpu_data(), output_spatial_dim_, bias, 0,
					 1.0f, output, output_spatial_dim_ * output_Channel_, num);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void convolution::Forward_cpu_batch(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			const int num = bottom->data_shape()[0];
			const float* bottom_data = bottom->cpu_data();
			const float* weights = weights_->cpu_data();
			const float* bias = bias_->cpu_data();
			order_ = bottom->order();
			intput_shape_.clear();
			intput_shape_ = bottom->data_shape();

			if (order_ == NCHW)
			{
				output_dim_h_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_dim_w_ = (bottom->data_shape()[3] + 2 * pad_ - kernelSize_) / stride_ + 1;

				top.reset(new tensor<float>(std::vector<int>{num, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
				//

				float* top_data = (top)->mutable_cpu_data();
				col_buffer_.reset(new tensor<float>(std::vector<int>{num*kernel_dim_, output_dim_h_, output_dim_w_}, device_));
				bias_multiplier_.reset(new tensor<float>(std::vector<int>{num*output_dim_w_*output_dim_h_}, device_));
				output_spatial_dim_ = output_dim_w_*output_dim_h_;
				output_spatial_dim_ = output_dim_w_*output_dim_h_;
				col_offset_ = kernel_dim_ * output_spatial_dim_;
				output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
				math_functions::cpu_set(num*output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
				//
				int bottom_dim_ = bottom->data_shape()[1] * bottom->data_shape()[2] * bottom->data_shape()[3];
				int top_dim = (top)->count(1, 4);

				forward_cpu_gemm_batch(bottom_data, weights, top_data, top_dim, num);
				if (bias_term_)
				{
					forward_cpu_bias_batch(top_data, bias, num);
				}

			}
			else if (order_ == NHWC)
			{
				output_dim_h_ = (bottom->data_shape()[1] + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_dim_w_ = (bottom->data_shape()[2] + 2 * pad_ - kernelSize_) / stride_ + 1;

				top.reset(new tensor<float>(std::vector<int>{num, output_dim_h_, output_dim_w_, output_Channel_}, device_, order_));
				//

				float* top_data = (top)->mutable_cpu_data();
				col_buffer_.reset(new tensor<float>(std::vector<int>{num*kernel_dim_, output_dim_h_, output_dim_w_}, device_));
				bias_multiplier_.reset(new tensor<float>(std::vector<int>{num*output_dim_w_*output_dim_h_}, device_));
				output_spatial_dim_ = output_dim_w_*output_dim_h_;
				output_spatial_dim_ = output_dim_w_*output_dim_h_;
				col_offset_ = kernel_dim_ * output_spatial_dim_;
				output_offset_ = output_Channel_ * output_spatial_dim_ / group_;
				math_functions::cpu_set(num*output_dim_w_*output_dim_h_, 1.0f, bias_multiplier_->mutable_cpu_data());
				//
				int bottom_dim_ = bottom->data_shape()[1] * bottom->data_shape()[2] * bottom->data_shape()[3];
				int top_dim = (top)->count(1, 4);

				forward_cpu_gemm_batch(bottom_data, weights, top_data, top_dim, num);

				if (bias_term_)
				{
					forward_cpu_bias_batch(top_data, bias, num);
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

#endif // !USE_MKL


		//entry
		void convolution::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
//			if (bottom->num() > 1)
//			{
//#ifdef USE_MKL
//				Forward_cpu_batch(bottom, top);
//#endif // !USE_MKL
//			}


			if (kernelSize_ == 3 && stride_ == 1)
			{
				Forward_cpu_winograd(bottom, top);
			}
			else
			{
				Forward_cpu_native(bottom, top);
			}
		}
	}
}