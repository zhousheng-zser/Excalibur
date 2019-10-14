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

				//1*1s1				
				if ((stride_ == 1) && (kernelSize_ == 1))
				{
					use_sgemm1x1 = true;
				}

				//winograd
				if ((stride_ == 1) && (kernelSize_ == 3))
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
							calculate_GgGT23(weights_data + kernel_length_ * n, U_data + tile_length_ * n);//calculate U
						}
					}

					if (use_sgemm1x1)
					{
						conv1x1s1_transform_kernel();
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

					if (useWinograd23)
					{
						//calculate U_
#ifdef _OPENMP
#pragma omp parallel for
#endif
						for (int n = 0; n < U_num_; ++n)
						{
							calculate_GgGT23(weights_int8_data + kernel_length_ * n, U_int16_data + tile_length_ * n);//calculate U
						}
					}
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

			//1*1s1
			bool use_sgemm1x1 = false;
			inline void conv1x1s1_transform_kernel()
			{
				int inch = input_Channel_;
				int outch = output_Channel_;
				std::shared_ptr<tensor<float>> weight_temp;
				weight_temp.reset(new tensor<float>(std::vector<int>{1, outch / 4 + outch % 4, inch / 4 + inch % 4, 4 * 4}, -1, NCHW));

				int p = 0;
				for (; p + 3 < outch; p += 4)
				{
					const float* kernel0 = weights_data + (p + 0)*inch;
					const float* kernel1 = weights_data + (p + 1)*inch;
					const float* kernel2 = weights_data + (p + 2)*inch;
					const float* kernel3 = weights_data + (p + 3)*inch;

					float* ktmp = weight_temp->mutable_cpu_data() + (p / 4) * weight_temp->width() * weight_temp->height();

					for (int q = 0; q < inch; q++)
					{
						// kernel0...3 0
						ktmp[0] = kernel0[0];
						ktmp[1] = kernel1[0];
						ktmp[2] = kernel2[0];
						ktmp[3] = kernel3[0];

						ktmp += 4;
						kernel0 += 1;
						kernel1 += 1;
						kernel2 += 1;
						kernel3 += 1;
					}
				}
				for (; p < outch; p++)
				{
					const float* kernel0 = weights_data + p * inch;
					float* ktmp = weight_temp->mutable_cpu_data() + (p / 4 + p % 4) * weight_temp->width() * weight_temp->height();

					for (int q = 0; q < inch; q++)
					{
						ktmp[0] = kernel0[0];
						ktmp++;
						kernel0++;
					}
				}

				weights_ = std::make_shared<tensor<float>>(weight_temp->clone());
				weights_data = weights_->cpu_data();
			}

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
			inline void calculate_GgGT23(const float *weight_data, float *u_data)
			{
#if SIMD_TYPE >= SIMDTYPE_SSE
				__m128 _d0, _d1, _d2;
				__m128 _w0, _w1, _w2, _w3;
				__m128 half = _mm_set1_ps(0.5);

				// load
				_d0 = _mm_loadu_ps(weight_data);
				_d1 = _mm_loadu_ps(weight_data + 3);
				_d2 = _mm_loadu_ps(weight_data + 6);

				// w = G * d
				_w0 = _d0;
				_w1 = _mm_add_ps(_mm_add_ps(_d0, _d1), _d2);
				_w1 = _mm_mul_ps(_w1, half);
				_w2 = _mm_add_ps(_mm_sub_ps(_d0, _d1), _d2);
				_w2 = _mm_mul_ps(_w2, half);
				_w3 = _d2;

				_MM_TRANSPOSE4_PS(_w0, _w1, _w2, _w3);

				__m128 _res0 = _w0;
				__m128 _res1 = _mm_add_ps(_mm_add_ps(_w0, _w1), _w2);
				_res1 = _mm_mul_ps(_res1, half);
				__m128 _res2 = _mm_add_ps(_mm_sub_ps(_w0, _w1), _w2);
				_res2 = _mm_mul_ps(_res2, half);
				__m128 _res3 = _w2;

				u_data[0] = _res0.m128_f32[0];
				u_data[1] = _res1.m128_f32[0];
				u_data[2] = _res2.m128_f32[0];
				u_data[3] = _res3.m128_f32[0];

				u_data[4] = _res0.m128_f32[1];
				u_data[5] = _res1.m128_f32[1];
				u_data[6] = _res2.m128_f32[1];
				u_data[7] = _res3.m128_f32[1];

				u_data[8] = _res0.m128_f32[2];
				u_data[9] = _res1.m128_f32[2];
				u_data[10] = _res2.m128_f32[2];
				u_data[11] = _res3.m128_f32[2];

				u_data[12] = _res0.m128_f32[3];
				u_data[13] = _res1.m128_f32[3];
				u_data[14] = _res2.m128_f32[3];
				u_data[15] = _res3.m128_f32[3];
#else
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
#endif
			}

			inline void calculate_GgGT43(const float *weight_data, float *u_data)
			{
#if SIMD_TYPE >= SIMDTYPE_AVX
				__m128 _d0, _d1, _d2;
				__m128 _w0, _w1, _w2, _w3, _w4, _w5;
				__m128 _1d2 = _mm_set1_ps(1.0f / 2);
				__m128 _1d4 = _mm_set1_ps(1.0f / 4);
				__m128 _1d6 = _mm_set1_ps(1.0f / 6);
				__m128 _n1 = _mm_set1_ps(-1);

				// load
				_d0 = _mm_loadu_ps(weight_data);
				_d1 = _mm_loadu_ps(weight_data + 3);
				_d2 = _mm_loadu_ps(weight_data + 6);

				// w = B_t * d
				_w0 = _mm_mul_ps(_d0, _1d4);

				__m128 temp_0_add_2 = _mm_add_ps(_d0, _d2);
				_w1 = _mm_add_ps(temp_0_add_2, _d1);
				_w1 = _mm_mul_ps(_w1, _1d6);
				_w1 = _mm_mul_ps(_w1, _n1);

				_w2 = _mm_sub_ps(_d1, temp_0_add_2);
				_w2 = _mm_mul_ps(_w2, _1d6);

				temp_0_add_2 = _mm_add_ps(_d2, _mm_mul_ps(_d0, _1d4));
				_w3 = _mm_add_ps(temp_0_add_2, _mm_mul_ps(_d1, _1d2));
				_w3 = _mm_mul_ps(_w3, _1d6);

				_w4 = _mm_sub_ps(temp_0_add_2, _mm_mul_ps(_d1, _1d2));
				_w4 = _mm_mul_ps(_w4, _1d6);

				_w5 = _d2;

				__m256 _m0 = _mm256_set_ps(0, 0, _w5.m128_f32[0], _w4.m128_f32[0], _w3.m128_f32[0], _w2.m128_f32[0], _w1.m128_f32[0], _w0.m128_f32[0]);
				__m256 _m1 = _mm256_set_ps(0, 0, _w5.m128_f32[1], _w4.m128_f32[1], _w3.m128_f32[1], _w2.m128_f32[1], _w1.m128_f32[1], _w0.m128_f32[1]);
				__m256 _m2 = _mm256_set_ps(0, 0, _w5.m128_f32[2], _w4.m128_f32[2], _w3.m128_f32[2], _w2.m128_f32[2], _w1.m128_f32[2], _w0.m128_f32[2]);

				__m256 _256_1d2 = _mm256_set1_ps(1.0f / 2);
				__m256 _256_1d4 = _mm256_set1_ps(1.0f / 4);
				__m256 _256_1d6 = _mm256_set1_ps(1.0f / 6);
				__m256 _256_n1 = _mm256_set1_ps(-1);

				__m256 _res0 = _mm256_mul_ps(_m0, _256_1d4);

				__m256 temp_256_0_add_2 = _mm256_add_ps(_m0, _m2);
				__m256 _res1 = _mm256_add_ps(_m1, temp_256_0_add_2);
				_res1 = _mm256_mul_ps(_res1, _256_1d6);
				_res1 = _mm256_mul_ps(_res1, _256_n1);

				__m256 _res2 = _mm256_sub_ps(_m1, temp_256_0_add_2);
				_res2 = _mm256_mul_ps(_res2, _256_1d6);

				temp_256_0_add_2 = _mm256_add_ps(_mm256_mul_ps(_m0, _256_1d4), _m2);
				__m256 _res3 = _mm256_add_ps(temp_256_0_add_2, _mm256_mul_ps(_m1, _256_1d2));
				_res3 = _mm256_mul_ps(_res3, _256_1d6);

				__m256 _res4 = _mm256_sub_ps(temp_256_0_add_2, _mm256_mul_ps(_m1, _256_1d2));
				_res4 = _mm256_mul_ps(_res4, _256_1d6);

				__m256 _res5 = _m2;

				// save to u_data
				u_data[0] = _res0.m256_f32[0];
				u_data[1] = _res1.m256_f32[0];
				u_data[2] = _res2.m256_f32[0];
				u_data[3] = _res3.m256_f32[0];
				u_data[4] = _res4.m256_f32[0];
				u_data[5] = _res5.m256_f32[0];

				u_data[6] = _res0.m256_f32[1];
				u_data[7] = _res1.m256_f32[1];
				u_data[8] = _res2.m256_f32[1];
				u_data[9] = _res3.m256_f32[1];
				u_data[10] = _res4.m256_f32[1];
				u_data[11] = _res5.m256_f32[1];

				u_data[12] = _res0.m256_f32[2];
				u_data[13] = _res1.m256_f32[2];
				u_data[14] = _res2.m256_f32[2];
				u_data[15] = _res3.m256_f32[2];
				u_data[16] = _res4.m256_f32[2];
				u_data[17] = _res5.m256_f32[2];

				u_data[18] = _res0.m256_f32[3];
				u_data[19] = _res1.m256_f32[3];
				u_data[20] = _res2.m256_f32[3];
				u_data[21] = _res3.m256_f32[3];
				u_data[22] = _res4.m256_f32[3];
				u_data[23] = _res5.m256_f32[3];

				u_data[24] = _res0.m256_f32[4];
				u_data[25] = _res1.m256_f32[4];
				u_data[26] = _res2.m256_f32[4];
				u_data[27] = _res3.m256_f32[4];
				u_data[28] = _res4.m256_f32[4];
				u_data[29] = _res5.m256_f32[4];

				u_data[30] = _res0.m256_f32[5];
				u_data[31] = _res1.m256_f32[5];
				u_data[32] = _res2.m256_f32[5];
				u_data[33] = _res3.m256_f32[5];
				u_data[34] = _res4.m256_f32[5];
				u_data[35] = _res5.m256_f32[5];

#else
				//// save to u_data
				//u_data[0] = 0.25 * _w0.m128_f32[0];
				//u_data[1] = -1.0f / 6 * (_w0.m128_f32[0] + _w0.m128_f32[1] + _w0.m128_f32[2]);
				//u_data[2] = -1.0f / 6 * (_w0.m128_f32[0] - _w0.m128_f32[1] + _w0.m128_f32[2]);
				//u_data[3] = 1.0f / 24 * _w0.m128_f32[0] + 1.0f / 12 * _w0.m128_f32[1] + 1.0f / 6 * _w0.m128_f32[2];
				//u_data[4] = 1.0f / 24 * _w0.m128_f32[0] - 1.0f / 12 * _w0.m128_f32[1] + 1.0f / 6 * _w0.m128_f32[2];
				//u_data[5] = _w0.m128_f32[2];

				//u_data[6] = 0.25 * _w1.m128_f32[0];
				//u_data[7] = -1.0f / 6 * (_w1.m128_f32[0] + _w1.m128_f32[1] + _w1.m128_f32[2]);
				//u_data[8] = -1.0f / 6 * (_w1.m128_f32[0] - _w1.m128_f32[1] + _w1.m128_f32[2]);
				//u_data[9] = 1.0f / 24 * _w1.m128_f32[0] + 1.0f / 12 * _w1.m128_f32[1] + 1.0f / 6 * _w1.m128_f32[2];
				//u_data[10] = 1.0f / 24 * _w1.m128_f32[0] - 1.0f / 12 * _w1.m128_f32[1] + 1.0f / 6 * _w1.m128_f32[2];
				//u_data[11] = _w1.m128_f32[2];

				//u_data[12] = 0.25 * _w2.m128_f32[0];
				//u_data[13] = -1.0f / 6 * (_w2.m128_f32[0] + _w2.m128_f32[1] + _w2.m128_f32[2]);
				//u_data[14] = -1.0f / 6 * (_w2.m128_f32[0] - _w2.m128_f32[1] + _w2.m128_f32[2]);
				//u_data[15] = 1.0f / 24 * _w2.m128_f32[0] + 1.0f / 12 * _w2.m128_f32[1] + 1.0f / 6 * _w2.m128_f32[2];
				//u_data[16] = 1.0f / 24 * _w2.m128_f32[0] - 1.0f / 12 * _w2.m128_f32[1] + 1.0f / 6 * _w2.m128_f32[2];
				//u_data[17] = _w2.m128_f32[2];

				//u_data[18] = 0.25 * _w3.m128_f32[0];
				//u_data[19] = -1.0f / 6 * (_w3.m128_f32[0] + _w3.m128_f32[1] + _w3.m128_f32[2]);
				//u_data[20] = -1.0f / 6 * (_w3.m128_f32[0] - _w3.m128_f32[1] + _w3.m128_f32[2]);
				//u_data[21] = 1.0f / 24 * _w3.m128_f32[0] + 1.0f / 12 * _w3.m128_f32[1] + 1.0f / 6 * _w3.m128_f32[2];
				//u_data[22] = 1.0f / 24 * _w3.m128_f32[0] - 1.0f / 12 * _w3.m128_f32[1] + 1.0f / 6 * _w3.m128_f32[2];
				//u_data[23] = _w3.m128_f32[2];

				//u_data[24] = 0.25 * _w4.m128_f32[0];
				//u_data[25] = -1.0f / 6 * (_w4.m128_f32[0] + _w4.m128_f32[1] + _w4.m128_f32[2]);
				//u_data[26] = -1.0f / 6 * (_w4.m128_f32[0] - _w4.m128_f32[1] + _w4.m128_f32[2]);
				//u_data[27] = 1.0f / 24 * _w4.m128_f32[0] + 1.0f / 12 * _w4.m128_f32[1] + 1.0f / 6 * _w4.m128_f32[2];
				//u_data[28] = 1.0f / 24 * _w4.m128_f32[0] - 1.0f / 12 * _w4.m128_f32[1] + 1.0f / 6 * _w4.m128_f32[2];
				//u_data[29] = _w4.m128_f32[2];

				//u_data[30] = 0.25 * _w5.m128_f32[0];
				//u_data[31] = -1.0f / 6 * (_w5.m128_f32[0] + _w5.m128_f32[1] + _w5.m128_f32[2]);
				//u_data[32] = -1.0f / 6 * (_w5.m128_f32[0] - _w5.m128_f32[1] + _w5.m128_f32[2]);
				//u_data[33] = 1.0f / 24 * _w5.m128_f32[0] + 1.0f / 12 * _w5.m128_f32[1] + 1.0f / 6 * _w5.m128_f32[2];
				//u_data[34] = 1.0f / 24 * _w5.m128_f32[0] - 1.0f / 12 * _w5.m128_f32[1] + 1.0f / 6 * _w5.m128_f32[2];
				//u_data[35] = _w5.m128_f32[2];
#endif
			}

			inline void calculate_BTdB23(const float *row_data1, const float *row_data2, const float *row_data3, const float *row_data4, float *v_data)
			{
#if SIMD_TYPE >= SIMDTYPE_SSE
				__m128 _d0, _d1, _d2, _d3;
				__m128 _w0, _w1, _w2, _w3;

				// load
				_d0 = _mm_loadu_ps(row_data1);
				_d1 = _mm_loadu_ps(row_data2);
				_d2 = _mm_loadu_ps(row_data3);
				_d3 = _mm_loadu_ps(row_data4);

				// w = B_t * d
				_w0 = _mm_sub_ps(_d0, _d2);
				_w1 = _mm_add_ps(_d1, _d2);
				_w2 = _mm_sub_ps(_d2, _d1);
				_w3 = _mm_sub_ps(_d1, _d3);

				_MM_TRANSPOSE4_PS(_w0, _w1, _w2, _w3);

				__m128 _res0 = _mm_sub_ps(_w0, _w2);
				__m128 _res1 = _mm_add_ps(_w1, _w2);
				__m128 _res2 = _mm_sub_ps(_w2, _w1);
				__m128 _res3 = _mm_sub_ps(_w1, _w3);

				// save to V_data
				v_data[0] = _res0.m128_f32[0];
				v_data[1] = _res1.m128_f32[0];
				v_data[2] = _res2.m128_f32[0];
				v_data[3] = _res3.m128_f32[0];
				v_data[4] = _res0.m128_f32[1];
				v_data[5] = _res1.m128_f32[1];
				v_data[6] = _res2.m128_f32[1];
				v_data[7] = _res3.m128_f32[1];
				v_data[8] = _res0.m128_f32[2];
				v_data[9] = _res1.m128_f32[2];
				v_data[10] = _res2.m128_f32[2];
				v_data[11] = _res3.m128_f32[2];
				v_data[12] = _res0.m128_f32[3];
				v_data[13] = _res1.m128_f32[3];
				v_data[14] = _res2.m128_f32[3];
				v_data[15] = _res3.m128_f32[3];
#else
				v_data[0] = row_data1[0] - row_data1[2] - row_data3[0] + row_data3[2];
				v_data[1] = row_data1[1] + row_data1[2] - row_data3[1] - row_data3[2];
				v_data[2] = -row_data1[1] + row_data1[2] + row_data3[1] - row_data3[2];
				v_data[3] = row_data1[1] - row_data1[3] - row_data3[1] + row_data3[3];
				v_data[4] = row_data2[0] - row_data2[2] + row_data3[0] - row_data3[2];
				v_data[5] = row_data2[1] + row_data2[2] + row_data3[1] + row_data3[2];
				v_data[6] = -row_data2[1] + row_data2[2] - row_data3[1] + row_data3[2];
				v_data[7] = row_data2[1] - row_data2[3] + row_data3[1] - row_data3[3];
				v_data[8] = -row_data2[0] + row_data2[2] + row_data3[0] - row_data3[2];
				v_data[9] = -row_data2[1] - row_data2[2] + row_data3[1] + row_data3[2];
				v_data[10] = row_data2[1] - row_data2[2] - row_data3[1] + row_data3[2];
				v_data[11] = -row_data2[1] + row_data2[3] + row_data3[1] - row_data3[3];
				v_data[12] = row_data2[0] - row_data2[2] - row_data4[0] + row_data4[2];
				v_data[13] = row_data2[1] + row_data2[2] - row_data4[1] - row_data4[2];
				v_data[14] = -row_data2[1] + row_data2[2] + row_data4[1] - row_data4[2];
				v_data[15] = row_data2[1] - row_data2[3] - row_data4[1] + row_data4[3];
#endif
			}

			inline void calculate_BTdB43(const float *row_data1, const float *row_data2, const float *row_data3, const float *row_data4, const float *row_data5, const float *row_data6, float *v_data)
			{
#if SIMD_TYPE >= SIMDTYPE_AVX
				__m256 _d0, _d1, _d2, _d3, _d4, _d5;
				__m256 _w0, _w1, _w2, _w3, _w4, _w5;

				__m256 _x2 = _mm256_set1_ps(2);
				__m256 _x4 = _mm256_set1_ps(4);
				__m256 _x5 = _mm256_set1_ps(5);

				// load
				_d0 = _mm256_loadu_ps(row_data1);
				_d1 = _mm256_loadu_ps(row_data2);
				_d2 = _mm256_loadu_ps(row_data3);
				_d3 = _mm256_loadu_ps(row_data4);
				_d4 = _mm256_loadu_ps(row_data5);
				_d5 = _mm256_loadu_ps(row_data6);

				// w = B_t * d
				_w0 = _mm256_mul_ps(_d0, _x4);
				_w0 = _mm256_sub_ps(_w0, _mm256_mul_ps(_d2, _x5));
				_w0 = _mm256_add_ps(_w0, _d4);
				
				__m256 temp_3_sub_1 = _mm256_sub_ps(_d3, _mm256_mul_ps(_d1, _x4));
				__m256 temp_4_sub_2 = _mm256_sub_ps(_d4, _mm256_mul_ps(_d2, _x4));
				_w1 = _mm256_add_ps(temp_3_sub_1, temp_4_sub_2);
				_w2 = _mm256_sub_ps(temp_4_sub_2, temp_3_sub_1);

				temp_3_sub_1 = _mm256_sub_ps(_d3, _d1);
				temp_3_sub_1 = _mm256_mul_ps(temp_3_sub_1, _x2);
				temp_4_sub_2 = _mm256_sub_ps(_d4, _d2);
				_w3 = _mm256_add_ps(temp_3_sub_1, temp_4_sub_2);
				_w4 = _mm256_sub_ps(temp_4_sub_2, temp_3_sub_1);

				_w5 = _mm256_mul_ps(_d1, _x4);
				_w5 = _mm256_sub_ps(_w5, _mm256_mul_ps(_d3, _x5));
				_w5 = _mm256_add_ps(_w5, _d5);

				// m = B_t * d
				__m256 _m0 = _mm256_set_ps(0, 0, _w5.m256_f32[0], _w4.m256_f32[0], _w3.m256_f32[0], _w2.m256_f32[0], _w1.m256_f32[0], _w0.m256_f32[0]);
				__m256 _m1 = _mm256_set_ps(0, 0, _w5.m256_f32[1], _w4.m256_f32[1], _w3.m256_f32[1], _w2.m256_f32[1], _w1.m256_f32[1], _w0.m256_f32[1]);
				__m256 _m2 = _mm256_set_ps(0, 0, _w5.m256_f32[2], _w4.m256_f32[2], _w3.m256_f32[2], _w2.m256_f32[2], _w1.m256_f32[2], _w0.m256_f32[2]);
				__m256 _m3 = _mm256_set_ps(0, 0, _w5.m256_f32[3], _w4.m256_f32[3], _w3.m256_f32[3], _w2.m256_f32[3], _w1.m256_f32[3], _w0.m256_f32[3]);
				__m256 _m4 = _mm256_set_ps(0, 0, _w5.m256_f32[4], _w4.m256_f32[4], _w3.m256_f32[4], _w2.m256_f32[4], _w1.m256_f32[4], _w0.m256_f32[4]);
				__m256 _m5 = _mm256_set_ps(0, 0, _w5.m256_f32[5], _w4.m256_f32[5], _w3.m256_f32[5], _w2.m256_f32[5], _w1.m256_f32[5], _w0.m256_f32[5]);

				__m256 _res0 = _mm256_mul_ps(_m0, _x4);
				_res0 = _mm256_sub_ps(_res0, _mm256_mul_ps(_m2, _x5));
				_res0 = _mm256_add_ps(_res0, _m4);

				temp_3_sub_1 = _mm256_sub_ps(_m3, _mm256_mul_ps(_m1, _x4));
				temp_4_sub_2 = _mm256_sub_ps(_m4, _mm256_mul_ps(_m2, _x4));
				__m256 _res1 = _mm256_add_ps(temp_3_sub_1, temp_4_sub_2);
				__m256 _res2 = _mm256_sub_ps(temp_4_sub_2, temp_3_sub_1);

				temp_3_sub_1 = _mm256_sub_ps(_m3, _m1);
				temp_3_sub_1 = _mm256_mul_ps(temp_3_sub_1, _x2);
				temp_4_sub_2 = _mm256_sub_ps(_m4, _m2);
				__m256 _res3 = _mm256_add_ps(temp_3_sub_1, temp_4_sub_2);
				__m256 _res4 = _mm256_sub_ps(temp_4_sub_2, temp_3_sub_1);

				__m256 _res5 = _mm256_mul_ps(_m1, _x4);
				_res5 = _mm256_sub_ps(_res5, _mm256_mul_ps(_m3, _x5));
				_res5 = _mm256_add_ps(_res5, _m5);

				// save to V_data
				v_data[0] = _res0.m256_f32[0];
				v_data[1] = _res1.m256_f32[0];
				v_data[2] = _res2.m256_f32[0];
				v_data[3] = _res3.m256_f32[0];
				v_data[4] = _res4.m256_f32[0];
				v_data[5] = _res5.m256_f32[0];

				v_data[6] = _res0.m256_f32[1];
				v_data[7] = _res1.m256_f32[1];
				v_data[8] = _res2.m256_f32[1];
				v_data[9] = _res3.m256_f32[1];
				v_data[10] = _res4.m256_f32[1];
				v_data[11] = _res5.m256_f32[1];
				
				v_data[12] = _res0.m256_f32[2];
				v_data[13] = _res1.m256_f32[2];
				v_data[14] = _res2.m256_f32[2];
				v_data[15] = _res3.m256_f32[2];
				v_data[16] = _res4.m256_f32[2];
				v_data[17] = _res5.m256_f32[2];

				v_data[18] = _res0.m256_f32[3];
				v_data[19] = _res1.m256_f32[3];
				v_data[20] = _res2.m256_f32[3];
				v_data[21] = _res3.m256_f32[3];
				v_data[22] = _res4.m256_f32[3];
				v_data[23] = _res5.m256_f32[3];

				v_data[24] = _res0.m256_f32[4];
				v_data[25] = _res1.m256_f32[4];
				v_data[26] = _res2.m256_f32[4];
				v_data[27] = _res3.m256_f32[4];
				v_data[28] = _res4.m256_f32[4];
				v_data[29] = _res5.m256_f32[4];

				v_data[30] = _res0.m256_f32[5];
				v_data[31] = _res1.m256_f32[5];
				v_data[32] = _res2.m256_f32[5];
				v_data[33] = _res3.m256_f32[5];
				v_data[34] = _res4.m256_f32[5];
				v_data[35] = _res5.m256_f32[5];

#else
				//// save to V_data
				//v_data[0] = 4 * _w0.m256_f32[0] - 5 * _w0.m256_f32[2] + _w0.m256_f32[4];
				//v_data[1] = -4 * _w0.m256_f32[1] - 4 * _w0.m256_f32[2] + _w0.m256_f32[3] + _w0.m256_f32[4];
				//v_data[2] = 4 * _w0.m256_f32[1] - 4 * _w0.m256_f32[2] - _w0.m256_f32[3] + _w0.m256_f32[4];
				//v_data[3] = -2 * _w0.m256_f32[1] - _w0.m256_f32[2] + 2 * _w0.m256_f32[3] + _w0.m256_f32[4];
				//v_data[4] = 2 * _w0.m256_f32[1] - _w0.m256_f32[2] - 2 * _w0.m256_f32[3] + _w0.m256_f32[4];
				//v_data[5] = 4 * _w0.m256_f32[1] - 5 * _w0.m256_f32[3] + _w0.m256_f32[5];

				//v_data[6] = 4 * _w1.m256_f32[0] - 5 * _w1.m256_f32[2] + _w1.m256_f32[4];
				//v_data[7] = -4 * _w1.m256_f32[1] - 4 * _w1.m256_f32[2] + _w1.m256_f32[3] + _w1.m256_f32[4];
				//v_data[8] = 4 * _w1.m256_f32[1] - 4 * _w1.m256_f32[2] - _w1.m256_f32[3] + _w1.m256_f32[4];
				//v_data[9] = -2 * _w1.m256_f32[1] - _w1.m256_f32[2] + 2 * _w1.m256_f32[3] + _w1.m256_f32[4];
				//v_data[10] = 2 * _w1.m256_f32[1] - _w1.m256_f32[2] - 2 * _w1.m256_f32[3] + _w1.m256_f32[4];
				//v_data[11] = 4 * _w1.m256_f32[1] - 5 * _w1.m256_f32[3] + _w1.m256_f32[5];

				//v_data[12] = 4 * _w2.m256_f32[0] - 5 * _w2.m256_f32[2] + _w2.m256_f32[4];
				//v_data[13] = -4 * _w2.m256_f32[1] - 4 * _w2.m256_f32[2] + _w2.m256_f32[3] + _w2.m256_f32[4];
				//v_data[14] = 4 * _w2.m256_f32[1] - 4 * _w2.m256_f32[2] - _w2.m256_f32[3] + _w2.m256_f32[4];
				//v_data[15] = -2 * _w2.m256_f32[1] - _w2.m256_f32[2] + 2 * _w2.m256_f32[3] + _w2.m256_f32[4];
				//v_data[16] = 2 * _w2.m256_f32[1] - _w2.m256_f32[2] - 2 * _w2.m256_f32[3] + _w2.m256_f32[4];
				//v_data[17] = 4 * _w2.m256_f32[1] - 5 * _w2.m256_f32[3] + _w2.m256_f32[5];

				//v_data[18] = 4 * _w3.m256_f32[0] - 5 * _w3.m256_f32[2] + _w3.m256_f32[4];
				//v_data[19] = -4 * _w3.m256_f32[1] - 4 * _w3.m256_f32[2] + _w3.m256_f32[3] + _w3.m256_f32[4];
				//v_data[20] = 4 * _w3.m256_f32[1] - 4 * _w3.m256_f32[2] - _w3.m256_f32[3] + _w3.m256_f32[4];
				//v_data[21] = -2 * _w3.m256_f32[1] - _w3.m256_f32[2] + 2 * _w3.m256_f32[3] + _w3.m256_f32[4];
				//v_data[22] = 2 * _w3.m256_f32[1] - _w3.m256_f32[2] - 2 * _w3.m256_f32[3] + _w3.m256_f32[4];
				//v_data[23] = 4 * _w3.m256_f32[1] - 5 * _w3.m256_f32[3] + _w3.m256_f32[5];

				//v_data[24] = 4 * _w4.m256_f32[0] - 5 * _w4.m256_f32[2] + _w4.m256_f32[4];
				//v_data[25] = -4 * _w4.m256_f32[1] - 4 * _w4.m256_f32[2] + _w4.m256_f32[3] + _w4.m256_f32[4];
				//v_data[26] = 4 * _w4.m256_f32[1] - 4 * _w4.m256_f32[2] - _w4.m256_f32[3] + _w4.m256_f32[4];
				//v_data[27] = -2 * _w4.m256_f32[1] - _w4.m256_f32[2] + 2 * _w4.m256_f32[3] + _w4.m256_f32[4];
				//v_data[28] = 2 * _w4.m256_f32[1] - _w4.m256_f32[2] - 2 * _w4.m256_f32[3] + _w4.m256_f32[4];
				//v_data[29] = 4 * _w4.m256_f32[1] - 5 * _w4.m256_f32[3] + _w4.m256_f32[5];

				//v_data[30] = 4 * _w5.m256_f32[0] - 5 * _w5.m256_f32[2] + _w5.m256_f32[4];
				//v_data[31] = -4 * _w5.m256_f32[1] - 4 * _w5.m256_f32[2] + _w5.m256_f32[3] + _w5.m256_f32[4];
				//v_data[32] = 4 * _w5.m256_f32[1] - 4 * _w5.m256_f32[2] - _w5.m256_f32[3] + _w5.m256_f32[4];
				//v_data[33] = -2 * _w5.m256_f32[1] - _w5.m256_f32[2] + 2 * _w5.m256_f32[3] + _w5.m256_f32[4];
				//v_data[34] = 2 * _w5.m256_f32[1] - _w5.m256_f32[2] - 2 * _w5.m256_f32[3] + _w5.m256_f32[4];
				//v_data[35] = 4 * _w5.m256_f32[1] - 5 * _w5.m256_f32[3] + _w5.m256_f32[5];
#endif
			}

			inline void calculate_ATmA23(const float *m_data, float *result)
			{
#if SIMD_TYPE >= SIMDTYPE_SSE
				__m128 _d0, _d1, _d2, _d3;
				__m128 _w0, _w1;

				// load
				_d0 = _mm_loadu_ps(m_data);
				_d1 = _mm_loadu_ps(m_data + 4);
				_d2 = _mm_loadu_ps(m_data + 8);
				_d3 = _mm_loadu_ps(m_data + 12);

				// w = A_t * d
				_w0 = _mm_add_ps(_mm_add_ps(_d0, _d1), _d2);
				_w1 = _mm_sub_ps(_mm_sub_ps(_d1, _d2), _d3);

				// save to result
				result[0] = _w0.m128_f32[0] + _w0.m128_f32[1] + _w0.m128_f32[2];
				result[1] = _w0.m128_f32[1] - _w0.m128_f32[2] - _w0.m128_f32[3];
				result[2] = _w1.m128_f32[0] + _w1.m128_f32[1] + _w1.m128_f32[2];
				result[3] = _w1.m128_f32[1] - _w1.m128_f32[2] - _w1.m128_f32[3];
#else
				result[0] = m_data[0] + m_data[1] + m_data[2] + m_data[4] + m_data[5] + m_data[6] + m_data[8] + m_data[9] + m_data[10];
				result[1] = m_data[1] - m_data[2] - m_data[3] + m_data[5] - m_data[6] - m_data[7] + m_data[9] - m_data[10] - m_data[11];
				result[2] = m_data[4] + m_data[5] + m_data[6] - m_data[8] - m_data[9] - m_data[10] - m_data[12] - m_data[13] - m_data[14];
				result[3] = m_data[5] - m_data[6] - m_data[7] - m_data[9] + m_data[10] + m_data[11] - m_data[13] + m_data[14] + m_data[15];
#endif
			}

			inline void calculate_ATmA43(const float *m_data, float *result)
			{
#if SIMD_TYPE >= SIMDTYPE_AVX
				__m256 _d0, _d1, _d2, _d3, _d4, _d5;
				__m256 _w0, _w1, _w2, _w3;
				__m256 _x2 = _mm256_set1_ps(2);
				__m256 _x4 = _mm256_set1_ps(4);

				// load
				_d0 = _mm256_loadu_ps(m_data);
				_d1 = _mm256_loadu_ps(m_data + 6);
				_d2 = _mm256_loadu_ps(m_data + 12);
				_d3 = _mm256_loadu_ps(m_data + 18);
				_d4 = _mm256_loadu_ps(m_data + 24);
				_d5 = _mm256_loadu_ps(m_data + 30);

				// w = A_t * d
				__m256 temp_1_add_2 = _mm256_add_ps(_d1, _d2);
				__m256 temp_3_add_4 = _mm256_add_ps(_d3, _d4);
				_w0 = _mm256_add_ps(_d0, _mm256_add_ps(temp_1_add_2, temp_3_add_4));

				__m256 temp_1_sub_2 = _mm256_sub_ps(_d1, _d2);
				__m256 temp_3_sub_4 = _mm256_sub_ps(_d3, _d4);
				temp_3_sub_4 = _mm256_mul_ps(temp_3_sub_4, _x2);
				_w1 = _mm256_add_ps(temp_1_sub_2, temp_3_sub_4);

				temp_3_add_4 = _mm256_mul_ps(temp_3_add_4, _x4);
				_w2 = _mm256_add_ps(temp_1_add_2, temp_3_add_4);

				temp_3_sub_4 = _mm256_mul_ps(temp_3_sub_4, _x4);
				_w3 = _mm256_add_ps(temp_1_sub_2, temp_3_sub_4);
				_w3 = _mm256_add_ps(_w3, _d5);

				__m128 _m0 = _mm_set_ps(_w3.m256_f32[0], _w2.m256_f32[0], _w1.m256_f32[0], _w0.m256_f32[0]);
				__m128 _m1 = _mm_set_ps(_w3.m256_f32[1], _w2.m256_f32[1], _w1.m256_f32[1], _w0.m256_f32[1]);
				__m128 _m2 = _mm_set_ps(_w3.m256_f32[2], _w2.m256_f32[2], _w1.m256_f32[2], _w0.m256_f32[2]);
				__m128 _m3 = _mm_set_ps(_w3.m256_f32[3], _w2.m256_f32[3], _w1.m256_f32[3], _w0.m256_f32[3]);
				__m128 _m4 = _mm_set_ps(_w3.m256_f32[4], _w2.m256_f32[4], _w1.m256_f32[4], _w0.m256_f32[4]);
				__m128 _m5 = _mm_set_ps(_w3.m256_f32[5], _w2.m256_f32[5], _w1.m256_f32[5], _w0.m256_f32[5]);

				__m128 _x2_128 = _mm_set1_ps(2);
				__m128 _x4_128 = _mm_set1_ps(4);

				__m128 temp_128_1_add_2 = _mm_add_ps(_m1, _m2);
				__m128 temp_128_3_add_4 = _mm_add_ps(_m3, _m4);
				__m128 _res0 = _mm_add_ps(_m0, _mm_add_ps(temp_128_1_add_2, temp_128_3_add_4));

				__m128 temp_128_1_sub_2 = _mm_sub_ps(_m1, _m2);
				__m128 temp_128_3_sub_4 = _mm_sub_ps(_m3, _m4);
				temp_128_3_sub_4 = _mm_mul_ps(temp_128_3_sub_4, _x2_128);
				__m128 _res1 = _mm_add_ps(temp_128_1_sub_2, temp_128_3_sub_4);

				temp_128_3_add_4 = _mm_mul_ps(temp_128_3_add_4, _x4_128);
				__m128 _res2 = _mm_add_ps(temp_128_1_add_2, temp_128_3_add_4);

				temp_128_3_sub_4 = _mm_mul_ps(temp_128_3_sub_4, _x4_128);
				__m128 _res3 = _mm_add_ps(temp_128_1_sub_2, temp_128_3_sub_4);
				_res3 = _mm_add_ps(_res3, _m5);

				// save to result
				result[0] = _res0.m128_f32[0];
				result[1] = _res1.m128_f32[0];
				result[2] = _res2.m128_f32[0];
				result[3] = _res3.m128_f32[0];

				result[4] = _res0.m128_f32[1];
				result[5] = _res1.m128_f32[1];
				result[6] = _res2.m128_f32[1];
				result[7] = _res3.m128_f32[1];

				result[8] = _res0.m128_f32[2];
				result[9] = _res1.m128_f32[2];
				result[10] = _res2.m128_f32[2];
				result[11] = _res3.m128_f32[2];

				result[12] = _res0.m128_f32[3];
				result[13] = _res1.m128_f32[3];
				result[14] = _res2.m128_f32[3];
				result[15] = _res3.m128_f32[3];

#else
				//// save to result
				//result[0] = _w0.m256_f32[0] + _w0.m256_f32[1] + _w0.m256_f32[2] + _w0.m256_f32[3] + _w0.m256_f32[4];
				//result[1] = _w0.m256_f32[1] - _w0.m256_f32[2] + 2 * _w0.m256_f32[3] - 2 * _w0.m256_f32[4];
				//result[2] = _w0.m256_f32[1] + _w0.m256_f32[2] + 4 * _w0.m256_f32[3] + 4 * _w0.m256_f32[4];
				//result[3] = _w0.m256_f32[1] - _w0.m256_f32[2] + 8 * _w0.m256_f32[3] - 8 * _w0.m256_f32[4] + _w0.m256_f32[5];

				//result[4] = _w1.m256_f32[0] + _w1.m256_f32[1] + _w1.m256_f32[2] + _w1.m256_f32[3] + _w1.m256_f32[4];
				//result[5] = _w1.m256_f32[1] - _w1.m256_f32[2] + 2 * _w1.m256_f32[3] - 2 * _w1.m256_f32[4];
				//result[6] = _w1.m256_f32[1] + _w1.m256_f32[2] + 4 * _w1.m256_f32[3] + 4 * _w1.m256_f32[4];
				//result[7] = _w1.m256_f32[1] - _w1.m256_f32[2] + 8 * _w1.m256_f32[3] - 8 * _w1.m256_f32[4] + _w1.m256_f32[5];

				//result[8] = _w2.m256_f32[0] + _w2.m256_f32[1] + _w2.m256_f32[2] + _w2.m256_f32[3] + _w2.m256_f32[4];
				//result[9] = _w2.m256_f32[1] - _w2.m256_f32[2] + 2 * _w2.m256_f32[3] - 2 * _w2.m256_f32[4];
				//result[10] = _w2.m256_f32[1] + _w2.m256_f32[2] + 4 * _w2.m256_f32[3] + 4 * _w2.m256_f32[4];
				//result[11] = _w2.m256_f32[1] - _w2.m256_f32[2] + 8 * _w2.m256_f32[3] - 8 * _w2.m256_f32[4] + _w2.m256_f32[5];

				//result[12] = _w3.m256_f32[0] + _w3.m256_f32[1] + _w3.m256_f32[2] + _w3.m256_f32[3] + _w3.m256_f32[4];
				//result[13] = _w3.m256_f32[1] - _w3.m256_f32[2] + 2 * _w3.m256_f32[3] - 2 * _w3.m256_f32[4];
				//result[14] = _w3.m256_f32[1] + _w3.m256_f32[2] + 4 * _w3.m256_f32[3] + 4 * _w3.m256_f32[4];
				//result[15] = _w3.m256_f32[1] - _w3.m256_f32[2] + 8 * _w3.m256_f32[3] - 8 * _w3.m256_f32[4] + _w3.m256_f32[5];
#endif
			}

			//int8
			inline void calculate_GgGT23(const signed char *weight_data, short *u_data)
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

			inline void calculate_BTdB23(const signed char *row_data1, const signed char *row_data2, const signed char *row_data3, const signed char *row_data4, short *v_data)
			{
				v_data[0] = short(row_data1[0]) - short(row_data1[2]) - short(row_data3[0]) + short(row_data3[2]);
				v_data[1] = short(row_data1[1]) + short(row_data1[2]) - short(row_data3[1]) - short(row_data3[2]);
				v_data[2] = -short(row_data1[1]) + short(row_data1[2]) + short(row_data3[1]) - short(row_data3[2]);
				v_data[3] = short(row_data1[1]) - short(row_data1[3]) - short(row_data3[1]) + short(row_data3[3]);
				v_data[4] = short(row_data2[0]) - short(row_data2[2]) + short(row_data3[0]) - short(row_data3[2]);
				v_data[5] = short(row_data2[1]) + short(row_data2[2]) + short(row_data3[1]) + short(row_data3[2]);
				v_data[6] = -short(row_data2[1]) + short(row_data2[2]) - short(row_data3[1]) + short(row_data3[2]);
				v_data[7] = short(row_data2[1]) - short(row_data2[3]) + short(row_data3[1]) - short(row_data3[3]);
				v_data[8] = -short(row_data2[0]) + short(row_data2[2]) + short(row_data3[0]) - short(row_data3[2]);
				v_data[9] = -short(row_data2[1]) - short(row_data2[2]) + short(row_data3[1]) + short(row_data3[2]);
				v_data[10] = short(row_data2[1]) - short(row_data2[2]) - short(row_data3[1]) + short(row_data3[2]);
				v_data[11] = -short(row_data2[1]) + short(row_data2[3]) + short(row_data3[1]) - short(row_data3[3]);
				v_data[12] = short(row_data2[0]) - short(row_data2[2]) - short(row_data4[0]) + short(row_data4[2]);
				v_data[13] = short(row_data2[1]) + short(row_data2[2]) - short(row_data4[1]) - short(row_data4[2]);
				v_data[14] = -short(row_data2[1]) + short(row_data2[2]) + short(row_data4[1]) - short(row_data4[2]);
				v_data[15] = short(row_data2[1]) - short(row_data2[3]) - short(row_data4[1]) + short(row_data4[3]);
			}

			inline void calculate_ATmA23(const int *m_data, int *result)
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
