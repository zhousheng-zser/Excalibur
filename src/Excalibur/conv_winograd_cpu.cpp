#include "conv_winograd_cpu.hpp"
#include <iostream>
namespace glasssix
{
	namespace excalibur
	{
		conv_winograd_cpu::conv_winograd_cpu(int input_Channel, int output_Channel, int group, int kernelSize, int stride, int pad, bool bias_term, int device, bool int8_quantization)
			: baseconv(input_Channel, output_Channel, group, kernelSize, stride, pad, bias_term, device, int8_quantization)
		{
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

		void conv_winograd_cpu::forward_gemm(const signed char* input, const signed char* weights, int* output, bool skip_im2col) {}

		void conv_winograd_cpu::forward_gemm(const float* input, const float* weights, float* output, bool skip_im2col) {}

		void conv_winograd_cpu::forward_bias(float* output, const float* bias)
		{
			if (order_ == NCHW)
			{
				math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_Channel_,
					output_spatial_dim_, 1, 1.0f, bias, bias_multiplier_data,
					1.0f, output);
			}
			else if (order_ == NHWC)
			{
				math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, output_spatial_dim_,
					output_Channel_, 1, 1.0f, bias_multiplier_data, bias,
					1.0f, output);
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		void conv_winograd_cpu::Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			CHECK_EQ(kernelSize_, 3);
			CHECK_EQ(stride_, 1);
			order_ = bottom->order();
			num_ = bottom->data_shape()[0];
			bottom_data = bottom->cpu_data();
			intput_shape_.clear();
			intput_shape_ = bottom->data_shape();
			bottom_dim_ = bottom->count(1, 4);

			if (int8_quantization_)
			{
				bottom_int8_.reset(new tensor<signed char>(std::vector<int>{num_ * bottom_dim_}));
				bottom_int8_data = bottom_int8_->mutable_cpu_data();

#if SIMD_TYPE >= SIMDTYPE_SSE
				int circle_num = num_ * bottom_dim_ / mm_align_size;
				int index = 0;
				mm_type scale = mm_set1_ps(scales_data[0]);

				for (; index < circle_num; index++)
				{
					int index_offset = index * mm_align_size;
					mm_type data = mm_load_ps(bottom_data + index_offset);
					mm_type res_mul = mm_mul_ps(data, scale);
					mm_type res_round = mm_round_ps(res_mul, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
					mm_store_ps(bottom_round_data_, res_round);
					for (int i = 0; i < mm_align_size; i++)
					{
						if (bottom_round_data_[i] > 127)
						{
							bottom_int8_data[index_offset + i] = (signed char)(127);
						}
						else if (bottom_round_data_[i] < -128)
						{
							bottom_int8_data[index_offset + i] = (signed char)(-128);
						}
						else
						{
							bottom_int8_data[index_offset + i] = (signed char)(bottom_round_data_[i]);
						}
					}
				}

				for (index = mm_align_size * index; index < num_ * bottom_dim_; index++)
				{
					bottom_int8_data[index] = float32_to_int8(bottom_data[index] * scales_data[0]);
				}
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (int index = 0; index < num_ * bottom_dim_; index++)
				{
					bottom_int8_data[index] = float32_to_int8(bottom_data[index] * scales_data[0]);
				}
#endif

				if (order_ == NCHW)
				{
					input_dim_h_ = bottom->data_shape()[2];
					input_dim_w_ = bottom->data_shape()[3];
					input_spatial_dim_ = input_dim_h_ * input_dim_w_;

					output_dim_h_ = (input_dim_h_ + 2 * pad_ - kernelSize_) / stride_ + 1;
					output_dim_w_ = (input_dim_w_ + 2 * pad_ - kernelSize_) / stride_ + 1;
					output_spatial_dim_ = output_dim_w_ * output_dim_h_;
					bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_spatial_dim_}, device_));
					bias_multiplier_data = bias_multiplier_->mutable_cpu_data();
					top.reset(new tensor<float>(std::vector<int>{num_, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
					top_data = top->mutable_cpu_data();
					top_dim_ = top->count(1, 4);

					top_int32_.reset(new tensor<int>(std::vector<int>{num_, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
					top_int32_data = top_int32_->mutable_cpu_data();

					int h_subtract_tilesize = input_dim_h_ + 2 * pad_ - tile_size_;
					int w_subtract_tilesize = input_dim_w_ + 2 * pad_ - tile_size_;
					h_tile_num_ = int(h_subtract_tilesize / m_ + 0.5f) + 1;//h_tile_num_ = ceil((H-(m+r-1))/m) + 1, H is height after padding
					w_tile_num_ = int(w_subtract_tilesize / m_ + 0.5f) + 1;//w_tile_num_ = ceil((W-(m+r-1))/m) + 1, W is width after padding
					int total_tile_num = h_tile_num_ * w_tile_num_;
					int w_tile_stride = w_tile_num_ * tile_length_;
					int h_w_tile_stride = h_tile_num_ * w_tile_stride;
					int h_aligned = (h_subtract_tilesize + m_ - 1) / m_ * m_;
					int w_aligned = (w_subtract_tilesize + m_ - 1) / m_ * m_;
					int add_h = h_aligned - h_subtract_tilesize;
					int add_w = w_aligned - w_subtract_tilesize;

					if (group_ > 1)
					{
						bool is_U_calculated = false;

						for (int n = 0; n < num_; n++)
						{
							int bottom_offset_num = n * bottom_dim_;
							int top_offset_num = n * top_dim_;
#ifdef _OPENMP
#pragma omp parallel for
#endif
							for (int och = 0; och < output_Channel_; och++)
							{
								std::shared_ptr<tensor<int>> M_, RESULT_;
								std::shared_ptr<tensor<signed char>> TILE_;
								std::shared_ptr<tensor<short>> v_;
								M_.reset(new tensor<int>(std::vector<int>{tile_length_}));
								RESULT_.reset(new tensor<int>(std::vector<int>{m_length_}));
								TILE_.reset(new tensor<signed char>(std::vector<int>{tile_length_}));
								v_.reset(new tensor<short>(std::vector<int>{tile_length_}));
								int *m_data = M_->mutable_cpu_data();
								int *result = RESULT_->mutable_cpu_data();
								signed char *tile_data = TILE_->mutable_cpu_data();
								short *v_data = v_->mutable_cpu_data();

								int U_offset_och = och * tile_length_;
								int top_offset_num_channel = top_offset_num + och * output_spatial_dim_;
								int bottom_offset_num_channel = bottom_offset_num + och * input_spatial_dim_;

								if (!is_U_calculated)
								{
									calculate_GgGT(weights_int8_data + kernel_length_ * och, U_int16_data + U_offset_och);//calculate U
								}

								//param declaration
								int row_in_output_data;
								int row_in_input_data;
								int bottom_offset_num_channel_row;
								int top_offset_num_channel_row;
								int num_w;
								int col_in_output_data;
								int col_in_input_data;
								int bottom_offset_num_channel_row_col;
								int top_offset_num_channel_row_col;
								int row_in_tile;
								int tile_offset_row;
								int real_row;
								int col_in_tile;
								int real_col;
								int tile_offset_row_col;
								int row;
								int col;
								int result_offset_row;

#if SIMD_TYPE >= SIMDTYPE_AVX
								__m128i u_1_int16;
								__m128i u_2_int16;
								__m128i v_1_int16;
								__m128i v_2_int16;
								mm_typei u_1_int32;
								mm_typei u_2_int32;
								mm_typei v_1_int32;
								mm_typei v_2_int32;
								mm_typei sum_1;
								mm_typei sum_2;
#elif SIMD_TYPE >= SIMDTYPE_SSE
								mm_typei u_1_int16;
								mm_typei u_2_int16;
								mm_typei u_3_int16;
								mm_typei u_4_int16;
								mm_typei v_1_int16;
								mm_typei v_2_int16;
								mm_typei v_3_int16;
								mm_typei v_4_int16;
								mm_typei u_1_int32;
								mm_typei u_2_int32;
								mm_typei u_3_int32;
								mm_typei u_4_int32;
								mm_typei v_1_int32;
								mm_typei v_2_int32;
								mm_typei v_3_int32;
								mm_typei v_4_int32;
								mm_typei sum_1;
								mm_typei sum_2;
								mm_typei sum_3;
								mm_typei sum_4;
#endif

								for (int num_h = 0; num_h < h_tile_num_; num_h++)
								{
									row_in_output_data = num_h * m_;
									row_in_input_data = row_in_output_data - pad_;
									bottom_offset_num_channel_row = bottom_offset_num_channel + row_in_input_data * input_dim_w_;
									top_offset_num_channel_row = top_offset_num_channel + row_in_output_data * output_dim_w_;
									for (num_w = 0; num_w < w_tile_num_; num_w++)
									{
										col_in_output_data = num_w * m_;
										col_in_input_data = col_in_output_data - pad_;
										bottom_offset_num_channel_row_col = bottom_offset_num_channel_row + col_in_input_data;
										top_offset_num_channel_row_col = top_offset_num_channel_row + col_in_output_data;

										for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
										{
											tile_offset_row = row_in_tile * tile_size_;
											real_row = row_in_input_data + row_in_tile;
											if (!is_a_ge_zero_and_a_lt_b(real_row, input_dim_h_))
											{
												memset(tile_data + tile_offset_row, 0, tile_size_ * sizeof(signed char));
											}
											else
											{
												for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
												{
													real_col = col_in_input_data + col_in_tile;
													if (!is_a_ge_zero_and_a_lt_b(real_col, input_dim_w_))
													{
														tile_data[tile_offset_row + col_in_tile] = 0;
													}
													else
													{
														tile_data[tile_offset_row + col_in_tile] = bottom_int8_data[bottom_offset_num_channel_row_col + col_in_tile];
													}
												}

											}
											bottom_offset_num_channel_row_col += input_dim_w_;
										}

										calculate_BTdB(tile_data, v_data);//calculate V

#if SIMD_TYPE >= SIMDTYPE_AVX
										u_1_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och));
										u_2_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och + 8));
										v_1_int16 = _mm_load_si128((__m128i*)(v_data));
										v_2_int16 = _mm_load_si128((__m128i*)(v_data + 8));
										u_1_int32 = mm_cvtepi16_epi32(u_1_int16);
										u_2_int32 = mm_cvtepi16_epi32(u_2_int16);
										v_1_int32 = mm_cvtepi16_epi32(v_1_int16);
										v_2_int32 = mm_cvtepi16_epi32(v_2_int16);
										sum_1 = mm_mullo_epi32(u_1_int32, v_1_int32);
										sum_2 = mm_mullo_epi32(u_2_int32, v_2_int32);
										mm_store_si((mm_typei*)m_data, sum_1);
										mm_store_si((mm_typei*)(m_data + 8), sum_2);
#elif SIMD_TYPE >= SIMDTYPE_SSE
										u_1_int16 = _mm_load_si128((mm_typei*)(U_int16_data + U_offset_och));
										u_2_int16 = _mm_load_si128((mm_typei*)(U_int16_data + U_offset_och + 4));
										u_3_int16 = _mm_load_si128((mm_typei*)(U_int16_data + U_offset_och + 8));
										u_4_int16 = _mm_load_si128((mm_typei*)(U_int16_data + U_offset_och + 12));
										v_1_int16 = _mm_load_si128((mm_typei*)(v_data));
										v_2_int16 = _mm_load_si128((mm_typei*)(v_data + 4));
										v_3_int16 = _mm_load_si128((mm_typei*)(v_data + 8));
										v_4_int16 = _mm_load_si128((mm_typei*)(v_data + 12));
										u_1_int32 = mm_cvtepi16_epi32(u_1_int16);
										u_2_int32 = mm_cvtepi16_epi32(u_2_int16);
										u_3_int32 = mm_cvtepi16_epi32(u_3_int16);
										u_4_int32 = mm_cvtepi16_epi32(u_4_int16);
										v_1_int32 = mm_cvtepi16_epi32(v_1_int16);
										v_2_int32 = mm_cvtepi16_epi32(v_2_int16);
										v_3_int32 = mm_cvtepi16_epi32(v_3_int16);
										v_4_int32 = mm_cvtepi16_epi32(v_4_int16);
										sum_1 = mm_mullo_epi32(u_1_int32, v_1_int32);
										sum_2 = mm_mullo_epi32(u_2_int32, v_2_int32);
										sum_3 = mm_mullo_epi32(u_3_int32, v_3_int32);
										sum_4 = mm_mullo_epi32(u_4_int32, v_4_int32);
										mm_store_si((mm_typei*)m_data, sum_1);
										mm_store_si((mm_typei*)m_data + 4, sum_2);
										mm_store_si((mm_typei*)m_data + 8, sum_3);
										mm_store_si((mm_typei*)m_data + 12, sum_4);
#else
										for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
										{
											tile_offset_row = row_in_tile * tile_size_;
											for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
											{
												tile_offset_row_col = tile_offset_row + col_in_tile;
												m_data[tile_offset_row_col] = U_int16_data[U_offset_och + tile_offset_row_col] * v_data[tile_offset_row_col];
											}
										}
#endif

										calculate_ATmA(m_data, result);//calculate result

										if (num_h == h_tile_num_ - 1)
										{
											for (row = 0; row < m_ - add_h; row++)
											{
												result_offset_row = row * m_;
												if (num_w == w_tile_num_ - 1)
												{
													for (col = 0; col < m_ - add_w; col++)
													{
														top_int32_data[top_offset_num_channel_row_col + col] = result[result_offset_row + col];
													}
												}
												else
												{
													for (col = 0; col < m_; col++)
													{
														top_int32_data[top_offset_num_channel_row_col + col] = result[result_offset_row + col];
													}
												}
												top_offset_num_channel_row_col += output_dim_w_;
											}
										}
										else
										{
											for (row = 0; row < m_; row++)
											{
												result_offset_row = row * m_;
												if (num_w == w_tile_num_ - 1)
												{
													for (col = 0; col < m_ - add_w; col++)
													{
														top_int32_data[top_offset_num_channel_row_col + col] = result[result_offset_row + col];
													}
												}
												else
												{
													for (col = 0; col < m_; col++)
													{
														top_int32_data[top_offset_num_channel_row_col + col] = result[result_offset_row + col];
													}
												}
												top_offset_num_channel_row_col += output_dim_w_;
											}
										}
									}
								}
							}

							int offset = top_dim_ / group_;

#if SIMD_TYPE >= SIMDTYPE_SSE
							int circle_num = offset / mm_align_size;
							for (int j = 0; j < group_; j++)
							{
								float total_scale = scales_data[0] * scales_data[1 + j] * 4.0f;//we have multiply 4 in function: calculate_GgGT
								mm_type scale = mm_set1_ps(1.0f / total_scale);

								int index = 0;
								for (; index < circle_num; index++)
								{
									int index_offset = index * mm_align_size;
									mm_typei temp1 = mm_load_si((mm_typei*)(top_int32_data + top_offset_num + j * offset + index_offset));
									mm_type temp2 = mm_cvtepi32_ps(temp1);
									mm_type res = mm_mul_ps(temp2, scale);
									mm_store_ps(top_data + top_offset_num + j * offset + index_offset, res);
								}

								for (index = index * mm_align_size; index < offset; index++)
								{
									top_data[index + top_offset_num + j * offset] = top_int32_data[index + top_offset_num + j * offset] / total_scale;
								}
							}
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
							for (int j = 0; j < group_; j++)
							{
								float total_scale = scales_data[0] * scales_data[1 + j] * 4.0f;//we have multiply 4 in function: calculate_GgGT
								for (int index = 0; index < offset; index++)
								{
									top_data[index + n * top_dim_ + j * offset] = top_int32_data[index + n * top_dim_ + j * offset] / total_scale;
								}
							}
#endif

							math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_data);
							if (bias_term_)
							{
								forward_bias(top_data + top_offset_num, bias_data);
							}

							is_U_calculated = true;
						}
					}
					else if (group_ == 1)
					{
						//calculate U_
#ifdef _OPENMP
#pragma omp parallel for
#endif
						for (int n = 0; n < U_num_; ++n)
						{
							calculate_GgGT(weights_int8_data + kernel_length_ * n, U_int16_data + tile_length_ * n);//calculate U
						}

						//V=BT*d*B,so V has the same number as data tile, there are tile_length_ elements in single V
						V_num_ = input_Channel_ * total_tile_num;
						V_int16.reset(new tensor<short>(std::vector<int>{V_num_ * tile_length_}));
						V_int16_data = V_int16->mutable_cpu_data();
						int U_offset_single_och = input_Channel_ * tile_length_;

						for (int n = 0; n < num_; n++)
						{
							int bottom_offset_num = n * bottom_dim_;
							int top_offset_num = n * top_dim_;
							bool is_V_calculated = false;//only calculate V when is_V_calculated==false

#ifdef _OPENMP
#pragma omp parallel for
#endif
							for (int och = 0; och < output_Channel_; och++)
							{
								std::shared_ptr<tensor<int>> M_, RESULT_;
								std::shared_ptr<tensor<signed char>> TILE_;
								M_.reset(new tensor<int>(std::vector<int>{tile_length_}));
								RESULT_.reset(new tensor<int>(std::vector<int>{m_length_}));
								TILE_.reset(new tensor<signed char>(std::vector<int>{tile_length_}));
								int *m_data = M_->mutable_cpu_data();
								int *result = RESULT_->mutable_cpu_data();
								signed char *tile_data = TILE_->mutable_cpu_data();

								int U_offset_och = och * U_offset_single_och;
								int top_offset_num_channel = top_offset_num + och * output_spatial_dim_;

								//param declaration
								int row_in_output_data;
								int row_in_input_data;
								int top_offset_num_channel_row;
								int bottom_offset_num_row;
								int V_offset_row;
								int num_w;
								int col_in_output_data;
								int col_in_input_data;
								int top_offset_num_channel_row_col;
								int bottom_offset_num_row_col;
								int V_offset_row_col;
								int ich;
								int bottom_offset_num_channel_row_col;
								int V_offset_channel_row_col;
								int U_offset_och_ich;
								int row_in_tile;
								int tile_offset_row;
								int tile_offset_row_col;
								int real_row;
								int col_in_tile;
								int real_col;
								int row;
								int col;
								int result_offset_row;

#if SIMD_TYPE >= SIMDTYPE_AVX
								__m128i u_1_int16;
								__m128i u_2_int16;
								__m128i v_1_int16;
								__m128i v_2_int16;
								mm_typei u_1_int32;
								mm_typei u_2_int32;
								mm_typei v_1_int32;
								mm_typei v_2_int32;
								__m128i res_mul_1_int16;
								__m128i res_mul_2_int16;
								mm_typei res_mul_1_int32;
								mm_typei res_mul_2_int32;
								mm_typei sum_1;
								mm_typei sum_2;
#elif SIMD_TYPE >= SIMDTYPE_SSE
								mm_typei u_1_int16;
								mm_typei u_2_int16;
								mm_typei u_3_int16;
								mm_typei u_4_int16;
								mm_typei v_1_int16;
								mm_typei v_2_int16;
								mm_typei v_3_int16;
								mm_typei v_4_int16;
								mm_typei u_1_int32;
								mm_typei u_2_int32;
								mm_typei u_3_int32;
								mm_typei u_4_int32;
								mm_typei v_1_int32;
								mm_typei v_2_int32;
								mm_typei v_3_int32;
								mm_typei v_4_int32;
								mm_typei res_mul_1_int32;
								mm_typei res_mul_2_int32;
								mm_typei res_mul_3_int32;
								mm_typei res_mul_4_int32;
								mm_typei sum_1;
								mm_typei sum_2;
								mm_typei sum_3;
								mm_typei sum_4;
#endif


								for (int num_h = 0; num_h < h_tile_num_; num_h++)
								{
									row_in_output_data = num_h * m_;
									row_in_input_data = row_in_output_data - pad_;
									top_offset_num_channel_row = top_offset_num_channel + row_in_output_data * output_dim_w_;
									bottom_offset_num_row = bottom_offset_num + row_in_input_data * input_dim_w_;
									V_offset_row = num_h * w_tile_stride;
									for (num_w = 0; num_w < w_tile_num_; num_w++)
									{
										col_in_output_data = num_w * m_;
										col_in_input_data = col_in_output_data - pad_;
										top_offset_num_channel_row_col = top_offset_num_channel_row + col_in_output_data;
										bottom_offset_num_row_col = bottom_offset_num_row + col_in_input_data;
										V_offset_row_col = V_offset_row + num_w * tile_length_;

#if SIMD_TYPE >= SIMDTYPE_AVX
										sum_1 = mm_setzero_si();
										sum_2 = mm_setzero_si();
#elif SIMD_TYPE >= SIMDTYPE_SSE
										sum_1 = mm_setzero_si();
										sum_2 = mm_setzero_si();
										sum_3 = mm_setzero_si();
										sum_4 = mm_setzero_si();
#else 
										memset(m_data, 0, tile_length_ * sizeof(int));
#endif

										for (ich = 0; ich < input_Channel_; ich++)
										{
											bottom_offset_num_channel_row_col = bottom_offset_num_row_col + ich * input_spatial_dim_;
											V_offset_channel_row_col = ich * h_w_tile_stride + V_offset_row_col;
											U_offset_och_ich = U_offset_och + ich * tile_length_;

											//calculate V when is_V_calculated==false
											if (!is_V_calculated)
											{
												for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
												{
													tile_offset_row = row_in_tile * tile_size_;
													real_row = row_in_input_data + row_in_tile;
													if (!is_a_ge_zero_and_a_lt_b(real_row, input_dim_h_))
													{
														memset(tile_data + tile_offset_row, 0, tile_size_ * sizeof(signed char));
													}
													else
													{
														if (is_a_ge_zero_and_a_lt_b(col_in_input_data, input_dim_w_) && is_a_ge_zero_and_a_lt_b(col_in_input_data + tile_size_, input_dim_w_))
														{
															memcpy(tile_data + tile_offset_row, bottom_int8_data + bottom_offset_num_channel_row_col, tile_size_ * sizeof(signed char));
														}
														else
														{
															for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
															{
																real_col = col_in_input_data + col_in_tile;
																if (!is_a_ge_zero_and_a_lt_b(real_col, input_dim_w_))
																{
																	tile_data[tile_offset_row + col_in_tile] = (signed char)0;
																}
																else
																{
																	tile_data[tile_offset_row + col_in_tile] = bottom_int8_data[bottom_offset_num_channel_row_col + col_in_tile];
																}
															}
														}
													}
													bottom_offset_num_channel_row_col += input_dim_w_;
												}

												calculate_BTdB(tile_data, V_int16_data + V_offset_channel_row_col);
											}

#if SIMD_TYPE >= SIMDTYPE_AVX
											u_1_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich));
											u_2_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich + 8));
											v_1_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_channel_row_col));
											v_2_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_channel_row_col + 8));
											u_1_int32 = mm_cvtepi16_epi32(u_1_int16);
											u_2_int32 = mm_cvtepi16_epi32(u_2_int16);
											v_1_int32 = mm_cvtepi16_epi32(v_1_int16);
											v_2_int32 = mm_cvtepi16_epi32(v_2_int16);
											res_mul_1_int32 = mm_mullo_epi32(u_1_int32, v_1_int32);
											res_mul_2_int32 = mm_mullo_epi32(u_2_int32, v_2_int32);
											sum_1 = mm_add_epi32(res_mul_1_int32, sum_1);
											sum_2 = mm_add_epi32(res_mul_2_int32, sum_2);
#elif SIMD_TYPE >= SIMDTYPE_SSE
											u_1_int16 = _mm_load_si128((mm_typei*)(U_int16_data + U_offset_och_ich));
											u_2_int16 = _mm_load_si128((mm_typei*)(U_int16_data + U_offset_och_ich + 4));
											u_3_int16 = _mm_load_si128((mm_typei*)(U_int16_data + U_offset_och_ich + 8));
											u_4_int16 = _mm_load_si128((mm_typei*)(U_int16_data + U_offset_och_ich + 12));
											v_1_int16 = _mm_load_si128((mm_typei*)(V_int16_data + V_offset_channel_row_col));
											v_2_int16 = _mm_load_si128((mm_typei*)(V_int16_data + V_offset_channel_row_col + 4));
											v_3_int16 = _mm_load_si128((mm_typei*)(V_int16_data + V_offset_channel_row_col + 8));
											v_4_int16 = _mm_load_si128((mm_typei*)(V_int16_data + V_offset_channel_row_col + 12));
											u_1_int32 = mm_cvtepi16_epi32(u_1_int16);
											u_2_int32 = mm_cvtepi16_epi32(u_2_int16);
											u_3_int32 = mm_cvtepi16_epi32(u_3_int16);
											u_4_int32 = mm_cvtepi16_epi32(u_4_int16);
											v_1_int32 = mm_cvtepi16_epi32(v_1_int16);
											v_2_int32 = mm_cvtepi16_epi32(v_2_int16);
											v_3_int32 = mm_cvtepi16_epi32(v_3_int16);
											v_4_int32 = mm_cvtepi16_epi32(v_4_int16);
											res_mul_1_int32 = mm_mullo_epi32(u_1_int32, v_1_int32);
											res_mul_2_int32 = mm_mullo_epi32(u_2_int32, v_2_int32);
											res_mul_3_int32 = mm_mullo_epi32(u_3_int32, v_3_int32);
											res_mul_4_int32 = mm_mullo_epi32(u_4_int32, v_4_int32);
											sum_1 = mm_add_epi32(res_mul_1_int32, sum_1);
											sum_2 = mm_add_epi32(res_mul_2_int32, sum_2);
											sum_3 = mm_add_epi32(res_mul_3_int32, sum_3);
											sum_4 = mm_add_epi32(res_mul_4_int32, sum_4);
#else
											for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
											{
												tile_offset_row = row_in_tile * tile_size_;
												for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
												{
													tile_offset_row_col = tile_offset_row + col_in_tile;
													m_data[tile_offset_row_col] += U_int16_data[U_offset_och_ich + tile_offset_row_col] * V_int16_data[V_offset_channel_row_col + tile_offset_row_col];
												}
											}
#endif
										}

#if SIMD_TYPE >= SIMDTYPE_AVX
										mm_store_si((mm_typei*)m_data, sum_1);
										mm_store_si((mm_typei*)(m_data + 8), sum_2);
#elif SIMD_TYPE >= SIMDTYPE_SSE
										mm_store_si((mm_typei*)m_data, sum_1);
										mm_store_si((mm_typei*)(m_data + 4), sum_2);
										mm_store_si((mm_typei*)(m_data + 8), sum_3);
										mm_store_si((mm_typei*)(m_data + 12), sum_4);
#endif

										calculate_ATmA(m_data, result);

										if (num_h == h_tile_num_ - 1)
										{
											for (row = 0; row < m_ - add_h; row++)
											{
												result_offset_row = row * m_;
												if (num_w == w_tile_num_ - 1)
												{
													for (col = 0; col < m_ - add_w; col++)
													{
														top_int32_data[top_offset_num_channel_row_col + col] = result[result_offset_row + col];
													}
												}
												else
												{
													for (col = 0; col < m_; col++)
													{
														top_int32_data[top_offset_num_channel_row_col + col] = result[result_offset_row + col];
													}
												}
												top_offset_num_channel_row_col += output_dim_w_;
											}
										}
										else
										{
											for (row = 0; row < m_; row++)
											{
												result_offset_row = row * m_;
												if (num_w == w_tile_num_ - 1)
												{
													for (col = 0; col < m_ - add_w; col++)
													{
														top_int32_data[top_offset_num_channel_row_col + col] = result[result_offset_row + col];
													}
												}
												else
												{
													for (col = 0; col < m_; col++)
													{
														top_int32_data[top_offset_num_channel_row_col + col] = result[result_offset_row + col];
													}
												}
												top_offset_num_channel_row_col += output_dim_w_;
											}
										}
									}
								}

								is_V_calculated = true;
							}

							int offset = top_dim_ / group_;

#if SIMD_TYPE >= SIMDTYPE_SSE
							int circle_num = offset / mm_align_size;
							for (int j = 0; j < group_; j++)
							{
								float total_scale = scales_data[0] * scales_data[1 + j] * 4.0f;//we have mutiply 4 in function: calculate_GgGT
								mm_type scale = mm_set1_ps(1.0f / total_scale);
								int index = 0;

								for (; index < circle_num; index++)
								{
									int index_offset = index * mm_align_size;
									mm_typei temp1 = mm_load_si((mm_typei*)(top_int32_data + top_offset_num + j * offset + index_offset));
									mm_type temp2 = mm_cvtepi32_ps(temp1);
									mm_type res = mm_mul_ps(temp2, scale);
									mm_store_ps(top_data + top_offset_num + j * offset + index_offset, res);
								}

								for (index = index * mm_align_size; index < offset; index++)
								{
									top_data[index + top_offset_num + j * offset] = top_int32_data[index + top_offset_num + j * offset] / total_scale;
								}
							}
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
							for (int j = 0; j < group_; j++)
							{
								float total_scale = scales_data[0] * scales_data[1 + j] * 4.0f;//we have mutiply 4 in function: calculate_GgGT
								for (int index = 0; index < offset; index++)
								{
									top_data[index + n * top_dim_ + j * offset] = top_int32_data[index + n * top_dim_ + j * offset] / total_scale;
								}
							}
#endif

							math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_data);
							if (bias_term_)
							{
								forward_bias(top_data + top_offset_num, bias_data);
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
					input_dim_h_ = bottom->data_shape()[1];
					input_dim_w_ = bottom->data_shape()[2];
					input_spatial_dim_ = input_dim_h_ * input_dim_w_;
					int input_w_stride = input_dim_w_ * input_Channel_;
					bottom_dim_ = bottom->count(1, 4);

					output_dim_h_ = (input_dim_h_ + 2 * pad_ - kernelSize_) / stride_ + 1;
					output_dim_w_ = (input_dim_w_ + 2 * pad_ - kernelSize_) / stride_ + 1;
					output_spatial_dim_ = output_dim_h_ * output_dim_w_;
					int output_w_stride = output_dim_w_ * output_Channel_;
					bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_spatial_dim_}, device_));
					bias_multiplier_data = bias_multiplier_->mutable_cpu_data();
					top.reset(new tensor<float>(std::vector<int>{num_, output_dim_h_, output_dim_w_, output_Channel_}, device_, order_));
					top_dim_ = (top)->count(1, 4);
					float* top_data = top->mutable_cpu_data();

					top_int32_.reset(new tensor<int>(std::vector<int>{num_, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
					top_int32_data = top_int32_->mutable_cpu_data();

					int h_subtract_tilesize = input_dim_h_ + 2 * pad_ - tile_size_;
					int w_subtract_tilesize = input_dim_w_ + 2 * pad_ - tile_size_;
					h_tile_num_ = int(h_subtract_tilesize / m_ + 0.5f) + 1;//h_tile_num_ = ceil((H-(m+r-1))/m) + 1, H is height after padding
					w_tile_num_ = int(w_subtract_tilesize / m_ + 0.5f) + 1;//w_tile_num_ = ceil((W-(m+r-1))/m) + 1, W is width after padding
					int total_tile_num = h_tile_num_ * w_tile_num_;
					int w_tile_stride = w_tile_num_ * tile_length_;
					int h_w_tile_stride = h_tile_num_ * w_tile_stride;
					int h_aligned = (h_subtract_tilesize + m_ - 1) / m_ * m_;
					int w_aligned = (w_subtract_tilesize + m_ - 1) / m_ * m_;
					int add_h = h_aligned - h_subtract_tilesize;
					int add_w = w_aligned - w_subtract_tilesize;

					if (group_ > 1)
					{
						bool is_U_calculated = false;

						for (int n = 0; n < num_; n++)
						{
							int bottom_offset_num = n * bottom_dim_;
							int top_offset_num = n * top_dim_;
#ifdef _OPENMP
#pragma omp parallel for
#endif
							for (int och = 0; och < output_Channel_; och++)
							{
								std::shared_ptr<tensor<int>> M_, RESULT_;
								std::shared_ptr<tensor<signed char>> TILE_;
								std::shared_ptr<tensor<short>> v_;
								M_.reset(new tensor<int>(std::vector<int>{tile_length_}));
								RESULT_.reset(new tensor<int>(std::vector<int>{m_length_}));
								TILE_.reset(new tensor<signed char>(std::vector<int>{tile_length_}));
								v_.reset(new tensor<short>(std::vector<int>{tile_length_}));
								int *m_data = M_->mutable_cpu_data();
								int *result = RESULT_->mutable_cpu_data();
								signed char *tile_data = TILE_->mutable_cpu_data();
								short *v_data = v_->mutable_cpu_data();

								int bottom_offset_num_channel = bottom_offset_num + och;
								int top_offset_num_channel = top_offset_num + och;
								int U_offset_och = och * tile_length_;

								if (!is_U_calculated)
								{
									calculate_GgGT(weights_int8_data + kernel_length_ * och, U_int16_data + U_offset_och);//calculate U
								}

								//param declaration
								int row_in_output_data;
								int row_in_input_data;
								int bottom_offset_num_channel_row;
								int top_offset_num_channel_row;
								int num_w;
								int col_in_output_data;
								int col_in_input_data;
								int bottom_offset_num_channel_row_col;
								int top_offset_num_channel_row_col;
								int row_in_tile;
								int tile_offset_row;
								int real_row;
								int col_in_tile;
								int real_col;
								int tile_offset_row_col;
								int row;
								int result_offset_row;
								int col;

#if SIMD_TYPE >= SIMDTYPE_AVX
								__m128i u_1_int16;
								__m128i u_2_int16;
								__m128i v_1_int16;
								__m128i v_2_int16;
								mm_typei u_1_int32;
								mm_typei u_2_int32;
								mm_typei v_1_int32;
								mm_typei v_2_int32;
								mm_typei sum_1;
								mm_typei sum_2;
#elif SIMD_TYPE >= SIMDTYPE_SSE
								mm_typei u_1_int16;
								mm_typei u_2_int16;
								mm_typei u_3_int16;
								mm_typei u_4_int16;
								mm_typei v_1_int16;
								mm_typei v_2_int16;
								mm_typei v_3_int16;
								mm_typei v_4_int16;
								mm_typei u_1_int32;
								mm_typei u_2_int32;
								mm_typei u_3_int32;
								mm_typei u_4_int32;
								mm_typei v_1_int32;
								mm_typei v_2_int32;
								mm_typei v_3_int32;
								mm_typei v_4_int32;
								mm_typei sum_1;
								mm_typei sum_2;
								mm_typei sum_3;
								mm_typei sum_4;
#endif

								for (int num_h = 0; num_h < h_tile_num_; num_h++)
								{
									row_in_output_data = num_h * m_;
									row_in_input_data = row_in_output_data - pad_;
									bottom_offset_num_channel_row = bottom_offset_num_channel + row_in_input_data * input_w_stride;
									top_offset_num_channel_row = top_offset_num_channel + row_in_output_data * output_w_stride;
									for (num_w = 0; num_w < w_tile_num_; num_w++)
									{
										col_in_output_data = num_w * m_;
										col_in_input_data = col_in_output_data - pad_;
										bottom_offset_num_channel_row_col = bottom_offset_num_channel_row + col_in_input_data * input_Channel_;
										top_offset_num_channel_row_col = top_offset_num_channel_row + col_in_output_data * output_Channel_;

										for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
										{
											tile_offset_row = row_in_tile * tile_size_;
											real_row = row_in_input_data + row_in_tile;
											if (!is_a_ge_zero_and_a_lt_b(real_row, input_dim_h_))
											{
												memset(tile_data + tile_offset_row, 0, tile_size_ * sizeof(signed char));
											}
											else
											{
												for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
												{
													real_col = col_in_input_data + col_in_tile;
													if (!is_a_ge_zero_and_a_lt_b(real_col, input_dim_w_))
													{
														tile_data[tile_offset_row + col_in_tile] = 0;
													}
													else
													{
														//output_Channel_ and input_Channel_ has the same value, so we use output_Channel_ instead
														tile_data[tile_offset_row + col_in_tile] = bottom_int8_data[bottom_offset_num_channel_row_col + col_in_tile * input_Channel_];
													}
												}
											}
											bottom_offset_num_channel_row_col += input_w_stride;
										}

										calculate_BTdB(tile_data, v_data);//calculate V

#if SIMD_TYPE >= SIMDTYPE_AVX
										u_1_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och));
										u_2_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och + 8));
										v_1_int16 = _mm_load_si128((__m128i*)(v_data));
										v_2_int16 = _mm_load_si128((__m128i*)(v_data + 8));
										u_1_int32 = mm_cvtepi16_epi32(u_1_int16);
										u_2_int32 = mm_cvtepi16_epi32(u_2_int16);
										v_1_int32 = mm_cvtepi16_epi32(v_1_int16);
										v_2_int32 = mm_cvtepi16_epi32(v_2_int16);
										sum_1 = _mm256_mullo_epi32(u_1_int32, v_1_int32);
										sum_2 = _mm256_mullo_epi32(u_2_int32, v_2_int32);
										_mm256_store_si256((mm_typei*)m_data, sum_1);
										_mm256_store_si256((mm_typei*)(m_data + 8), sum_2);
#elif SIMD_TYPE >= SIMDTYPE_SSE
										u_1_int16 = _mm_load_si128((mm_typei*)(U_int16_data + U_offset_och));
										u_2_int16 = _mm_load_si128((mm_typei*)(U_int16_data + U_offset_och + 4));
										u_3_int16 = _mm_load_si128((mm_typei*)(U_int16_data + U_offset_och + 8));
										u_4_int16 = _mm_load_si128((mm_typei*)(U_int16_data + U_offset_och + 12));
										v_1_int16 = _mm_load_si128((mm_typei*)(v_data));
										v_2_int16 = _mm_load_si128((mm_typei*)(v_data + 4));
										v_3_int16 = _mm_load_si128((mm_typei*)(v_data + 8));
										v_4_int16 = _mm_load_si128((mm_typei*)(v_data + 12));
										u_1_int32 = mm_cvtepi16_epi32(u_1_int16);
										u_2_int32 = mm_cvtepi16_epi32(u_2_int16);
										u_3_int32 = mm_cvtepi16_epi32(u_3_int16);
										u_4_int32 = mm_cvtepi16_epi32(u_4_int16);
										v_1_int32 = mm_cvtepi16_epi32(v_1_int16);
										v_2_int32 = mm_cvtepi16_epi32(v_2_int16);
										v_3_int32 = mm_cvtepi16_epi32(v_3_int16);
										v_4_int32 = mm_cvtepi16_epi32(v_4_int16);
										sum_1 = mm_mullo_epi32(u_1_int32, v_1_int32);
										sum_2 = mm_mullo_epi32(u_2_int32, v_2_int32);
										sum_3 = mm_mullo_epi32(u_3_int32, v_3_int32);
										sum_4 = mm_mullo_epi32(u_4_int32, v_4_int32);
										mm_store_si((mm_typei*)m_data, sum_1);
										mm_store_si((mm_typei*)m_data + 4, sum_2);
										mm_store_si((mm_typei*)m_data + 8, sum_3);
										mm_store_si((mm_typei*)m_data + 12, sum_4);
#else
										for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
										{
											tile_offset_row = row_in_tile * tile_size_;
											for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
											{
												tile_offset_row_col = tile_offset_row + col_in_tile;
												m_data[tile_offset_row_col] = U_int16_data[U_offset_och + tile_offset_row_col] * v_data[tile_offset_row_col];
											}
										}
#endif

										calculate_ATmA(m_data, result);//calculate result

										if (num_h == h_tile_num_ - 1)
										{
											for (row = 0; row < m_ - add_h; row++)
											{
												result_offset_row = row * m_;
												if (num_w == w_tile_num_ - 1)
												{
													for (col = 0; col < m_ - add_w; col++)
													{
														top_int32_data[top_offset_num_channel_row_col + col * output_Channel_] = result[result_offset_row + col];
													}
												}
												else
												{
													for (col = 0; col < m_; col++)
													{
														top_int32_data[top_offset_num_channel_row_col + col * output_Channel_] = result[result_offset_row + col];
													}
												}
												top_offset_num_channel_row_col += output_w_stride;
											}
										}
										else
										{
											for (row = 0; row < m_; row++)
											{
												result_offset_row = row * m_;
												if (num_w == w_tile_num_ - 1)
												{
													for (col = 0; col < m_ - add_w; col++)
													{
														top_int32_data[top_offset_num_channel_row_col + col * output_Channel_] = result[result_offset_row + col];
													}
												}
												else
												{
													for (col = 0; col < m_; col++)
													{
														top_int32_data[top_offset_num_channel_row_col + col * output_Channel_] = result[result_offset_row + col];
													}
												}
												top_offset_num_channel_row_col += output_w_stride;
											}
										}
									}
								}
							}

							int offset = top_dim_ / group_;
							for (int j = 0; j < group_; j++)
							{
								float total_scale = scales_data[0] * scales_data[1 + j] * 4.0f;//we have mutiply 4 in function: calculate_GgGT
								for (int index = 0; index < offset; index++)
								{
									top_data[index * group_ + n * top_dim_ + j] = top_int32_data[index * group_ + n * top_dim_ + j] / total_scale;
								}
							}

							math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_data);
							if (bias_term_)
							{
								forward_bias(top_data + top_offset_num, bias_data);
							}

							is_U_calculated = true;
						}
					}
					else if (group_ == 1)
					{
						//calculate U_
#ifdef _OPENMP
#pragma omp parallel for
#endif
						for (int n = 0; n < U_num_; ++n)
						{
							calculate_GgGT(weights_int8_data + kernel_length_ * n, U_int16_data + tile_length_ * n);//calculate U
						}

						//V=BT*d*B,so V has the same number as data tile, there are tile_length_ elements in single V
						V_num_ = input_Channel_ * total_tile_num;
						V_int16.reset(new tensor<short>(std::vector<int>{V_num_ * tile_length_}));
						V_int16_data = V_int16->mutable_cpu_data();
						int U_offset_single_och = input_Channel_ * tile_length_;

						for (int n = 0; n < num_; n++)
						{
							int bottom_offset_num = n * bottom_dim_;
							int top_offset_num = n * top_dim_;
							bool is_V_calculated = false;//only calculate V when is_V_calculated==false

#ifdef _OPENMP
#pragma omp parallel for
#endif
							for (int och = 0; och < output_Channel_; och++)
							{
								std::shared_ptr<tensor<int>> M_, RESULT_;
								std::shared_ptr<tensor<signed char>> TILE_;
								M_.reset(new tensor<int>(std::vector<int>{tile_length_}));
								RESULT_.reset(new tensor<int>(std::vector<int>{m_length_}));
								TILE_.reset(new tensor<signed char>(std::vector<int>{tile_length_}));
								int *m_data = M_->mutable_cpu_data();
								int *result = RESULT_->mutable_cpu_data();
								signed char *tile_data = TILE_->mutable_cpu_data();

								int U_offset_och = och * U_offset_single_och;
								int top_offset_num_channel = top_offset_num + och;

								//param declaration
								int row_in_output_data;
								int row_in_input_data;
								int bottom_offset_num_row;
								int top_offset_num_channel_row;
								int V_offset_row;
								int num_w;
								int col_in_output_data;
								int col_in_input_data;
								int bottom_offset_num_row_col;
								int top_offset_num_channel_row_col;
								int V_offset_row_col;
								int ich;
								int V_offset_channel_row_col;
								int U_offset_och_ich;
								int bottom_offset_num_channel_row_col;
								int row_in_tile;
								int tile_offset_row;
								int real_row;
								int col_in_tile;
								int real_col;
								int row;
								int col;
								int tile_offset_row_col;
								int U_offset_och_ich_row;
								int V_offset_channel_row_col_rowt;
								int result_offset_row;

#if SIMD_TYPE >= SIMDTYPE_AVX
								__m128i u_1_int16;
								__m128i u_2_int16;
								__m128i v_1_int16;
								__m128i v_2_int16;
								mm_typei u_1_int32;
								mm_typei u_2_int32;
								mm_typei v_1_int32;
								mm_typei v_2_int32;
								mm_typei res_mul_1_int32;
								mm_typei res_mul_2_int32;
								mm_typei sum_1;
								mm_typei sum_2;
#elif SIMD_TYPE >= SIMDTYPE_SSE
								mm_typei u_1_int16;
								mm_typei u_2_int16;
								mm_typei u_3_int16;
								mm_typei u_4_int16;
								mm_typei v_1_int16;
								mm_typei v_2_int16;
								mm_typei v_3_int16;
								mm_typei v_4_int16;
								mm_typei u_1_int32;
								mm_typei u_2_int32;
								mm_typei u_3_int32;
								mm_typei u_4_int32;
								mm_typei v_1_int32;
								mm_typei v_2_int32;
								mm_typei v_3_int32;
								mm_typei v_4_int32;
								mm_typei res_mul_1_int32;
								mm_typei res_mul_2_int32;
								mm_typei res_mul_3_int32;
								mm_typei res_mul_4_int32;
								mm_typei sum_1;
								mm_typei sum_2;
								mm_typei sum_3;
								mm_typei sum_4;
#endif

								for (int num_h = 0; num_h < h_tile_num_; num_h++)
								{
									row_in_output_data = num_h * m_;
									row_in_input_data = row_in_output_data - pad_;
									bottom_offset_num_row = bottom_offset_num + row_in_input_data * input_w_stride;
									top_offset_num_channel_row = top_offset_num_channel + row_in_output_data * output_w_stride;
									V_offset_row = num_h * w_tile_stride;

									for (num_w = 0; num_w < w_tile_num_; num_w++)
									{
										col_in_output_data = num_w * m_;
										col_in_input_data = col_in_output_data - pad_;
										bottom_offset_num_row_col = bottom_offset_num_row + col_in_input_data * input_Channel_;
										top_offset_num_channel_row_col = top_offset_num_channel_row + col_in_output_data * output_Channel_;
										V_offset_row_col = V_offset_row + num_w * tile_length_;

#if SIMD_TYPE >= SIMDTYPE_AVX
										sum_1 = _mm256_setzero_si256();
										sum_2 = _mm256_setzero_si256();
#elif SIMD_TYPE >= SIMDTYPE_SSE
										sum_1 = mm_setzero_si();
										sum_2 = mm_setzero_si();
										sum_3 = mm_setzero_si();
										sum_4 = mm_setzero_si();
#else 
										memset(m_data, 0, tile_length_ * sizeof(int));
#endif

										for (ich = 0; ich < input_Channel_; ich++)
										{
											V_offset_channel_row_col = ich * h_w_tile_stride + V_offset_row_col;
											U_offset_och_ich = U_offset_och + ich * tile_length_;
											bottom_offset_num_channel_row_col = bottom_offset_num_row_col + ich;

											//calculate V when is_first==true
											if (!is_V_calculated)
											{
												for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
												{
													tile_offset_row = row_in_tile * tile_size_;
													real_row = row_in_input_data + row_in_tile;
													if (!is_a_ge_zero_and_a_lt_b(real_row, input_dim_h_))
													{
														memset(tile_data + tile_offset_row, 0, tile_size_ * sizeof(signed char));
													}
													else
													{
														for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
														{
															real_col = col_in_input_data + col_in_tile;
															if (!is_a_ge_zero_and_a_lt_b(real_col, input_dim_w_))
															{
																tile_data[tile_offset_row + col_in_tile] = 0;
															}
															else
															{
																tile_data[tile_offset_row + col_in_tile] = bottom_int8_data[bottom_offset_num_channel_row_col + col_in_tile * input_Channel_];
															}
														}
													}
													bottom_offset_num_channel_row_col += input_w_stride;
												}

												calculate_BTdB(tile_data, V_int16_data + V_offset_channel_row_col);//calculate v
											}

#if SIMD_TYPE >= SIMDTYPE_AVX
											u_1_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich));
											u_2_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich + 8));
											v_1_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_channel_row_col));
											v_2_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_channel_row_col + 8));
											u_1_int32 = mm_cvtepi16_epi32(u_1_int16);
											u_2_int32 = mm_cvtepi16_epi32(u_2_int16);
											v_1_int32 = mm_cvtepi16_epi32(v_1_int16);
											v_2_int32 = mm_cvtepi16_epi32(v_2_int16);
											res_mul_1_int32 = _mm256_mullo_epi32(u_1_int32, v_1_int32);
											res_mul_2_int32 = _mm256_mullo_epi32(u_2_int32, v_2_int32);
											sum_1 = _mm256_add_epi32(res_mul_1_int32, sum_1);
											sum_2 = _mm256_add_epi32(res_mul_2_int32, sum_2);
#elif SIMD_TYPE >= SIMDTYPE_SSE
											u_1_int16 = _mm_load_si128((mm_typei*)(U_int16_data + U_offset_och_ich));
											u_2_int16 = _mm_load_si128((mm_typei*)(U_int16_data + U_offset_och_ich + 4));
											u_3_int16 = _mm_load_si128((mm_typei*)(U_int16_data + U_offset_och_ich + 8));
											u_4_int16 = _mm_load_si128((mm_typei*)(U_int16_data + U_offset_och_ich + 12));
											v_1_int16 = _mm_load_si128((mm_typei*)(V_int16_data + V_offset_channel_row_col));
											v_2_int16 = _mm_load_si128((mm_typei*)(V_int16_data + V_offset_channel_row_col + 4));
											v_3_int16 = _mm_load_si128((mm_typei*)(V_int16_data + V_offset_channel_row_col + 8));
											v_4_int16 = _mm_load_si128((mm_typei*)(V_int16_data + V_offset_channel_row_col + 12));
											u_1_int32 = mm_cvtepi16_epi32(u_1_int16);
											u_2_int32 = mm_cvtepi16_epi32(u_2_int16);
											u_3_int32 = mm_cvtepi16_epi32(u_3_int16);
											u_4_int32 = mm_cvtepi16_epi32(u_4_int16);
											v_1_int32 = mm_cvtepi16_epi32(v_1_int16);
											v_2_int32 = mm_cvtepi16_epi32(v_2_int16);
											v_3_int32 = mm_cvtepi16_epi32(v_3_int16);
											v_4_int32 = mm_cvtepi16_epi32(v_4_int16);
											res_mul_1_int32 = mm_mullo_epi32(u_1_int32, v_1_int32);
											res_mul_2_int32 = mm_mullo_epi32(u_2_int32, v_2_int32);
											res_mul_3_int32 = mm_mullo_epi32(u_3_int32, v_3_int32);
											res_mul_4_int32 = mm_mullo_epi32(u_4_int32, v_4_int32);
											sum_1 = mm_add_epi32(res_mul_1_int32, sum_1);
											sum_2 = mm_add_epi32(res_mul_2_int32, sum_2);
											sum_3 = mm_add_epi32(res_mul_3_int32, sum_3);
											sum_4 = mm_add_epi32(res_mul_4_int32, sum_4);
#else
											for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
											{
												tile_offset_row = row_in_tile * tile_size_;
												for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
												{
													tile_offset_row_col = tile_offset_row + col_in_tile;
													m_data[tile_offset_row_col] += U_int16_data[U_offset_och_ich + tile_offset_row_col] * V_int16_data[V_offset_channel_row_col + tile_offset_row_col];
												}
											}
#endif
										}

#if SIMD_TYPE >= SIMDTYPE_AVX
										mm_store_si((mm_typei*)m_data, sum_1);
										mm_store_si((mm_typei*)(m_data + 8), sum_2);
#elif SIMD_TYPE >= SIMDTYPE_SSE
										mm_store_si((mm_typei*)m_data, sum_1);
										mm_store_si((mm_typei*)(m_data + 4), sum_2);
										mm_store_si((mm_typei*)(m_data + 8), sum_3);
										mm_store_si((mm_typei*)(m_data + 12), sum_4);
#endif

										calculate_ATmA(m_data, result);//calculate result

										if (num_h == h_tile_num_ - 1)
										{
											for (row = 0; row < m_ - add_h; row++)
											{
												result_offset_row = row * m_;
												if (num_w == w_tile_num_ - 1)
												{
													for (col = 0; col < m_ - add_w; col++)
													{
														top_int32_data[top_offset_num_channel_row_col + col * output_Channel_] = result[result_offset_row + col];
													}
												}
												else
												{
													for (col = 0; col < m_; col++)
													{
														top_int32_data[top_offset_num_channel_row_col + col * output_Channel_] = result[result_offset_row + col];
													}
												}
												top_offset_num_channel_row_col += output_w_stride;
											}
										}
										else
										{
											for (row = 0; row < m_; row++)
											{
												result_offset_row = row * m_;
												if (num_w == w_tile_num_ - 1)
												{
													for (col = 0; col < m_ - add_w; col++)
													{
														top_int32_data[top_offset_num_channel_row_col + col * output_Channel_] = result[result_offset_row + col];
													}
												}
												else
												{
													for (col = 0; col < m_; col++)
													{
														top_int32_data[top_offset_num_channel_row_col + col * output_Channel_] = result[result_offset_row + col];
													}
												}
												top_offset_num_channel_row_col += output_w_stride;
											}
										}
									}
								}

								is_V_calculated = true;
							}

							int offset = top_dim_ / group_;
							for (int j = 0; j < group_; j++)
							{
								float total_scale = scales_data[0] * scales_data[1 + j] * 4.0f;//we have mutiply 4 in function: calculate_GgGT
								for (int index = 0; index < offset; index++)
								{
									top_data[index * group_ + n * top_dim_ + j] = top_int32_data[index * group_ + n * top_dim_ + j] / total_scale;
								}
							}

							math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_data);
							if (bias_term_)
							{
								forward_bias(top_data + top_offset_num, bias_data);
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
			else
			{
				if (order_ == NCHW)
				{
					input_dim_h_ = bottom->data_shape()[2];
					input_dim_w_ = bottom->data_shape()[3];
					input_spatial_dim_ = input_dim_h_ * input_dim_w_;
					bottom_dim_ = bottom->count(1, 4);

					output_dim_h_ = (input_dim_h_ + 2 * pad_ - kernelSize_) / stride_ + 1;
					output_dim_w_ = (input_dim_w_ + 2 * pad_ - kernelSize_) / stride_ + 1;
					output_spatial_dim_ = output_dim_w_ * output_dim_h_;
					bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_spatial_dim_}, device_));
					bias_multiplier_data = bias_multiplier_->mutable_cpu_data();
					top.reset(new tensor<float>(std::vector<int>{num_, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
					top_data = top->mutable_cpu_data();
					top_dim_ = top->count(1, 4);

					int h_subtract_tilesize = input_dim_h_ + 2 * pad_ - tile_size_;
					int w_subtract_tilesize = input_dim_w_ + 2 * pad_ - tile_size_;
					h_tile_num_ = int(h_subtract_tilesize / m_ + 0.5f) + 1;//h_tile_num_ = ceil((H-(m+r-1))/m) + 1, H is height after padding
					w_tile_num_ = int(w_subtract_tilesize / m_ + 0.5f) + 1;//w_tile_num_ = ceil((W-(m+r-1))/m) + 1, W is width after padding
					int total_tile_num = h_tile_num_ * w_tile_num_;
					int w_tile_stride = w_tile_num_ * tile_length_;
					int h_w_tile_stride = h_tile_num_ * w_tile_stride;
					int h_aligned = (h_subtract_tilesize + m_ - 1) / m_ * m_;
					int w_aligned = (w_subtract_tilesize + m_ - 1) / m_ * m_;
					int add_h = h_aligned - h_subtract_tilesize;
					int add_w = w_aligned - w_subtract_tilesize;

					if (group_ > 1)
					{
						bool is_U_calculated = false;

						for (int n = 0; n < num_; n++)
						{
							int bottom_offset_num = n * bottom_dim_;
							int top_offset_num = n * top_dim_;
#ifdef _OPENMP
#pragma omp parallel for
#endif
							for (int och = 0; och < output_Channel_; och++)
							{
								std::shared_ptr<tensor<float>> TILE_, M_, RESULT_, v_;
								TILE_.reset(new tensor<float>(std::vector<int>{tile_length_}));
								M_.reset(new tensor<float>(std::vector<int>{tile_length_}));
								RESULT_.reset(new tensor<float>(std::vector<int>{m_length_}));
								v_.reset(new tensor<float>(std::vector<int>{tile_length_}));
								float *tile_data = TILE_->mutable_cpu_data();
								float *m_data = M_->mutable_cpu_data();
								float *result = RESULT_->mutable_cpu_data();
								float *v_data = v_->mutable_cpu_data();

								int U_offset_och = och * tile_length_;
								int top_offset_num_channel = top_offset_num + och * output_spatial_dim_;
								int bottom_offset_num_channel = bottom_offset_num + och * input_spatial_dim_;

								if (!is_U_calculated)
								{
									calculate_GgGT(weights_data + kernel_length_ * och, U_data + U_offset_och);//calculate U
								}

								//param declaration
								int row_in_output_data;
								int row_in_input_data;
								int bottom_offset_num_channel_row;
								int top_offset_num_channel_row;
								int num_w;
								int col_in_output_data;
								int col_in_input_data;
								int bottom_offset_num_channel_row_col;
								int top_offset_num_channel_row_col;
								int row_in_tile;
								int tile_offset_row;
								int real_row;
								int col_in_tile;
								int real_col;
								int tile_offset_row_col;
								int row;
								int col;
								int result_offset_row;

#if SIMD_TYPE >= SIMDTYPE_AVX
								mm_type sum_1;
								mm_type sum_2;
								mm_type u_1;
								mm_type u_2;
								mm_type v_1;
								mm_type v_2;
#elif SIMD_TYPE >= SIMDTYPE_SSE
								mm_type sum_1;
								mm_type sum_2;
								mm_type sum_3;
								mm_type sum_4;
								mm_type u_1;
								mm_type u_2;
								mm_type u_3;
								mm_type u_4;
								mm_type v_1;
								mm_type v_2;
								mm_type v_3;
								mm_type v_4;
#endif

								for (int num_h = 0; num_h < h_tile_num_; num_h++)
								{
									row_in_output_data = num_h * m_;
									row_in_input_data = row_in_output_data - pad_;
									bottom_offset_num_channel_row = bottom_offset_num_channel + row_in_input_data * input_dim_w_;
									top_offset_num_channel_row = top_offset_num_channel + row_in_output_data * output_dim_w_;
									for (num_w = 0; num_w < w_tile_num_; num_w++)
									{
										col_in_output_data = num_w * m_;
										col_in_input_data = col_in_output_data - pad_;
										bottom_offset_num_channel_row_col = bottom_offset_num_channel_row + col_in_input_data;
										top_offset_num_channel_row_col = top_offset_num_channel_row + col_in_output_data;

										for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
										{
											tile_offset_row = row_in_tile * tile_size_;
											real_row = row_in_input_data + row_in_tile;
											if (!is_a_ge_zero_and_a_lt_b(real_row, input_dim_h_))
											{
												memset(tile_data + tile_offset_row, 0, tile_size_ * sizeof(float));
											}
											else
											{
												for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
												{
													real_col = col_in_input_data + col_in_tile;
													if (!is_a_ge_zero_and_a_lt_b(real_col, input_dim_w_))
													{
														tile_data[tile_offset_row + col_in_tile] = 0;
													}
													else
													{
														tile_data[tile_offset_row + col_in_tile] = bottom_data[bottom_offset_num_channel_row_col + col_in_tile];
													}
												}

											}
											bottom_offset_num_channel_row_col += input_dim_w_;
										}

										calculate_BTdB(tile_data, v_data);//calculate V

#if SIMD_TYPE >= SIMDTYPE_AVX
										u_1 = mm_load_ps(U_data + U_offset_och);
										u_2 = mm_load_ps(U_data + U_offset_och + 8);
										v_1 = mm_load_ps(v_data);
										v_2 = mm_load_ps(v_data + 8);
										sum_1 = mm_mul_ps(u_1, v_1);
										sum_2 = mm_mul_ps(u_2, v_2);
										mm_store_ps(m_data, sum_1);
										mm_store_ps(m_data + 8, sum_2);
#elif SIMD_TYPE >= SIMDTYPE_SSE
										u_1 = mm_load_ps(U_data + U_offset_och);
										u_2 = mm_load_ps(U_data + U_offset_och + 4);
										u_3 = mm_load_ps(U_data + U_offset_och + 8);
										u_4 = mm_load_ps(U_data + U_offset_och + 12);
										v_1 = mm_load_ps(v_data);
										v_2 = mm_load_ps(v_data + 4);
										v_3 = mm_load_ps(v_data + 8);
										v_4 = mm_load_ps(v_data + 12);
										sum_1 = mm_mul_ps(u_1, v_1);
										sum_2 = mm_mul_ps(u_2, v_2);
										sum_3 = mm_mul_ps(u_3, v_3);
										sum_4 = mm_mul_ps(u_4, v_4);
										mm_store_ps(m_data, sum_1);
										mm_store_ps(m_data + 4, sum_2);
										mm_store_ps(m_data + 8, sum_3);
										mm_store_ps(m_data + 12, sum_4);
#else
										for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
										{
											tile_offset_row = row_in_tile * tile_size_;
											for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
											{
												tile_offset_row_col = tile_offset_row + col_in_tile;
												m_data[tile_offset_row_col] = U_data[U_offset_och + tile_offset_row_col] * v_data[tile_offset_row_col];
											}
										}
#endif

										calculate_ATmA(m_data, result);//calculate result

										if (num_h == h_tile_num_ - 1)
										{
											for (row = 0; row < m_ - add_h; row++)
											{
												result_offset_row = row * m_;
												if (num_w == w_tile_num_ - 1)
												{
													for (col = 0; col < m_ - add_w; col++)
													{
														top_data[top_offset_num_channel_row_col + col] = result[result_offset_row + col];
													}
												}
												else
												{
													for (col = 0; col < m_; col++)
													{
														top_data[top_offset_num_channel_row_col + col] = result[result_offset_row + col];
													}
												}
												top_offset_num_channel_row_col += output_dim_w_;
											}
										}
										else
										{
											for (row = 0; row < m_; row++)
											{
												result_offset_row = row * m_;
												if (num_w == w_tile_num_ - 1)
												{
													for (col = 0; col < m_ - add_w; col++)
													{
														top_data[top_offset_num_channel_row_col + col] = result[result_offset_row + col];
													}
												}
												else
												{
													for (col = 0; col < m_; col++)
													{
														top_data[top_offset_num_channel_row_col + col] = result[result_offset_row + col];
													}
												}
												top_offset_num_channel_row_col += output_dim_w_;
											}
										}
									}
								}
							}

							math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_data);
							if (bias_term_)
							{
								forward_bias(top_data + top_offset_num, bias_data);
							}

							is_U_calculated = true;
						}
					}
					else if (group_ == 1)
					{
						//calculate U_
#ifdef _OPENMP
#pragma omp parallel for
#endif
						for (int n = 0; n < U_num_; ++n)
						{
							calculate_GgGT(weights_data + kernel_length_ * n, U_data + tile_length_ * n);//calculate U
						}

						//V=BT*d*B,so V has the same number as data tile, there are tile_length_ elements in single V
						V_num_ = input_Channel_ * total_tile_num;
						V_.reset(new tensor<float>(std::vector<int>{V_num_ * tile_length_}));
						V_data = V_->mutable_cpu_data();
						int U_offset_single_och = input_Channel_ * tile_length_;

						for (int n = 0; n < num_; n++)
						{
							int bottom_offset_num = n * bottom_dim_;
							int top_offset_num = n * top_dim_;
							bool is_V_calculated = false;//only calculate V when is_V_calculated==false
#ifdef _OPENMP
#pragma omp parallel for
#endif
							for (int och = 0; och < output_Channel_; och++)
							{
								std::shared_ptr<tensor<float>> TILE_, M_, RESULT_;
								TILE_.reset(new tensor<float>(std::vector<int>{tile_length_}));
								M_.reset(new tensor<float>(std::vector<int>{tile_length_}));
								RESULT_.reset(new tensor<float>(std::vector<int>{m_length_}));
								float *tile_data = TILE_->mutable_cpu_data();
								float *m_data = M_->mutable_cpu_data();
								float *result = RESULT_->mutable_cpu_data();

								int U_offset_och = och * U_offset_single_och;
								int top_offset_num_channel = top_offset_num + och * output_spatial_dim_;

								//param declaration
								int row_in_output_data;
								int row_in_input_data;
								int top_offset_num_channel_row;
								int bottom_offset_num_row;
								int V_offset_row;
								int num_w;
								int col_in_output_data;
								int col_in_input_data;
								int top_offset_num_channel_row_col;
								int bottom_offset_num_row_col;
								int V_offset_row_col;
								int ich;
								int bottom_offset_num_channel_row_col;
								int V_offset_channel_row_col;
								int U_offset_och_ich;
								int row_in_tile;
								int tile_offset_row;
								int tile_offset_row_col;
								int real_row;
								int col_in_tile;
								int real_col;
								int row;
								int col;
								int result_offset_row;

#if SIMD_TYPE >= SIMDTYPE_AVX
								mm_type sum_1;
								mm_type sum_2;
								mm_type u_1;
								mm_type u_2;
								mm_type v_1;
								mm_type v_2;
#elif SIMD_TYPE >= SIMDTYPE_SSE
								mm_type sum_1;
								mm_type sum_2;
								mm_type sum_3;
								mm_type sum_4;
								mm_type u_1;
								mm_type u_2;
								mm_type u_3;
								mm_type u_4;
								mm_type v_1;
								mm_type v_2;
								mm_type v_3;
								mm_type v_4;
#endif

								for (int num_h = 0; num_h < h_tile_num_; num_h++)
								{
									row_in_output_data = num_h * m_;
									row_in_input_data = row_in_output_data - pad_;
									top_offset_num_channel_row = top_offset_num_channel + row_in_output_data * output_dim_w_;
									bottom_offset_num_row = bottom_offset_num + row_in_input_data * input_dim_w_;
									V_offset_row = num_h * w_tile_stride;
									for (num_w = 0; num_w < w_tile_num_; num_w++)
									{
										col_in_output_data = num_w * m_;
										col_in_input_data = col_in_output_data - pad_;
										top_offset_num_channel_row_col = top_offset_num_channel_row + col_in_output_data;
										bottom_offset_num_row_col = bottom_offset_num_row + col_in_input_data;
										V_offset_row_col = V_offset_row + num_w * tile_length_;

#if SIMD_TYPE >= SIMDTYPE_AVX
										sum_1 = mm_setzero_ps();
										sum_2 = mm_setzero_ps();
#elif SIMD_TYPE >= SIMDTYPE_SSE
										sum_1 = mm_setzero_ps();
										sum_2 = mm_setzero_ps();
										sum_3 = mm_setzero_ps();
										sum_4 = mm_setzero_ps();
#else
										memset(m_data, 0, tile_length_ * sizeof(float));
#endif // SIMD_TYPE >= SIMDTYPE_AVX

										for (ich = 0; ich < input_Channel_; ich++)
										{
											bottom_offset_num_channel_row_col = bottom_offset_num_row_col + ich * input_spatial_dim_;
											V_offset_channel_row_col = ich * h_w_tile_stride + V_offset_row_col;
											U_offset_och_ich = U_offset_och + ich * tile_length_;

											//calculate V when is_V_calculated==false
											if (!is_V_calculated)
											{
												for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
												{
													tile_offset_row = row_in_tile * tile_size_;
													real_row = row_in_input_data + row_in_tile;
													if (!is_a_ge_zero_and_a_lt_b(real_row, input_dim_h_))
													{
														memset(tile_data + tile_offset_row, 0, tile_size_ * sizeof(float));
													}
													else
													{
														if (is_a_ge_zero_and_a_lt_b(col_in_input_data, input_dim_w_) && is_a_ge_zero_and_a_lt_b(col_in_input_data + tile_size_, input_dim_w_))
														{
															memcpy(tile_data + tile_offset_row, bottom_data + bottom_offset_num_channel_row_col, tile_size_ * sizeof(float));
														}
														else
														{
															for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
															{
																real_col = col_in_input_data + col_in_tile;
																if (!is_a_ge_zero_and_a_lt_b(real_col, input_dim_w_))
																{
																	tile_data[tile_offset_row + col_in_tile] = 0;
																}
																else
																{
																	tile_data[tile_offset_row + col_in_tile] = bottom_data[bottom_offset_num_channel_row_col + col_in_tile];
																}
															}
														}
													}
													bottom_offset_num_channel_row_col += input_dim_w_;
												}

												calculate_BTdB(tile_data, V_data + V_offset_channel_row_col);
											}

#if SIMD_TYPE >= SIMDTYPE_AVX
											u_1 = mm_load_ps(U_data + U_offset_och_ich);
											u_2 = mm_load_ps(U_data + U_offset_och_ich + 8);
											v_1 = mm_load_ps(V_data + V_offset_channel_row_col);
											v_2 = mm_load_ps(V_data + V_offset_channel_row_col + 8);
											sum_1 = mm_fmadd_ps(v_1, u_1, sum_1);
											sum_2 = mm_fmadd_ps(v_2, u_2, sum_2);
#elif SIMD_TYPE >= SIMDTYPE_SSE
											u_1 = mm_load_ps(U_data + U_offset_och_ich);
											u_2 = mm_load_ps(U_data + U_offset_och_ich + 4);
											u_3 = mm_load_ps(U_data + U_offset_och_ich + 8);
											u_4 = mm_load_ps(U_data + U_offset_och_ich + 12);
											v_1 = mm_load_ps(V_data + V_offset_channel_row_col);
											v_2 = mm_load_ps(V_data + V_offset_channel_row_col + 4);
											v_3 = mm_load_ps(V_data + V_offset_channel_row_col + 8);
											v_4 = mm_load_ps(V_data + V_offset_channel_row_col + 12);
											sum_1 = mm_fmadd_ps(v_1, u_1, sum_1);
											sum_2 = mm_fmadd_ps(v_2, u_2, sum_2);
											sum_3 = mm_fmadd_ps(v_3, u_3, sum_3);
											sum_4 = mm_fmadd_ps(v_4, u_4, sum_4);
#else
											for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
											{
												tile_offset_row = row_in_tile * tile_size_;
												for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
												{
													tile_offset_row_col = tile_offset_row + col_in_tile;
													m_data[tile_offset_row_col] += U_data[U_offset_och_ich + tile_offset_row_col] * V_data[V_offset_channel_row_col + tile_offset_row_col];
												}
											}
#endif
										}

#if SIMD_TYPE >= SIMDTYPE_AVX
										mm_store_ps(m_data, sum_1);
										mm_store_ps(m_data + 8, sum_2);
#elif SIMD_TYPE >= SIMDTYPE_SSE
										mm_store_ps(m_data, sum_1);
										mm_store_ps(m_data + 4, sum_2);
										mm_store_ps(m_data + 8, sum_3);
										mm_store_ps(m_data + 12, sum_4);
#endif // SIMD_TYPE >= SIMDTYPE_AVX

										calculate_ATmA(m_data, result);

										if (num_h == h_tile_num_ - 1)
										{
											for (row = 0; row < m_ - add_h; row++)
											{
												result_offset_row = row * m_;
												if (num_w == w_tile_num_ - 1)
												{
													for (col = 0; col < m_ - add_w; col++)
													{
														top_data[top_offset_num_channel_row_col + col] = result[result_offset_row + col];
													}
												}
												else
												{
													for (col = 0; col < m_; col++)
													{
														top_data[top_offset_num_channel_row_col + col] = result[result_offset_row + col];
													}
												}
												top_offset_num_channel_row_col += output_dim_w_;
											}
										}
										else
										{
											for (row = 0; row < m_; row++)
											{
												result_offset_row = row * m_;
												if (num_w == w_tile_num_ - 1)
												{
													for (col = 0; col < m_ - add_w; col++)
													{
														top_data[top_offset_num_channel_row_col + col] = result[result_offset_row + col];
													}
												}
												else
												{
													for (col = 0; col < m_; col++)
													{
														top_data[top_offset_num_channel_row_col + col] = result[result_offset_row + col];
													}
												}
												top_offset_num_channel_row_col += output_dim_w_;
											}
										}
									}
								}

								is_V_calculated = true;
							}

							math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_data);
							if (bias_term_)
							{
								forward_bias(top_data + top_offset_num, bias_data);
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
					input_dim_h_ = bottom->data_shape()[1];
					input_dim_w_ = bottom->data_shape()[2];
					input_spatial_dim_ = input_dim_h_ * input_dim_w_;
					int input_w_stride = input_dim_w_ * input_Channel_;
					bottom_dim_ = bottom->count(1, 4);

					output_dim_h_ = (input_dim_h_ + 2 * pad_ - kernelSize_) / stride_ + 1;
					output_dim_w_ = (input_dim_w_ + 2 * pad_ - kernelSize_) / stride_ + 1;
					output_spatial_dim_ = output_dim_h_ * output_dim_w_;
					int output_w_stride = output_dim_w_ * output_Channel_;
					bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_spatial_dim_}, device_));
					bias_multiplier_data = bias_multiplier_->mutable_cpu_data();
					top.reset(new tensor<float>(std::vector<int>{num_, output_dim_h_, output_dim_w_, output_Channel_}, device_, order_));
					top_dim_ = (top)->count(1, 4);
					float* top_data = top->mutable_cpu_data();

					int h_subtract_tilesize = input_dim_h_ + 2 * pad_ - tile_size_;
					int w_subtract_tilesize = input_dim_w_ + 2 * pad_ - tile_size_;
					h_tile_num_ = int(h_subtract_tilesize / m_ + 0.5f) + 1;//h_tile_num_ = ceil((H-(m+r-1))/m) + 1, H is height after padding
					w_tile_num_ = int(w_subtract_tilesize / m_ + 0.5f) + 1;//w_tile_num_ = ceil((W-(m+r-1))/m) + 1, W is width after padding
					int total_tile_num = h_tile_num_ * w_tile_num_;
					int w_tile_stride = w_tile_num_ * tile_length_;
					int h_w_tile_stride = h_tile_num_ * w_tile_stride;
					int h_aligned = (h_subtract_tilesize + m_ - 1) / m_ * m_;
					int w_aligned = (w_subtract_tilesize + m_ - 1) / m_ * m_;
					int add_h = h_aligned - h_subtract_tilesize;
					int add_w = w_aligned - w_subtract_tilesize;

					if (group_ > 1)
					{
						bool is_U_calculated = false;

						for (int n = 0; n < num_; n++)
						{
							int bottom_offset_num = n * bottom_dim_;
							int top_offset_num = n * top_dim_;
#ifdef _OPENMP
#pragma omp parallel for
#endif
							for (int och = 0; och < output_Channel_; och++)
							{
								std::shared_ptr<tensor<float>> TILE_, M_, RESULT_, v_;
								TILE_.reset(new tensor<float>(std::vector<int>{tile_length_}));
								M_.reset(new tensor<float>(std::vector<int>{tile_length_}));
								RESULT_.reset(new tensor<float>(std::vector<int>{m_length_}));
								v_.reset(new tensor<float>(std::vector<int>{tile_length_}));
								float *tile_data = TILE_->mutable_cpu_data();
								float *m_data = M_->mutable_cpu_data();
								float *result = RESULT_->mutable_cpu_data();
								float *v_data = v_->mutable_cpu_data();

								int bottom_offset_num_channel = bottom_offset_num + och;
								int top_offset_num_channel = top_offset_num + och;
								int U_offset_och = och * tile_length_;

								if (!is_U_calculated)
								{
									calculate_GgGT(weights_data + kernel_length_ * och, U_data + U_offset_och);//calculate U
								}

								//param declaration
								int row_in_output_data;
								int row_in_input_data;
								int bottom_offset_num_channel_row;
								int top_offset_num_channel_row;
								int num_w;
								int col_in_output_data;
								int col_in_input_data;
								int bottom_offset_num_channel_row_col;
								int top_offset_num_channel_row_col;
								int row_in_tile;
								int tile_offset_row;
								int real_row;
								int col_in_tile;
								int real_col;
								int tile_offset_row_col;
								int row;
								int result_offset_row;
								int col;

#if SIMD_TYPE >= SIMDTYPE_AVX
								mm_type u_1;
								mm_type u_2;
								mm_type v_1;
								mm_type v_2;
								mm_type sum_1;
								mm_type sum_2;
#elif SIMD_TYPE >= SIMDTYPE_SSE
								mm_type u_1;
								mm_type u_2;
								mm_type u_3;
								mm_type u_4;
								mm_type v_1;
								mm_type v_2;
								mm_type v_3;
								mm_type v_4;
								mm_type sum_1;
								mm_type sum_2;
								mm_type sum_3;
								mm_type sum_4;
#endif

								for (int num_h = 0; num_h < h_tile_num_; num_h++)
								{
									row_in_output_data = num_h * m_;
									row_in_input_data = row_in_output_data - pad_;
									bottom_offset_num_channel_row = bottom_offset_num_channel + row_in_input_data * input_w_stride;
									top_offset_num_channel_row = top_offset_num_channel + row_in_output_data * output_w_stride;
									for (num_w = 0; num_w < w_tile_num_; num_w++)
									{
										col_in_output_data = num_w * m_;
										col_in_input_data = col_in_output_data - pad_;
										bottom_offset_num_channel_row_col = bottom_offset_num_channel_row + col_in_input_data * input_Channel_;
										top_offset_num_channel_row_col = top_offset_num_channel_row + col_in_output_data * output_Channel_;

										for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
										{
											tile_offset_row = row_in_tile * tile_size_;
											real_row = row_in_input_data + row_in_tile;
											if (!is_a_ge_zero_and_a_lt_b(real_row, input_dim_h_))
											{
												memset(tile_data + tile_offset_row, 0, tile_size_ * sizeof(float));
											}
											else
											{
												for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
												{
													real_col = col_in_input_data + col_in_tile;
													if (!is_a_ge_zero_and_a_lt_b(real_col, input_dim_w_))
													{
														tile_data[tile_offset_row + col_in_tile] = 0;
													}
													else
													{
														//output_Channel_ and input_Channel_ has the same value, so we use output_Channel_ instead
														tile_data[tile_offset_row + col_in_tile] = bottom_data[bottom_offset_num_channel_row_col + col_in_tile * input_Channel_];
													}
												}
											}
											bottom_offset_num_channel_row_col += input_w_stride;
										}

										calculate_BTdB(tile_data, v_data);//calculate V

#if SIMD_TYPE >= SIMDTYPE_AVX
										u_1 = mm_load_ps(U_data + U_offset_och);
										u_2 = mm_load_ps(U_data + U_offset_och + 8);
										v_1 = mm_load_ps(v_data);
										v_2 = mm_load_ps(v_data + 8);
										sum_1 = mm_mul_ps(u_1, v_1);
										sum_2 = mm_mul_ps(u_2, v_2);
										mm_store_ps(m_data, sum_1);
										mm_store_ps(m_data + 8, sum_2);
#elif SIMD_TYPE >= SIMDTYPE_SSE
										u_1 = mm_load_ps(U_data + U_offset_och);
										u_2 = mm_load_ps(U_data + U_offset_och + 4);
										u_3 = mm_load_ps(U_data + U_offset_och + 8);
										u_4 = mm_load_ps(U_data + U_offset_och + 12);
										v_1 = mm_load_ps(v_data);
										v_2 = mm_load_ps(v_data + 4);
										v_3 = mm_load_ps(v_data + 8);
										v_4 = mm_load_ps(v_data + 12);
										sum_1 = mm_mul_ps(u_1, v_1);
										sum_2 = mm_mul_ps(u_2, v_2);
										sum_3 = mm_mul_ps(u_3, v_3);
										sum_4 = mm_mul_ps(u_4, v_4);
										mm_store_ps(m_data, sum_1);
										mm_store_ps(m_data + 4, sum_2);
										mm_store_ps(m_data + 8, sum_3);
										mm_store_ps(m_data + 12, sum_4);
#else
										for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
										{
											tile_offset_row = row_in_tile * tile_size_;
											for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
											{
												tile_offset_row_col = tile_offset_row + col_in_tile;
												m_data[tile_offset_row_col] = U_data[U_offset_och + tile_offset_row_col] * v_data[tile_offset_row_col];
											}
										}
#endif

										calculate_ATmA(m_data, result);//calculate result

										if (num_h == h_tile_num_ - 1)
										{
											for (row = 0; row < m_ - add_h; row++)
											{
												result_offset_row = row * m_;
												if (num_w == w_tile_num_ - 1)
												{
													for (col = 0; col < m_ - add_w; col++)
													{
														top_data[top_offset_num_channel_row_col + col * output_Channel_] = result[result_offset_row + col];
													}
												}
												else
												{
													for (col = 0; col < m_; col++)
													{
														top_data[top_offset_num_channel_row_col + col * output_Channel_] = result[result_offset_row + col];
													}
												}
												top_offset_num_channel_row_col += output_w_stride;
											}
										}
										else
										{
											for (row = 0; row < m_; row++)
											{
												result_offset_row = row * m_;
												if (num_w == w_tile_num_ - 1)
												{
													for (col = 0; col < m_ - add_w; col++)
													{
														top_data[top_offset_num_channel_row_col + col * output_Channel_] = result[result_offset_row + col];
													}
												}
												else
												{
													for (col = 0; col < m_; col++)
													{
														top_data[top_offset_num_channel_row_col + col * output_Channel_] = result[result_offset_row + col];
													}
												}
												top_offset_num_channel_row_col += output_w_stride;
											}
										}
									}
								}
							}

							math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_data);
							if (bias_term_)
							{
								forward_bias(top_data + top_offset_num, bias_data);
							}

							is_U_calculated = true;
						}
					}
					else if (group_ == 1)
					{
						//calculate U_
#ifdef _OPENMP
#pragma omp parallel for
#endif
						for (int n = 0; n < U_num_; ++n)
						{
							calculate_GgGT(weights_data + kernel_length_ * n, U_data + tile_length_ * n);//calculate U
						}

						//V=BT*d*B,so V has the same number as data tile, there are tile_length_ elements in single V
						V_num_ = input_Channel_ * total_tile_num;
						V_.reset(new tensor<float>(std::vector<int>{V_num_ * tile_length_}));
						V_data = V_->mutable_cpu_data();
						int U_offset_single_och = input_Channel_ * tile_length_;

						for (int n = 0; n < num_; n++)
						{
							int bottom_offset_num = n * bottom_dim_;
							int top_offset_num = n * top_dim_;
							bool is_V_calculated = false;//only calculate V when is_V_calculated==false
#ifdef _OPENMP
#pragma omp parallel for
#endif
							for (int och = 0; och < output_Channel_; och++)
							{
								std::shared_ptr<tensor<float>> TILE_, M_, RESULT_;
								TILE_.reset(new tensor<float>(std::vector<int>{tile_length_}));
								M_.reset(new tensor<float>(std::vector<int>{tile_length_}));
								RESULT_.reset(new tensor<float>(std::vector<int>{m_length_}));
								float *tile_data = TILE_->mutable_cpu_data();
								float *m_data = M_->mutable_cpu_data();
								float *result = RESULT_->mutable_cpu_data();

								int U_offset_och = och * U_offset_single_och;
								int top_offset_num_channel = top_offset_num + och;

								//param declaration
								int row_in_output_data;
								int row_in_input_data;
								int bottom_offset_num_row;
								int top_offset_num_channel_row;
								int V_offset_row;
								int num_w;
								int col_in_output_data;
								int col_in_input_data;
								int bottom_offset_num_row_col;
								int top_offset_num_channel_row_col;
								int V_offset_row_col;
								int ich;
								int V_offset_channel_row_col;
								int U_offset_och_ich;
								int bottom_offset_num_channel_row_col;
								int row_in_tile;
								int tile_offset_row;
								int real_row;
								int col_in_tile;
								int real_col;
								int row;
								int col;
								int tile_offset_row_col;
								int U_offset_och_ich_row;
								int V_offset_channel_row_col_rowt;
								int result_offset_row;

#if SIMD_TYPE >= SIMDTYPE_AVX
								mm_type sum_1;
								mm_type sum_2;
								mm_type u_1;
								mm_type u_2;
								mm_type v_1;
								mm_type v_2;
#elif SIMD_TYPE >= SIMDTYPE_SSE
								mm_type sum_1;
								mm_type sum_2;
								mm_type sum_3;
								mm_type sum_4;
								mm_type u_1;
								mm_type u_2;
								mm_type u_3;
								mm_type u_4;
								mm_type v_1;
								mm_type v_2;
								mm_type v_3;
								mm_type v_4;
#endif

								for (int num_h = 0; num_h < h_tile_num_; num_h++)
								{
									row_in_output_data = num_h * m_;
									row_in_input_data = row_in_output_data - pad_;
									bottom_offset_num_row = bottom_offset_num + row_in_input_data * input_w_stride;
									top_offset_num_channel_row = top_offset_num_channel + row_in_output_data * output_w_stride;
									V_offset_row = num_h * w_tile_stride;

									for (num_w = 0; num_w < w_tile_num_; num_w++)
									{
										col_in_output_data = num_w * m_;
										col_in_input_data = col_in_output_data - pad_;
										bottom_offset_num_row_col = bottom_offset_num_row + col_in_input_data * input_Channel_;
										top_offset_num_channel_row_col = top_offset_num_channel_row + col_in_output_data * output_Channel_;
										V_offset_row_col = V_offset_row + num_w * tile_length_;

#if SIMD_TYPE >= SIMDTYPE_AVX
										sum_1 = mm_setzero_ps();
										sum_2 = mm_setzero_ps();
#elif SIMD_TYPE >= SIMDTYPE_SSE
										sum_1 = mm_setzero_ps();
										sum_2 = mm_setzero_ps();
										sum_3 = mm_setzero_ps();
										sum_4 = mm_setzero_ps();
#else
										memset(m_data, 0, tile_length_ * sizeof(float));
#endif // SIMD_TYPE >= SIMDTYPE_AVX

										for (ich = 0; ich < input_Channel_; ich++)
										{
											V_offset_channel_row_col = ich * h_w_tile_stride + V_offset_row_col;
											U_offset_och_ich = U_offset_och + ich * tile_length_;
											bottom_offset_num_channel_row_col = bottom_offset_num_row_col + ich;

											//calculate V when is_first==true
											if (!is_V_calculated)
											{
												for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
												{
													tile_offset_row = row_in_tile * tile_size_;
													real_row = row_in_input_data + row_in_tile;
													if (!is_a_ge_zero_and_a_lt_b(real_row, input_dim_h_))
													{
														memset(tile_data + tile_offset_row, 0, tile_size_ * sizeof(float));
													}
													else
													{
														for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
														{
															real_col = col_in_input_data + col_in_tile;
															if (!is_a_ge_zero_and_a_lt_b(real_col, input_dim_w_))
															{
																tile_data[tile_offset_row + col_in_tile] = 0;
															}
															else
															{
																tile_data[tile_offset_row + col_in_tile] = bottom_data[bottom_offset_num_channel_row_col + col_in_tile * input_Channel_];
															}
														}
													}
													bottom_offset_num_channel_row_col += input_w_stride;
												}

												calculate_BTdB(tile_data, V_data + V_offset_channel_row_col);//calculate v
											}

#if SIMD_TYPE >= SIMDTYPE_AVX
											u_1 = mm_load_ps(U_data + U_offset_och_ich);
											u_2 = mm_load_ps(U_data + U_offset_och_ich + 8);
											v_1 = mm_load_ps(V_data + V_offset_channel_row_col);
											v_2 = mm_load_ps(V_data + V_offset_channel_row_col + 8);
											sum_1 = mm_fmadd_ps(v_1, u_1, sum_1);
											sum_2 = mm_fmadd_ps(v_2, u_2, sum_2);
#elif SIMD_TYPE >= SIMDTYPE_SSE
											u_1 = mm_load_ps(U_data + U_offset_och_ich);
											u_2 = mm_load_ps(U_data + U_offset_och_ich + 4);
											u_3 = mm_load_ps(U_data + U_offset_och_ich + 8);
											u_4 = mm_load_ps(U_data + U_offset_och_ich + 12);
											v_1 = mm_load_ps(V_data + V_offset_channel_row_col);
											v_2 = mm_load_ps(V_data + V_offset_channel_row_col + 4);
											v_3 = mm_load_ps(V_data + V_offset_channel_row_col + 8);
											v_4 = mm_load_ps(V_data + V_offset_channel_row_col + 12);
											sum_1 = mm_fmadd_ps(v_1, u_1, sum_1);
											sum_2 = mm_fmadd_ps(v_2, u_2, sum_2);
											sum_3 = mm_fmadd_ps(v_3, u_3, sum_3);
											sum_4 = mm_fmadd_ps(v_4, u_4, sum_4);
#else
											for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
											{
												tile_offset_row = row_in_tile * tile_size_;
												for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
												{
													tile_offset_row_col = tile_offset_row + col_in_tile;
													m_data[tile_offset_row_col] += U_data[U_offset_och_ich + tile_offset_row_col] * V_data[V_offset_channel_row_col + tile_offset_row_col];
												}
											}
#endif
										}

#if SIMD_TYPE >= SIMDTYPE_AVX
										mm_store_ps(m_data, sum_1);
										mm_store_ps(m_data + 8, sum_2);
#elif SIMD_TYPE >= SIMDTYPE_SSE
										mm_store_ps(m_data, sum_1);
										mm_store_ps(m_data + 4, sum_2);
										mm_store_ps(m_data + 8, sum_3);
										mm_store_ps(m_data + 12, sum_4);
#endif // SIMD_TYPE >= SIMDTYPE_AVX

										calculate_ATmA(m_data, result);//calculate result

										if (num_h == h_tile_num_ - 1)
										{
											for (row = 0; row < m_ - add_h; row++)
											{
												result_offset_row = row * m_;
												if (num_w == w_tile_num_ - 1)
												{
													for (col = 0; col < m_ - add_w; col++)
													{
														top_data[top_offset_num_channel_row_col + col * output_Channel_] = result[result_offset_row + col];
													}
												}
												else
												{
													for (col = 0; col < m_; col++)
													{
														top_data[top_offset_num_channel_row_col + col * output_Channel_] = result[result_offset_row + col];
													}
												}
												top_offset_num_channel_row_col += output_w_stride;
											}
										}
										else
										{
											for (row = 0; row < m_; row++)
											{
												result_offset_row = row * m_;
												if (num_w == w_tile_num_ - 1)
												{
													for (col = 0; col < m_ - add_w; col++)
													{
														top_data[top_offset_num_channel_row_col + col * output_Channel_] = result[result_offset_row + col];
													}
												}
												else
												{
													for (col = 0; col < m_; col++)
													{
														top_data[top_offset_num_channel_row_col + col * output_Channel_] = result[result_offset_row + col];
													}
												}
												top_offset_num_channel_row_col += output_w_stride;
											}
										}
									}
								}

								is_V_calculated = true;
							}

							math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_data);
							if (bias_term_)
							{
								forward_bias(top_data + top_offset_num, bias_data);
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
		}

	}
}