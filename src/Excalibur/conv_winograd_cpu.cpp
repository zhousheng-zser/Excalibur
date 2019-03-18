#include "conv_winograd_cpu.hpp"
#include "../../include/Julius/simd_helper.hpp"

namespace glasssix
{
	namespace excalibur
	{
		conv_winograd_cpu::conv_winograd_cpu(int input_Channel, int output_Channel, int group, int kernelSize, int stride, int pad, bool bias_term, int device)
			: baseconv(input_Channel, output_Channel, group, kernelSize, stride, pad, bias_term, device)
		{
			tile_size_ = m_ + kernelSize_ - 1;//m+r-1
			tile_length_ = tile_size_ * tile_size_;
			kernel_length_ = kernelSize_ * kernelSize_;
			m_length_ = m_ * m_;
			U_num_ = output_Channel_ * input_Channel_ / group_;
			//U=G*g*GT,so U has the same number as kernel g, there are tile_size_ * tile_size_ elements in single U
			U_ = (float*)_aligned_malloc(U_num_ * tile_length_ * sizeof(float), mm_align_size * sizeof(float));
		}

		conv_winograd_cpu::~conv_winograd_cpu()
		{
			_aligned_free(U_);
		}

		void conv_winograd_cpu::forward_gemm(const float* input, const float* weights, float* output, bool skip_im2col) {}

		void conv_winograd_cpu::forward_bias(float* output, const float* bias)
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

#if SIMD_TYPE >= SIMDTYPE_AVX
		void conv_winograd_cpu::Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			CHECK_EQ(kernelSize_, 3);
			CHECK_EQ(stride_, 1);

			num_ = bottom->data_shape()[0];
			const float* bottom_data = bottom->cpu_data();
			const float* weights = weights_->cpu_data();
			const float* bias = bias_->cpu_data();
			order_ = bottom->order();

			intput_shape_.clear();
			intput_shape_ = bottom->data_shape();

			if (order_ == NCHW)
			{
				input_dim_h_ = bottom->data_shape()[2];
				input_dim_w_ = bottom->data_shape()[3];
				input_spatial_dim_ = input_dim_h_ * input_dim_w_;
				bottom_dim_ = bottom->count(1, 4);

				output_dim_h_ = (input_dim_h_ + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_dim_w_ = (input_dim_w_ + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_spatial_dim_ = output_dim_w_*output_dim_h_;
				bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_spatial_dim_}, device_));
				top.reset(new tensor<float>(std::vector<int>{num_, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
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

#pragma omp parallel for
						for (int och = 0; och < output_Channel_; och++)
						{
							float *tile_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
							float *v_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
							float *m_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
							float *result = (float*)_aligned_malloc(m_length_ * sizeof(float), mm_align_size * sizeof(float));

							int U_och_offset = och * tile_length_;
							int top_offset_num_channel = top_offset_num + och * output_spatial_dim_;
							int bottom_offset_num_channel = bottom_offset_num + och * input_spatial_dim_;

							if (!is_U_calculated)
							{
								calculate_GgGT(weights + kernel_length_ * och, U_ + U_och_offset);//calculate U
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
							mm_type sum_1;
							mm_type sum_2;
							mm_type u_1;
							mm_type u_2;
							mm_type v_1;
							mm_type v_2;

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

									u_1 = mm_load_ps(U_ + U_och_offset);
									u_2 = mm_load_ps(U_ + U_och_offset + 8);
									v_1 = mm_load_ps(v_data);
									v_2 = mm_load_ps(v_data + 8);
									sum_1 = mm_mul_ps(u_1, v_1);
									sum_2 = mm_mul_ps(u_2, v_2);
									mm_store_ps(m_data, sum_1);
									mm_store_ps(m_data + 8, sum_2);

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

							_aligned_free(tile_data);
							_aligned_free(v_data);
							_aligned_free(m_data);
							_aligned_free(result);
						}

						math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_->mutable_cpu_data());
						if (bias_term_)
						{
							forward_bias(top_data + top_offset_num, bias);
						}

						is_U_calculated = true;
					}
				}
				else if (group_ == 1)
				{
					//calculate U_
#pragma omp parallel for
					for (int n = 0; n < U_num_; ++n)
					{
						calculate_GgGT(weights + kernel_length_ * n, U_ + tile_length_ * n);//calculate U
					}

					//V=BT*d*B,so V has the same number as data tile, there are tile_length_ elements in single V
					V_num_ = input_Channel_ * total_tile_num;
					V_ = (float*)_aligned_malloc(V_num_ * tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					int U_offset_single_och = input_Channel_ * tile_length_;

					for (int n = 0; n < num_; n++)
					{
						int bottom_offset_num = n * bottom_dim_;
						int top_offset_num = n * top_dim_;
						bool is_V_calculated = false;//only calculate V when is_V_calculated==false

#pragma omp parallel for
						for (int och = 0; och < output_Channel_; och++)
						{
							float *tile_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
							float *m_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
							float *result = (float*)_aligned_malloc(m_length_ * sizeof(float), mm_align_size * sizeof(float));

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
							int real_row;
							int col_in_tile;
							int real_col;
							int row;
							int col;
							int result_offset_row;
							mm_type sum_1;
							mm_type sum_2;
							mm_type u_1;
							mm_type u_2;
							mm_type v_1;
							mm_type v_2;

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
									sum_1 = mm_setzero_ps();
									sum_2 = mm_setzero_ps();

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

											calculate_BTdB(tile_data, V_ + V_offset_channel_row_col);
										}

										u_1 = mm_load_ps(U_ + U_offset_och_ich);
										u_2 = mm_load_ps(U_ + U_offset_och_ich + 8);
										v_1 = mm_load_ps(V_ + V_offset_channel_row_col);
										v_2 = mm_load_ps(V_ + V_offset_channel_row_col + 8);
										sum_1 = mm_fmadd_ps(v_1, u_1, sum_1);
										sum_2 = mm_fmadd_ps(v_2, u_2, sum_2);
									}

									mm_store_ps(m_data, sum_1);
									mm_store_ps(m_data + 8, sum_2);
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
							_aligned_free(tile_data);
							_aligned_free(m_data);
							_aligned_free(result);
						}

						math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_->mutable_cpu_data());
						if (bias_term_)
						{
							forward_bias(top_data + top_offset_num, bias);
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

#pragma omp parallel for
						for (int och = 0; och < output_Channel_; och++)
						{
							float *tile_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
							float *v_data = (float*)malloc(tile_length_ * sizeof(float));
							float *m_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
							float *result = (float*)_aligned_malloc(m_length_ * sizeof(float), mm_align_size * sizeof(float));

							int bottom_offset_num_channel = bottom_offset_num + och;
							int top_offset_num_channel = top_offset_num + och;
							int U_och_offset = och * tile_length_;

							if (!is_U_calculated)
							{
								calculate_GgGT(weights + kernel_length_ * och, U_ + U_och_offset);//calculate U
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
							mm_type u_1;
							mm_type u_2;
							mm_type v_1;
							mm_type v_2;
							mm_type sum_1;
							mm_type sum_2;

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

									u_1 = mm_load_ps(U_ + U_och_offset);
									u_2 = mm_load_ps(U_ + U_och_offset + 8);
									v_1 = mm_load_ps(v_data);
									v_2 = mm_load_ps(v_data + 8);
									sum_1 = mm_mul_ps(u_1, v_1);
									sum_2 = mm_mul_ps(u_2, v_2);
									mm_store_ps(m_data, sum_1);
									mm_store_ps(m_data + 8, sum_2);
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

						math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_->mutable_cpu_data());
						if (bias_term_)
						{
							forward_bias(top_data + top_offset_num, bias);
						}

						is_U_calculated = true;
					}
				}
				else if (group_ == 1)
				{
					//calculate U_
#pragma omp parallel for
					for (int n = 0; n < U_num_; ++n)
					{
						calculate_GgGT(weights + kernel_length_ * n, U_ + tile_length_ * n);//calculate U
					}

					//V=BT*d*B,so V has the same number as data tile, there are tile_length_ elements in single V
					V_num_ = input_Channel_ * total_tile_num;
					V_ = (float*)_aligned_malloc(V_num_ * tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					int U_offset_single_och = input_Channel_ * tile_length_;

					for (int n = 0; n < num_; n++)
					{
						int bottom_offset_num = n * bottom_dim_;
						int top_offset_num = n * top_dim_;
						bool is_V_calculated = false;//only calculate V when is_V_calculated==false

#pragma omp parallel for
						for (int och = 0; och < output_Channel_; och++)
						{
							float *tile_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
							float *m_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
							float *result = (float*)_aligned_malloc(m_length_ * sizeof(float), mm_align_size * sizeof(float));

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
							mm_type sum_1;
							mm_type sum_2;
							mm_type u_1;
							mm_type u_2;
							mm_type v_1;
							mm_type v_2;

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

									sum_1 = mm_setzero_ps();
									sum_2 = mm_setzero_ps();

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

											calculate_BTdB(tile_data, V_ + V_offset_channel_row_col);//calculate v
										}

										u_1 = mm_load_ps(U_ + U_offset_och_ich);
										u_2 = mm_load_ps(U_ + U_offset_och_ich + 8);
										v_1 = mm_load_ps(V_ + V_offset_channel_row_col);
										v_2 = mm_load_ps(V_ + V_offset_channel_row_col + 8);
										sum_1 = mm_fmadd_ps(v_1, u_1, sum_1);
										sum_2 = mm_fmadd_ps(v_2, u_2, sum_2);
									}

									mm_store_ps(m_data, sum_1);
									mm_store_ps(m_data + 8, sum_2);

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
							_aligned_free(tile_data);
							_aligned_free(m_data);
							_aligned_free(result);
						}

						math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_->mutable_cpu_data());
						if (bias_term_)
						{
							forward_bias(top_data + top_offset_num, bias);
						}
					}

					_aligned_free(V_);
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
#elif SIMD_TYPE >= SIMDTYPE_SSE
void conv_winograd_cpu::Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
{
	CHECK_EQ(kernelSize_, 3);
	CHECK_EQ(stride_, 1);

	num_ = bottom->data_shape()[0];
	const float* bottom_data = bottom->cpu_data();
	const float* weights = weights_->cpu_data();
	const float* bias = bias_->cpu_data();
	order_ = bottom->order();

	intput_shape_.clear();
	intput_shape_ = bottom->data_shape();

	if (order_ == NCHW)
	{
		input_dim_h_ = bottom->data_shape()[2];
		input_dim_w_ = bottom->data_shape()[3];
		input_spatial_dim_ = input_dim_h_ * input_dim_w_;
		bottom_dim_ = bottom->count(1, 4);

		output_dim_h_ = (input_dim_h_ + 2 * pad_ - kernelSize_) / stride_ + 1;
		output_dim_w_ = (input_dim_w_ + 2 * pad_ - kernelSize_) / stride_ + 1;
		output_spatial_dim_ = output_dim_w_*output_dim_h_;
		bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_spatial_dim_}, device_));
		top.reset(new tensor<float>(std::vector<int>{num_, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
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

#pragma omp parallel for
				for (int och = 0; och < output_Channel_; och++)
				{
					float *tile_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					float *v_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					float *m_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					float *result = (float*)_aligned_malloc(m_length_ * sizeof(float), mm_align_size * sizeof(float));

					int U_och_offset = och * tile_length_;
					int top_offset_num_channel = top_offset_num + och * output_spatial_dim_;
					int bottom_offset_num_channel = bottom_offset_num + och * input_spatial_dim_;

					if (!is_U_calculated)
					{
						calculate_GgGT(weights + kernel_length_ * och, U_ + U_och_offset);//calculate U
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

							u_1 = mm_load_ps(U_ + U_och_offset);
							u_2 = mm_load_ps(U_ + U_och_offset + 4);
							u_3 = mm_load_ps(U_ + U_och_offset + 8);
							u_4 = mm_load_ps(U_ + U_och_offset + 12);
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

					_aligned_free(tile_data);
					_aligned_free(v_data);
					_aligned_free(m_data);
					_aligned_free(result);
				}

				math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_->mutable_cpu_data());
				if (bias_term_)
				{
					forward_bias(top_data + top_offset_num, bias);
				}

				is_U_calculated = true;
			}
		}
		else if (group_ == 1)
		{
			//calculate U_
#pragma omp parallel for
			for (int n = 0; n < U_num_; ++n)
			{
				calculate_GgGT(weights + kernel_length_ * n, U_ + tile_length_ * n);//calculate U
			}

			//V=BT*d*B,so V has the same number as data tile, there are tile_length_ elements in single V
			V_num_ = input_Channel_ * total_tile_num;
			V_ = (float*)_aligned_malloc(V_num_ * tile_length_ * sizeof(float), mm_align_size * sizeof(float));
			int U_offset_single_och = input_Channel_ * tile_length_;

			for (int n = 0; n < num_; n++)
			{
				int bottom_offset_num = n * bottom_dim_;
				int top_offset_num = n * top_dim_;
				bool is_V_calculated = false;//only calculate V when is_V_calculated==false

#pragma omp parallel for
				for (int och = 0; och < output_Channel_; och++)
				{
					float *tile_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					float *m_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					float *result = (float*)_aligned_malloc(m_length_ * sizeof(float), mm_align_size * sizeof(float));

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
					int real_row;
					int col_in_tile;
					int real_col;
					int row;
					int col;
					int result_offset_row;
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
							sum_1 = mm_setzero_ps();
							sum_2 = mm_setzero_ps();
							sum_3 = mm_setzero_ps();
							sum_4 = mm_setzero_ps();

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

									calculate_BTdB(tile_data, V_ + V_offset_channel_row_col);
								}

								u_1 = mm_load_ps(U_ + U_offset_och_ich);
								u_2 = mm_load_ps(U_ + U_offset_och_ich + 4);
								u_3 = mm_load_ps(U_ + U_offset_och_ich + 8);
								u_4 = mm_load_ps(U_ + U_offset_och_ich + 12);
								v_1 = mm_load_ps(V_ + V_offset_channel_row_col);
								v_2 = mm_load_ps(V_ + V_offset_channel_row_col + 4);
								v_3 = mm_load_ps(V_ + V_offset_channel_row_col + 8);
								v_4 = mm_load_ps(V_ + V_offset_channel_row_col + 12);
								sum_1 = mm_fmadd_ps(v_1, u_1, sum_1);
								sum_2 = mm_fmadd_ps(v_2, u_2, sum_2);
								sum_3 = mm_fmadd_ps(v_3, u_3, sum_3);
								sum_4 = mm_fmadd_ps(v_4, u_4, sum_4);
							}

							mm_store_ps(m_data, sum_1);
							mm_store_ps(m_data + 4, sum_2);
							mm_store_ps(m_data + 8, sum_3);
							mm_store_ps(m_data + 12, sum_4);
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
					_aligned_free(tile_data);
					_aligned_free(m_data);
					_aligned_free(result);
				}

				math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_->mutable_cpu_data());
				if (bias_term_)
				{
					forward_bias(top_data + top_offset_num, bias);
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

#pragma omp parallel for
				for (int och = 0; och < output_Channel_; och++)
				{
					float *tile_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					float *v_data = (float*)malloc(tile_length_ * sizeof(float));
					float *m_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					float *result = (float*)_aligned_malloc(m_length_ * sizeof(float), mm_align_size * sizeof(float));

					int bottom_offset_num_channel = bottom_offset_num + och;
					int top_offset_num_channel = top_offset_num + och;
					int U_och_offset = och * tile_length_;

					if (!is_U_calculated)
					{
						calculate_GgGT(weights + kernel_length_ * och, U_ + U_och_offset);//calculate U
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

							u_1 = mm_load_ps(U_ + U_och_offset);
							u_2 = mm_load_ps(U_ + U_och_offset + 4);
							u_3 = mm_load_ps(U_ + U_och_offset + 8);
							u_4 = mm_load_ps(U_ + U_och_offset + 12);
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

				math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_->mutable_cpu_data());
				if (bias_term_)
				{
					forward_bias(top_data + top_offset_num, bias);
				}

				is_U_calculated = true;
			}
		}
		else if (group_ == 1)
		{
			//calculate U_
#pragma omp parallel for
			for (int n = 0; n < U_num_; ++n)
			{
				calculate_GgGT(weights + kernel_length_ * n, U_ + tile_length_ * n);//calculate U
			}

			//V=BT*d*B,so V has the same number as data tile, there are tile_length_ elements in single V
			V_num_ = input_Channel_ * total_tile_num;
			V_ = (float*)_aligned_malloc(V_num_ * tile_length_ * sizeof(float), mm_align_size * sizeof(float));
			int U_offset_single_och = input_Channel_ * tile_length_;

			for (int n = 0; n < num_; n++)
			{
				int bottom_offset_num = n * bottom_dim_;
				int top_offset_num = n * top_dim_;
				bool is_V_calculated = false;//only calculate V when is_V_calculated==false

#pragma omp parallel for
				for (int och = 0; och < output_Channel_; och++)
				{
					float *tile_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					float *m_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					float *result = (float*)_aligned_malloc(m_length_ * sizeof(float), mm_align_size * sizeof(float));

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

							sum_1 = mm_setzero_ps();
							sum_2 = mm_setzero_ps();
							sum_3 = mm_setzero_ps();
							sum_4 = mm_setzero_ps();

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

									calculate_BTdB(tile_data, V_ + V_offset_channel_row_col);//calculate v
								}

								u_1 = mm_load_ps(U_ + U_offset_och_ich);
								u_2 = mm_load_ps(U_ + U_offset_och_ich + 4);
								u_3 = mm_load_ps(U_ + U_offset_och_ich + 8);
								u_4 = mm_load_ps(U_ + U_offset_och_ich + 12);
								v_1 = mm_load_ps(V_ + V_offset_channel_row_col);
								v_2 = mm_load_ps(V_ + V_offset_channel_row_col + 4);
								v_3 = mm_load_ps(V_ + V_offset_channel_row_col + 8);
								v_4 = mm_load_ps(V_ + V_offset_channel_row_col + 12);
								sum_1 = mm_fmadd_ps(v_1, u_1, sum_1);
								sum_2 = mm_fmadd_ps(v_2, u_2, sum_2);
								sum_3 = mm_fmadd_ps(v_3, u_3, sum_3);
								sum_4 = mm_fmadd_ps(v_4, u_4, sum_4);
							}

							mm_store_ps(m_data, sum_1);
							mm_store_ps(m_data + 4, sum_2);
							mm_store_ps(m_data + 8, sum_3);
							mm_store_ps(m_data + 12, sum_4);

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
					_aligned_free(tile_data);
					_aligned_free(m_data);
					_aligned_free(result);
				}

				math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_->mutable_cpu_data());
				if (bias_term_)
				{
					forward_bias(top_data + top_offset_num, bias);
				}
			}

			_aligned_free(V_);
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
#elif SIMD_TYPE == SIMDTYPE_NONE
void conv_winograd_cpu::Forward(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
{
	CHECK_EQ(kernelSize_, 3);
	CHECK_EQ(stride_, 1);

	num_ = bottom->data_shape()[0];
	const float* bottom_data = bottom->cpu_data();
	const float* weights = weights_->cpu_data();
	const float* bias = bias_->cpu_data();
	order_ = bottom->order();

	intput_shape_.clear();
	intput_shape_ = bottom->data_shape();

	if (order_ == NCHW)
	{
		input_dim_h_ = bottom->data_shape()[2];
		input_dim_w_ = bottom->data_shape()[3];
		input_spatial_dim_ = input_dim_h_ * input_dim_w_;
		bottom_dim_ = bottom->count(1, 4);

		output_dim_h_ = (input_dim_h_ + 2 * pad_ - kernelSize_) / stride_ + 1;
		output_dim_w_ = (input_dim_w_ + 2 * pad_ - kernelSize_) / stride_ + 1;
		output_spatial_dim_ = output_dim_w_*output_dim_h_;
		bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_spatial_dim_}, device_));
		top.reset(new tensor<float>(std::vector<int>{num_, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
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

#pragma omp parallel for
				for (int och = 0; och < output_Channel_; och++)
				{
					float *tile_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					float *v_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					float *m_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					float *result = (float*)_aligned_malloc(m_length_ * sizeof(float), mm_align_size * sizeof(float));

					int U_offset_och = och * tile_length_;
					int top_offset_num_channel = top_offset_num + och * output_spatial_dim_;
					int bottom_offset_num_channel = bottom_offset_num + och * input_spatial_dim_;

					if (!is_U_calculated)
					{
						calculate_GgGT(weights + kernel_length_ * och, U_ + U_offset_och);//calculate U
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
							memset(m_data, 0, tile_length_ * sizeof(float));

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

							for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
							{
								tile_offset_row = row_in_tile * tile_size_;
								for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
								{
									tile_offset_row_col = tile_offset_row + col_in_tile;
									m_data[tile_offset_row_col] += U_[U_offset_och + tile_offset_row_col] * v_data[tile_offset_row_col];
								}
							}

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

					_aligned_free(tile_data);
					_aligned_free(v_data);
					_aligned_free(m_data);
					_aligned_free(result);
				}

				math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_->mutable_cpu_data());
				if (bias_term_)
				{
					forward_bias(top_data + top_offset_num, bias);
				}

				is_U_calculated = true;
			}
		}
		else if (group_ == 1)
		{
			//calculate U_
#pragma omp parallel for
			for (int n = 0; n < U_num_; ++n)
			{
				calculate_GgGT(weights + kernel_length_ * n, U_ + tile_length_ * n);//calculate U
			}

			//V=BT*d*B,so V has the same number as data tile, there are tile_length_ elements in single V
			V_num_ = input_Channel_ * total_tile_num;
			V_ = (float*)_aligned_malloc(V_num_ * tile_length_ * sizeof(float), mm_align_size * sizeof(float));
			int U_offset_single_och = input_Channel_ * tile_length_;

			for (int n = 0; n < num_; n++)
			{
				int bottom_offset_num = n * bottom_dim_;
				int top_offset_num = n * top_dim_;
				bool is_V_calculated = false;//only calculate V when is_V_calculated==false

#pragma omp parallel for
				for (int och = 0; och < output_Channel_; och++)
				{
					float *tile_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					float *m_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					float *result = (float*)_aligned_malloc(m_length_ * sizeof(float), mm_align_size * sizeof(float));

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
					int real_row;
					int col_in_tile;
					int real_col;
					int row;
					int col;
					int result_offset_row;
					int tile_offset_row_col;

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
							memset(m_data, 0, tile_length_ * sizeof(float));

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

									calculate_BTdB(tile_data, V_ + V_offset_channel_row_col);
								}

								for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
								{
									tile_offset_row = row_in_tile * tile_size_;
									for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
									{
										tile_offset_row_col = tile_offset_row + col_in_tile;
										m_data[tile_offset_row_col] += U_[U_offset_och_ich + tile_offset_row_col] * V_[V_offset_channel_row_col + tile_offset_row_col];
									}
								}
							}

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
					_aligned_free(tile_data);
					_aligned_free(m_data);
					_aligned_free(result);
				}

				math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_->mutable_cpu_data());
				if (bias_term_)
				{
					forward_bias(top_data + top_offset_num, bias);
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

#pragma omp parallel for
				for (int och = 0; och < output_Channel_; och++)
				{
					float *tile_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					float *v_data = (float*)malloc(tile_length_ * sizeof(float));
					float *m_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					float *result = (float*)_aligned_malloc(m_length_ * sizeof(float), mm_align_size * sizeof(float));

					int bottom_offset_num_channel = bottom_offset_num + och;
					int top_offset_num_channel = top_offset_num + och;
					int U_offset_och = och * tile_length_;

					if (!is_U_calculated)
					{
						calculate_GgGT(weights + kernel_length_ * och, U_ + U_offset_och);//calculate U
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
							memset(m_data, 0, tile_length_ * sizeof(float));

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

							for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
							{
								tile_offset_row = row_in_tile * tile_size_;
								for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
								{
									tile_offset_row_col = tile_offset_row + col_in_tile;
									m_data[tile_offset_row_col] += U_[U_offset_och + tile_offset_row_col] * v_data[tile_offset_row_col];
								}
							}

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

				math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_->mutable_cpu_data());
				if (bias_term_)
				{
					forward_bias(top_data + top_offset_num, bias);
				}

				is_U_calculated = true;
			}
		}
		else if (group_ == 1)
		{
			//calculate U_
#pragma omp parallel for
			for (int n = 0; n < U_num_; ++n)
			{
				calculate_GgGT(weights + kernel_length_ * n, U_ + tile_length_ * n);//calculate U
			}

			//V=BT*d*B,so V has the same number as data tile, there are tile_length_ elements in single V
			V_num_ = input_Channel_ * total_tile_num;
			V_ = (float*)_aligned_malloc(V_num_ * tile_length_ * sizeof(float), mm_align_size * sizeof(float));
			int U_offset_single_och = input_Channel_ * tile_length_;

			for (int n = 0; n < num_; n++)
			{
				int bottom_offset_num = n * bottom_dim_;
				int top_offset_num = n * top_dim_;
				bool is_V_calculated = false;//only calculate V when is_V_calculated==false

#pragma omp parallel for
				for (int och = 0; och < output_Channel_; och++)
				{
					float *tile_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					float *m_data = (float*)_aligned_malloc(tile_length_ * sizeof(float), mm_align_size * sizeof(float));
					float *result = (float*)_aligned_malloc(m_length_ * sizeof(float), mm_align_size * sizeof(float));

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

							memset(m_data, 0, tile_length_ * sizeof(float));
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

									calculate_BTdB(tile_data, V_ + V_offset_channel_row_col);//calculate v
								}


								for (row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
								{
									tile_offset_row = row_in_tile * tile_size_;
									for (col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
									{
										tile_offset_row_col = tile_offset_row + col_in_tile;
										m_data[tile_offset_row_col] += U_[U_offset_och_ich + tile_offset_row_col] * V_[V_offset_channel_row_col + tile_offset_row_col];
									}
								}
							}

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
					_aligned_free(tile_data);
					_aligned_free(m_data);
					_aligned_free(result);
				}

				math_functions::cpu_set(output_spatial_dim_, 1.0f, bias_multiplier_->mutable_cpu_data());
				if (bias_term_)
				{
					forward_bias(top_data + top_offset_num, bias);
				}
			}

			_aligned_free(V_);
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
#else
		NOT_IMPLEMENTED;
#endif//
		
	}
}