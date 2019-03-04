#include "conv_winograd_cpu.hpp"
namespace glasssix
{
	namespace excalibur
	{
		conv_winograd_cpu::conv_winograd_cpu(int input_Channel, int output_Channel, int kernelSize, int stride, int pad, bool bias_term, int device)
			: baseconv(input_Channel, output_Channel, kernelSize, stride, pad, bias_term, device)
		{
			tile_size_ = m_ + kernelSize_ - 1;//m+r-1
			tile_length_ = tile_size_ * tile_size_;
			U_num_ = output_Channel_ * input_Channel_ / group_;
			//U=G*g*GT,so U has the same number as kernel g, there are tile_size_ * tile_size_ elements in single U
			U_ = (float*)malloc(U_num_ * tile_length_ * sizeof(float));
			tile_data = (float*)malloc(tile_length_ * sizeof(float));
			v_data = (float*)malloc(tile_length_ * sizeof(float));
			m_data = (float*)malloc(tile_length_ * sizeof(float));
			result = (float*)malloc(m_ * m_ * sizeof(float));
		}

		conv_winograd_cpu::conv_winograd_cpu(int input_Channel, int output_Channel, int kernelSize, int group, int stride, int pad, bool bias_term, int device)
			: baseconv(input_Channel, output_Channel, kernelSize, group, stride, pad, bias_term, device)
		{
			tile_size_ = m_ + kernelSize_ - 1;//m+r-1
			tile_length_ = tile_size_ * tile_size_;
			U_num_ = output_Channel_ * input_Channel_ / group_;
			//U=G*g*GT,so U has the same number as kernel g, there are tile_size_ * tile_size_ elements in single U
			U_ = (float*)malloc(U_num_ * tile_length_ * sizeof(float));
			tile_data = (float*)malloc(tile_length_ * sizeof(float));
			v_data = (float*)malloc(tile_length_ * sizeof(float));
			m_data = (float*)malloc(tile_length_ * sizeof(float));
			result = (float*)malloc(m_ * m_ * sizeof(float));
		}

		conv_winograd_cpu::~conv_winograd_cpu()
		{
			delete U_;
			delete V_;
			delete tile_data;
			delete v_data;
			delete m_data;
			delete result;
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


			//calculate U_
#pragma omp parallel for
			for (int n = 0; n < U_num_; ++n)
			{
				calculate_GgGT(weights + kernelSize_ * kernelSize_ * n, U_ + tile_size_ * tile_size_ * n);//calculate U
			}

			if (order_ == NCHW)
			{
				input_dim_h_ = bottom->data_shape()[2];
				input_dim_w_ = bottom->data_shape()[3];
				input_spatial_dim_ = input_dim_h_ * input_dim_w_;
				bottom_dim_ = bottom->count(1, 4);

				output_dim_h_ = (input_dim_h_ + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_dim_w_ = (input_dim_w_ + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_spatial_dim_ = output_dim_w_*output_dim_h_;
				top_dim_ = (top)->count(1, 4);
				top.reset(new tensor<float>(std::vector<int>{num_, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
				float* top_data = top->mutable_cpu_data();
								
				int h_subtract_tilesize = input_dim_h_ + 2 * pad_ - tile_size_;
				int w_subtract_tilesize = input_dim_w_ + 2 * pad_ - tile_size_;
				h_tile_num_ = int(h_subtract_tilesize / m_ + 0.5f) + 1;//h_tile_num_ = ceil((H-(m+r-1))/m) + 1, H is height after padding
				w_tile_num_ = int(w_subtract_tilesize / m_ + 0.5f) + 1;//w_tile_num_ = ceil((W-(m+r-1))/m) + 1, W is width after padding
				int h_aligned = (h_subtract_tilesize + m_ - 1) / m_ * m_;
				int w_aligned = (w_subtract_tilesize + m_ - 1) / m_ * m_;
				int add_h = h_aligned - h_subtract_tilesize;
				int add_w = w_aligned - w_subtract_tilesize;

				if (group_ > 1)
				{
					for (int n = 0; n < num_; n++)
					{
						int num_offset_bottom = n * bottom_dim_;
						int num_offset_top = n * top_dim_;

						for (int och = 0; och < output_Channel_; och++)
						{
							int U_och_offset = och * tile_length_;
							int channel_offset_top = och * output_spatial_dim_;
							for (int num_h = 0; num_h < h_tile_num_; num_h++)
							{
								int row_in_output_data = num_h * m_;
								int row_in_input_data = row_in_output_data - pad_;
								for (int num_w = 0; num_w < w_tile_num_; num_w++)
								{
									int col_in_output_data = num_w * m_;
									int col_in_input_data = col_in_output_data - pad_;
									memset(m_data, 0, tile_length_ * sizeof(float));

									for (int row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
									{
										int row_offset = row_in_tile * tile_size_;
										int real_row = row_in_input_data + row_in_tile;
										if (!is_a_ge_zero_and_a_lt_b(real_row, input_dim_h_))
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
												if (!is_a_ge_zero_and_a_lt_b(real_col, input_dim_w_))
												{
													tile_data[row_offset + col_in_tile] = 0;
												}
												else
												{
													tile_data[row_offset + col_in_tile] = *(bottom_data + num_offset_bottom + och * input_spatial_dim_ + real_row * input_dim_w_ + real_col);
												}
											}
										}
									}

									calculate_BTdB(tile_data, v_data);//calculate V

									for (int row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
									{
										int row_offset = row_in_tile * tile_size_;
										for (int col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
										{
											int offset = row_offset + col_in_tile;
											m_data[offset] = v_data[offset] * *(U_ + U_och_offset + offset);
										}
									}

									calculate_ATmA(m_data, result);//calculate result

									if (num_h == h_tile_num_ - 1)
									{
										for (size_t row = 0; row < m_ - add_h; row++)
										{
											int row_offset = (row_in_output_data + row) * output_dim_w_;
											if (num_w == w_tile_num_ - 1)
											{
												for (size_t col = 0; col < m_ - add_w; col++)
												{
													*(top_data + num_offset_top + channel_offset_top + row_offset + col_in_output_data + col) = result[row * m_ + col];
												}
											}
											else
											{
												for (size_t col = 0; col < m_; col++)
												{
													*(top_data + num_offset_top + channel_offset_top + row_offset + col_in_output_data + col) = result[row * m_ + col];
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
													*(top_data + num_offset_top + channel_offset_top + row_offset + col_in_output_data + col) = result[row * m_ + col];
												}
											}
											else
											{
												for (int col = 0; col < m_; col++)
												{
													*(top_data + num_offset_top + channel_offset_top + row_offset + col_in_output_data + col) = result[row * m_ + col];
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
							forward_bias(top_data + n * top_dim_, bias);
						}
					}
				}
				else if (group_ == 1)
				{
					//V=BT*d*B,so V has the same number as data tile, there are tile_length_ elements in single V
					V_num_ = input_Channel_ * h_tile_num_ * w_tile_num_;
					V_ = (float*)malloc(V_num_ * tile_length_ * sizeof(float));
					for (int n = 0; n < num_; n++)
					{
						int num_offset_bottom = n * bottom_dim_;
						int num_offset_top = n * top_dim_;
						bool is_first = true;//only calculate V when is_first==true

											 //calculate final output
						for (int och = 0; och < output_Channel_; och++)
						{
							int U_och_offset = och * input_Channel_ * tile_length_;
							int channel_offset_top = och * output_spatial_dim_;
							for (int num_h = 0; num_h < h_tile_num_; num_h++)
							{
								int V_h_offset = num_h * w_tile_num_ * tile_length_;
								int row_in_output_data = num_h * m_;
								int row_in_input_data = row_in_output_data - pad_;
								for (int num_w = 0; num_w < w_tile_num_; num_w++)
								{
									int V_w_offset = num_w * tile_length_;
									int col_in_output_data = num_w * m_;
									int col_in_input_data = col_in_output_data - pad_;
									memset(m_data, 0, tile_length_ * sizeof(float));

									for (int ich = 0; ich < input_Channel_; ich++)
									{
										int V_ich_offset = ich * h_tile_num_ * w_tile_num_ * tile_length_;
										int U_ich_offset = ich * tile_length_;

										//calculate V when is_first==true
										if (is_first)
										{
											int V_sequence = (ich * h_tile_num_ + num_h) * w_tile_num_ + num_w;

											for (int row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
											{
												int real_row = row_in_input_data + row_in_tile;
												if (!is_a_ge_zero_and_a_lt_b(real_row, input_dim_h_))
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
														if (!is_a_ge_zero_and_a_lt_b(real_col, input_dim_w_))
														{
															tile_data[row_in_tile * tile_size_ + col_in_tile] = 0;
														}
														else
														{
															tile_data[row_in_tile * tile_size_ + col_in_tile] = *(bottom_data + num_offset_bottom + ich * input_spatial_dim_ + real_row * input_dim_w_ + real_col);
														}
													}
												}
											}

											calculate_BTdB(tile_data, V_ + tile_length_ * V_sequence);
										}

										for (int row = 0; row < tile_size_; row++)
										{
											int row_offset = row * tile_size_;
											for (int col = 0; col < tile_size_; col++)
											{
												m_data[row_offset + col] += *(V_ + V_ich_offset + V_h_offset + V_w_offset + row_offset + col) * *(U_ + U_och_offset + U_ich_offset + row_offset + col);
											}
										}
									}

									calculate_ATmA(m_data, result);

									if (num_h == h_tile_num_ - 1)
									{
										for (size_t row = 0; row < m_ - add_h; row++)
										{
											int row_offset = (row_in_output_data + row) * output_dim_w_;
											if (num_w == w_tile_num_ - 1)
											{
												for (size_t col = 0; col < m_ - add_w; col++)
												{
													*(top_data + num_offset_top + channel_offset_top + row_offset + col_in_output_data + col) = result[row * m_ + col];
												}
											}
											else
											{
												for (size_t col = 0; col < m_; col++)
												{
													*(top_data + num_offset_top + channel_offset_top + row_offset + col_in_output_data + col) = result[row * m_ + col];
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
													*(top_data + num_offset_top + channel_offset_top + row_offset + col_in_output_data + col) = result[row * m_ + col];
												}
											}
											else
											{
												for (int col = 0; col < m_; col++)
												{
													*(top_data + num_offset_top + channel_offset_top + row_offset + col_in_output_data + col) = result[row * m_ + col];
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
							forward_bias(top_data + n * top_dim_, bias);
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
				bottom_dim_ = bottom->count(1, 4);

				output_dim_h_ = (input_dim_h_ + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_dim_w_ = (input_dim_w_ + 2 * pad_ - kernelSize_) / stride_ + 1;
				output_spatial_dim_ = output_dim_w_*output_dim_h_;
				top_dim_ = (top)->count(1, 4);
				top.reset(new tensor<float>(std::vector<int>{num_, output_dim_h_, output_dim_w_, output_Channel_}, device_, order_));
				float* top_data = top->mutable_cpu_data();

				int h_subtract_tilesize = input_dim_h_ + 2 * pad_ - tile_size_;
				int w_subtract_tilesize = input_dim_w_ + 2 * pad_ - tile_size_;
				h_tile_num_ = int(h_subtract_tilesize / m_ + 0.5f) + 1;//h_tile_num_ = ceil((H-(m+r-1))/m) + 1, H is height after padding
				w_tile_num_ = int(w_subtract_tilesize / m_ + 0.5f) + 1;//w_tile_num_ = ceil((W-(m+r-1))/m) + 1, W is width after padding
				int h_aligned = (h_subtract_tilesize + m_ - 1) / m_ * m_;
				int w_aligned = (w_subtract_tilesize + m_ - 1) / m_ * m_;
				int add_h = h_aligned - h_subtract_tilesize;
				int add_w = w_aligned - w_subtract_tilesize;
				int tile_length_ = tile_size_ * tile_size_;

				if (group_ > 1)
				{
					for (int n = 0; n < num_; n++)
					{
						int num_offset_bottom = n * bottom_dim_;
						int num_offset_top = n * top_dim_;

						for (int och = 0; och < output_Channel_; och++)
						{
							int U_och_offset = och * tile_length_;
							for (int num_h = 0; num_h < h_tile_num_; num_h++)
							{
								int row_in_output_data = num_h * m_;
								int row_in_input_data = row_in_output_data - pad_;
								for (int num_w = 0; num_w < w_tile_num_; num_w++)
								{
									int col_in_output_data = num_w * m_;
									int col_in_input_data = col_in_output_data - pad_;
									memset(m_data, 0, tile_length_ * sizeof(float));

									for (int row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
									{
										int row_offset = row_in_tile * tile_size_;
										int real_row = row_in_input_data + row_in_tile;
										if (!is_a_ge_zero_and_a_lt_b(real_row, input_dim_h_))
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
												if (!is_a_ge_zero_and_a_lt_b(real_col, input_dim_w_))
												{
													tile_data[row_offset + col_in_tile] = 0;
												}
												else
												{
													//output_Channel_ and input_Channel_ has the same value, so we use output_Channel_ instead
													tile_data[row_offset + col_in_tile] = *(bottom_data + num_offset_bottom + (real_row * input_dim_w_ + real_col) * output_Channel_ + och);
												}
											}
										}
									}

									calculate_BTdB(tile_data, v_data);//calculate v

									for (int row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
									{
										int row_offset = row_in_tile * tile_size_;
										for (int col_in_tile = 0; col_in_tile < tile_size_; col_in_tile++)
										{
											int offset = row_offset + col_in_tile;
											m_data[offset] = v_data[offset] * *(U_ + U_och_offset + offset);
										}
									}

									calculate_ATmA(m_data, result);//calculate result

									if (num_h == h_tile_num_ - 1)
									{
										for (size_t row = 0; row < m_ - add_h; row++)
										{
											int row_offset = (row_in_output_data + row) * output_dim_w_ * output_Channel_;
											if (num_w == w_tile_num_ - 1)
											{
												for (size_t col = 0; col < m_ - add_w; col++)
												{
													*(top_data + num_offset_top + row_offset + (col_in_output_data + col) * output_Channel_ + och) = result[row * m_ + col];
												}
											}
											else
											{
												for (size_t col = 0; col < m_; col++)
												{
													*(top_data + num_offset_top + row_offset + (col_in_output_data + col) * output_Channel_ + och) = result[row * m_ + col];
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
													*(top_data + num_offset_top + row_offset + (col_in_output_data + col) * output_Channel_ + och) = result[row * m_ + col];
												}
											}
											else
											{
												for (int col = 0; col < m_; col++)
												{
													*(top_data + num_offset_top + row_offset + (col_in_output_data + col) * output_Channel_ + och) = result[row * m_ + col];
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
							forward_bias(top_data + n * top_dim_, bias);
						}
					}
				}
				else if (group_ == 1)
				{
					//V=BT*d*B,so V has the same number as data tile, there are tile_length_ elements in single V
					V_num_ = input_Channel_ * h_tile_num_ * w_tile_num_;
					V_ = (float*)malloc(V_num_ * tile_length_ * sizeof(float));

					for (int n = 0; n < num_; n++)
					{
						int num_offset_bottom = n * bottom_dim_;
						int num_offset_top = n * top_dim_;
						bool is_first = true;//only calculate V when is_first==true

						for (int och = 0; och < output_Channel_; och++)
						{
							int U_och_offset = och * input_Channel_ * tile_length_;
							for (int num_h = 0; num_h < h_tile_num_; num_h++)
							{
								int V_h_offset = num_h * w_tile_num_ * tile_length_;
								int row_in_output_data = num_h * m_;
								int row_in_input_data = row_in_output_data - pad_;
								for (int num_w = 0; num_w < w_tile_num_; num_w++)
								{
									int V_w_offset = num_w * tile_length_;
									int col_in_output_data = num_w * m_;
									int col_in_input_data = col_in_output_data - pad_;
									memset(m_data, 0, tile_length_ * sizeof(float));

									for (int ich = 0; ich < input_Channel_; ich++)
									{
										int V_ich_offset = ich * h_tile_num_ * w_tile_num_ * tile_length_;
										int U_ich_offset = ich * tile_length_;

										//calculate V when is_first==true
										if (is_first)
										{
											int V_sequence = (ich * h_tile_num_ + num_h) * w_tile_num_ + num_w;

											for (int row_in_tile = 0; row_in_tile < tile_size_; row_in_tile++)
											{
												int real_row = row_in_input_data + row_in_tile;
												if (!is_a_ge_zero_and_a_lt_b(real_row, input_dim_h_))
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
														if (!is_a_ge_zero_and_a_lt_b(real_col, input_dim_w_))
														{
															tile_data[row_in_tile * tile_size_ + col_in_tile] = 0;
														}
														else
														{
															tile_data[row_in_tile * tile_size_ + col_in_tile] = *(bottom_data + num_offset_bottom + (real_row * input_dim_w_ + real_col) * input_Channel_ + ich);
														}
													}
												}
											}

											calculate_BTdB(tile_data, V_ + tile_length_ * V_sequence);//calculate v
										}

										for (int row = 0; row < tile_size_; row++)
										{
											int row_offset = row * tile_size_;
											for (int col = 0; col < tile_size_; col++)
											{
												m_data[row_offset + col] += *(V_ + V_ich_offset + V_h_offset + V_w_offset + row_offset + col) * *(U_ + U_och_offset + U_ich_offset + row_offset + col);
											}
										}
									}

									calculate_ATmA(m_data, result);//calculate result

									if (num_h == h_tile_num_ - 1)
									{
										for (size_t row = 0; row < m_ - add_h; row++)
										{
											int row_offset = (row_in_output_data + row) * output_dim_w_ * output_Channel_;
											if (num_w == w_tile_num_ - 1)
											{
												for (size_t col = 0; col < m_ - add_w; col++)
												{
													*(top_data + num_offset_top + row_offset + (col_in_output_data + col) * output_Channel_ + och) = result[row * m_ + col];
												}
											}
											else
											{
												for (size_t col = 0; col < m_; col++)
												{
													*(top_data + num_offset_top + row_offset + (col_in_output_data + col) * output_Channel_ + och) = result[row * m_ + col];
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
													*(top_data + num_offset_top + row_offset + (col_in_output_data + col) * output_Channel_ + och) = result[row * m_ + col];
												}
											}
											else
											{
												for (int col = 0; col < m_; col++)
												{
													*(top_data + num_offset_top + row_offset + (col_in_output_data + col) * output_Channel_ + och) = result[row * m_ + col];
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
							forward_bias(top_data + n * top_dim_, bias);
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