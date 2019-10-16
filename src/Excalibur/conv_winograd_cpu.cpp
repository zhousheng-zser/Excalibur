#include "conv_winograd_cpu.hpp"
#include "../../include/Excalibur/tensor_operation_cpu.hpp"
#include <iostream>
#include <fstream>

namespace glasssix
{
	namespace excalibur
	{
		conv_winograd_cpu::conv_winograd_cpu(int input_Channel, int output_Channel, int group, int kernelSize, int stride, int pad, bool bias_term, int device, bool int8_quantization)
			: baseconv(input_Channel, output_Channel, group, kernelSize, stride, pad, bias_term, device, int8_quantization)
		{
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
			Forward_F23(bottom, top);
			//Forward_F43(bottom, top);
		}

		void conv_winograd_cpu::Forward_F23(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			CHECK_EQ(kernelSize_, 3);
			CHECK_EQ(stride_, 1);
			order_ = bottom->order();
			num_ = bottom->num();
			input_dim_h_ = bottom->height();
			input_dim_w_ = bottom->width();
			output_dim_h_ = (input_dim_h_ + 2 * pad_ - kernelSize_) / stride_ + 1;
			output_dim_w_ = (input_dim_w_ + 2 * pad_ - kernelSize_) / stride_ + 1;

			int h_subtract_tilesize = input_dim_h_ + 2 * pad_ - tile_size_;
			int w_subtract_tilesize = input_dim_w_ + 2 * pad_ - tile_size_;
			h_tile_num_ = int((float)h_subtract_tilesize / m_ + 0.5f) + 1;//h_tile_num_ = ceil((H-(m+r-1))/m) + 1, H is height after padding
			w_tile_num_ = int((float)w_subtract_tilesize / m_ + 0.5f) + 1;//w_tile_num_ = ceil((W-(m+r-1))/m) + 1, W is width after padding
			int total_tile_num = h_tile_num_ * w_tile_num_;
			int w_tile_stride = w_tile_num_ * tile_length_;
			int h_w_tile_stride = h_tile_num_ * w_tile_stride;
			int h_aligned = (h_subtract_tilesize + m_ - 1) / m_ * m_;
			int w_aligned = (w_subtract_tilesize + m_ - 1) / m_ * m_;
			int add_h = h_aligned - h_subtract_tilesize;
			int add_w = w_aligned - w_subtract_tilesize;

			input_dim_w_ += 2 * pad_ + add_w;
			input_dim_h_ += 2 * pad_ + add_h;
			output_dim_w_ += add_w;
			output_dim_h_ += add_h;
			input_spatial_dim_ = input_dim_w_ * input_dim_h_;
			output_spatial_dim_ = output_dim_w_ * output_dim_h_;

			std::shared_ptr<tensor<float>> bottom_bordered;
			tensor_operation_cpu::make_border_cpu(bottom, bottom_bordered, pad_, pad_ + add_h, pad_, pad_ + add_w);
			bottom_data = bottom_bordered->cpu_data();
			bottom_dim_ = bottom_bordered->count(1, 4);

			top.reset(new tensor<float>(std::vector<int>{num_, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
			top_data = top->mutable_cpu_data();
			top_dim_ = top->count(1, 4);

			if (int8_quantization_)
			{
				bias_multiplier_.reset(new tensor<float>(std::vector<int>{output_spatial_dim_}, device_));
				bias_multiplier_data = bias_multiplier_->mutable_cpu_data();

				top_int32_.reset(new tensor<int>(std::vector<int>{num_, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
				top_int32_data = top_int32_->mutable_cpu_data();

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
				//V=BT*d*B,so V has the same number as data tile, there are tile_length_ elements in single V
				//V_int16.reset(new tensor<short>(std::vector<int>{1, input_Channel_, total_tile_num, tile_length_}));
				//V_int16_data = V_int16->mutable_cpu_data();
				V_int16_data = new short[input_Channel_ * total_tile_num * tile_length_];

				if (order_ == NCHW)
				{
					if (group_ > 1)
					{
						for (int n = 0; n < num_; n++)
						{
							int bottom_offset_num = n * bottom_dim_;
							int top_offset_num = n * top_dim_;

							//get transformed bottom_data: V
							{
								int nn_inch = input_Channel_ >> 2;
								int remain_inch_start = nn_inch << 2;

#ifdef _OPENMP
#pragma omp parallel for
#endif
								for (int nn = 0; nn < nn_inch; nn++)
								{
									int ich = nn * 4;
									int bottom_offset_num_ich_1 = bottom_offset_num + ich * input_spatial_dim_;
									int bottom_offset_num_ich_2 = bottom_offset_num_ich_1 + input_spatial_dim_;
									int bottom_offset_num_ich_3 = bottom_offset_num_ich_2 + input_spatial_dim_;
									int bottom_offset_num_ich_4 = bottom_offset_num_ich_3 + input_spatial_dim_;
									int V_offset_ich_1 = ich * h_w_tile_stride;
									int V_offset_ich_2 = V_offset_ich_1 + h_w_tile_stride;
									int V_offset_ich_3 = V_offset_ich_2 + h_w_tile_stride;
									int V_offset_ich_4 = V_offset_ich_3 + h_w_tile_stride;

									for (int i = 0; i < total_tile_num; i++)
									{
										int row_in_input_data = (i / w_tile_num_) * m_;
										int col_in_input_data = (i % w_tile_num_) * m_;
										int bottom_offset_row_col = row_in_input_data * input_dim_w_ + col_in_input_data;
										int bottom_offset_num_ich_row_col_1 = bottom_offset_num_ich_1 + bottom_offset_row_col;
										int bottom_offset_num_ich_row_col_2 = bottom_offset_num_ich_2 + bottom_offset_row_col;
										int bottom_offset_num_ich_row_col_3 = bottom_offset_num_ich_3 + bottom_offset_row_col;
										int bottom_offset_num_ich_row_col_4 = bottom_offset_num_ich_4 + bottom_offset_row_col;

										int tile_offset = i * tile_length_;
										int V_offset_ich_row_col_1 = V_offset_ich_1 + tile_offset;
										int V_offset_ich_row_col_2 = V_offset_ich_2 + tile_offset;
										int V_offset_ich_row_col_3 = V_offset_ich_3 + tile_offset;
										int V_offset_ich_row_col_4 = V_offset_ich_4 + tile_offset;

										const signed char *row_data1_1 = bottom_int8_data + bottom_offset_num_ich_row_col_1;
										const signed char *row_data1_2 = row_data1_1 + input_dim_w_;
										const signed char *row_data1_3 = row_data1_2 + input_dim_w_;
										const signed char *row_data1_4 = row_data1_3 + input_dim_w_;
										calculate_BTdB23(row_data1_1, row_data1_2, row_data1_3, row_data1_4, V_int16_data + V_offset_ich_row_col_1);

										const signed char *row_data2_1 = bottom_int8_data + bottom_offset_num_ich_row_col_2;
										const signed char *row_data2_2 = row_data2_1 + input_dim_w_;
										const signed char *row_data2_3 = row_data2_2 + input_dim_w_;
										const signed char *row_data2_4 = row_data2_3 + input_dim_w_;
										calculate_BTdB23(row_data2_1, row_data2_2, row_data2_3, row_data2_4, V_int16_data + V_offset_ich_row_col_2);

										const signed char *row_data3_1 = bottom_int8_data + bottom_offset_num_ich_row_col_3;
										const signed char *row_data3_2 = row_data3_1 + input_dim_w_;
										const signed char *row_data3_3 = row_data3_2 + input_dim_w_;
										const signed char *row_data3_4 = row_data3_3 + input_dim_w_;
										calculate_BTdB23(row_data3_1, row_data3_2, row_data3_3, row_data3_4, V_int16_data + V_offset_ich_row_col_3);

										const signed char *row_data4_1 = bottom_int8_data + bottom_offset_num_ich_row_col_4;
										const signed char *row_data4_2 = row_data4_1 + input_dim_w_;
										const signed char *row_data4_3 = row_data4_2 + input_dim_w_;
										const signed char *row_data4_4 = row_data4_3 + input_dim_w_;
										calculate_BTdB23(row_data4_1, row_data4_2, row_data4_3, row_data4_4, V_int16_data + V_offset_ich_row_col_4);
									}
								}

								for (int ich = remain_inch_start; ich < input_Channel_; ich++)
								{
									int bottom_offset_num_ich_1 = bottom_offset_num + ich * input_spatial_dim_;
									int V_offset_ich_1 = ich * h_w_tile_stride;

									for (int i = 0; i < total_tile_num; i++)
									{
										int row_in_input_data = (i / w_tile_num_) * m_;
										int col_in_input_data = (i % w_tile_num_) * m_;
										int bottom_offset_row_col = row_in_input_data * input_dim_w_ + col_in_input_data;
										int bottom_offset_num_ich_row_col_1 = bottom_offset_num_ich_1 + bottom_offset_row_col;

										int tile_offset = i * tile_length_;
										int V_offset_ich_row_col_1 = V_offset_ich_1 + tile_offset;

										const signed char *row_data1_1 = bottom_int8_data + bottom_offset_num_ich_row_col_1;
										const signed char *row_data1_2 = row_data1_1 + input_dim_w_;
										const signed char *row_data1_3 = row_data1_2 + input_dim_w_;
										const signed char *row_data1_4 = row_data1_3 + input_dim_w_;
										calculate_BTdB23(row_data1_1, row_data1_2, row_data1_3, row_data1_4, V_int16_data + V_offset_ich_row_col_1);
									}
								}
							}

							//multiply
							{
								int nn_outch = output_Channel_ >> 2;
								int remain_outch_start = nn_outch << 2;
#ifdef _OPENMP
#pragma omp parallel for
#endif
								for (int nn = 0; nn < nn_outch; nn++)
								{
#if SIMD_TYPE >= SIMDTYPE_AVX
									mm_typei sum_1_0;
									mm_typei sum_1_8;
									mm_typei sum_2_0;
									mm_typei sum_2_8;
									mm_typei sum_3_0;
									mm_typei sum_3_8;
									mm_typei sum_4_0;
									mm_typei sum_4_8;
									mm_typei u_1_0;
									mm_typei u_1_8;
									mm_typei u_2_0;
									mm_typei u_2_8;
									mm_typei u_3_0;
									mm_typei u_3_8;
									mm_typei u_4_0;
									mm_typei u_4_8;
									__m128i u_1_0_int16;
									__m128i u_1_8_int16;
									__m128i u_2_0_int16;
									__m128i u_2_8_int16;
									__m128i u_3_0_int16;
									__m128i u_3_8_int16;
									__m128i u_4_0_int16;
									__m128i u_4_8_int16;
									mm_typei v_1_0;
									mm_typei v_1_8;
									mm_typei v_2_0;
									mm_typei v_2_8;
									mm_typei v_3_0;
									mm_typei v_3_8;
									mm_typei v_4_0;
									mm_typei v_4_8;
									__m128i v_1_0_int16;
									__m128i v_1_8_int16;
									__m128i v_2_0_int16;
									__m128i v_2_8_int16;
									__m128i v_3_0_int16;
									__m128i v_3_8_int16;
									__m128i v_4_0_int16;
									__m128i v_4_8_int16;
#elif SIMD_TYPE >= SIMDTYPE_SSE
									mm_typei sum_1_0;
									mm_typei sum_1_4;
									mm_typei sum_1_8;
									mm_typei sum_1_12;
									mm_typei sum_2_0;
									mm_typei sum_2_4;
									mm_typei sum_2_8;
									mm_typei sum_2_12;
									mm_typei sum_3_0;
									mm_typei sum_3_4;
									mm_typei sum_3_8;
									mm_typei sum_3_12;
									mm_typei sum_4_0;
									mm_typei sum_4_4;
									mm_typei sum_4_8;
									mm_typei sum_4_12;
									mm_typei u_1_0;
									mm_typei u_1_4;
									mm_typei u_1_8;
									mm_typei u_1_12;
									mm_typei u_2_0;
									mm_typei u_2_4;
									mm_typei u_2_8;
									mm_typei u_2_12;
									mm_typei u_3_0;
									mm_typei u_3_4;
									mm_typei u_3_8;
									mm_typei u_3_12;
									mm_typei u_4_0;
									mm_typei u_4_4;
									mm_typei u_4_8;
									mm_typei u_4_12;
									mm_typei u_1_0_int16;
									mm_typei u_1_4_int16;
									mm_typei u_1_8_int16;
									mm_typei u_1_12_int16;
									mm_typei u_2_0_int16;
									mm_typei u_2_4_int16;
									mm_typei u_2_8_int16;
									mm_typei u_2_12_int16;
									mm_typei u_3_0_int16;
									mm_typei u_3_4_int16;
									mm_typei u_3_8_int16;
									mm_typei u_3_12_int16;
									mm_typei u_4_0_int16;
									mm_typei u_4_4_int16;
									mm_typei u_4_8_int16;
									mm_typei u_4_12_int16;
									mm_typei v_1_0;
									mm_typei v_1_4;
									mm_typei v_1_8;
									mm_typei v_1_12;
									mm_typei v_2_0;
									mm_typei v_2_4;
									mm_typei v_2_8;
									mm_typei v_2_12;
									mm_typei v_3_0;
									mm_typei v_3_4;
									mm_typei v_3_8;
									mm_typei v_3_12;
									mm_typei v_4_0;
									mm_typei v_4_4;
									mm_typei v_4_8;
									mm_typei v_4_12;
									mm_typei v_1_0_int16;
									mm_typei v_1_4_int16;
									mm_typei v_1_8_int16;
									mm_typei v_1_12_int16;
									mm_typei v_2_0_int16;
									mm_typei v_2_4_int16;
									mm_typei v_2_8_int16;
									mm_typei v_2_12_int16;
									mm_typei v_3_0_int16;
									mm_typei v_3_4_int16;
									mm_typei v_3_8_int16;
									mm_typei v_3_12_int16;
									mm_typei v_4_0_int16;
									mm_typei v_4_4_int16;
									mm_typei v_4_8_int16;
									mm_typei v_4_12_int16;
#endif

									int och = nn * 4;
									int U_offset_och_1 = och * tile_length_;
									int U_offset_och_2 = U_offset_och_1 + tile_length_;
									int U_offset_och_3 = U_offset_och_2 + tile_length_;
									int U_offset_och_4 = U_offset_och_3 + tile_length_;

									int V_offset_och_1 = och * h_w_tile_stride;
									int V_offset_och_2 = V_offset_och_1 + h_w_tile_stride;
									int V_offset_och_3 = V_offset_och_2 + h_w_tile_stride;
									int V_offset_och_4 = V_offset_och_3 + h_w_tile_stride;

									float bias1 = bias_data[och];
									float bias2 = bias_data[och + 1];
									float bias3 = bias_data[och + 2];
									float bias4 = bias_data[och + 3];
									int result1[4], result2[4], result3[4], result4[4];

									int top_offset_num_och_1 = top_offset_num + och * output_spatial_dim_;
									int top_offset_num_och_2 = top_offset_num_och_1 + output_spatial_dim_;
									int top_offset_num_och_3 = top_offset_num_och_2 + output_spatial_dim_;
									int top_offset_num_och_4 = top_offset_num_och_3 + output_spatial_dim_;

									for (int i = 0; i < total_tile_num; i++)
									{
										int mult_data1[16] = { 0 };
										int mult_data2[16] = { 0 };
										int mult_data3[16] = { 0 };
										int mult_data4[16] = { 0 };

										int V_offset_row_col = i * tile_length_;
										int V_offset_och_row_col_1 = V_offset_och_1 + V_offset_row_col;
										int V_offset_och_row_col_2 = V_offset_och_2 + V_offset_row_col;
										int V_offset_och_row_col_3 = V_offset_och_3 + V_offset_row_col;
										int V_offset_och_row_col_4 = V_offset_och_4 + V_offset_row_col;

#if SIMD_TYPE >= SIMDTYPE_AVX
										u_1_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_1));
										u_1_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_1 + 8));
										u_2_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_2));
										u_2_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_2 + 8));
										u_3_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_3));
										u_3_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_3 + 8));
										u_4_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_4));
										u_4_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_4 + 8));
										u_1_0 = mm_cvtepi16_epi32(u_1_0_int16);
										u_1_8 = mm_cvtepi16_epi32(u_1_8_int16);
										u_2_0 = mm_cvtepi16_epi32(u_2_0_int16);
										u_2_8 = mm_cvtepi16_epi32(u_2_8_int16);
										u_3_0 = mm_cvtepi16_epi32(u_3_0_int16);
										u_3_8 = mm_cvtepi16_epi32(u_3_8_int16);
										u_4_0 = mm_cvtepi16_epi32(u_4_0_int16);
										u_4_8 = mm_cvtepi16_epi32(u_4_8_int16);

										v_1_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_1));
										v_1_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_1 + 8));
										v_2_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_2));
										v_2_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_2 + 8));
										v_3_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_3));
										v_3_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_3 + 8));
										v_4_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_4));
										v_4_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_4 + 8));
										v_1_0 = mm_cvtepi16_epi32(v_1_0_int16);
										v_1_8 = mm_cvtepi16_epi32(v_1_8_int16);
										v_2_0 = mm_cvtepi16_epi32(v_2_0_int16);
										v_2_8 = mm_cvtepi16_epi32(v_2_8_int16);
										v_3_0 = mm_cvtepi16_epi32(v_3_0_int16);
										v_3_8 = mm_cvtepi16_epi32(v_3_8_int16);
										v_4_0 = mm_cvtepi16_epi32(v_4_0_int16);
										v_4_8 = mm_cvtepi16_epi32(v_4_8_int16);

										sum_1_0 = mm_mullo_epi32(v_1_0, u_1_0);
										sum_1_8 = mm_mullo_epi32(v_1_8, u_1_8);
										sum_2_0 = mm_mullo_epi32(v_2_0, u_2_0);
										sum_2_8 = mm_mullo_epi32(v_2_8, u_2_8);
										sum_3_0 = mm_mullo_epi32(v_3_0, u_3_0);
										sum_3_8 = mm_mullo_epi32(v_3_8, u_3_8);
										sum_4_0 = mm_mullo_epi32(v_4_0, u_4_0);
										sum_4_8 = mm_mullo_epi32(v_4_8, u_4_8);

										mm_store_si((mm_typei*)mult_data1, sum_1_0);
										mm_store_si((mm_typei*)(mult_data1 + 8), sum_1_8);
										mm_store_si((mm_typei*)mult_data2, sum_2_0);
										mm_store_si((mm_typei*)(mult_data2 + 8), sum_2_8);
										mm_store_si((mm_typei*)mult_data3, sum_3_0);
										mm_store_si((mm_typei*)(mult_data3 + 8), sum_3_8);
										mm_store_si((mm_typei*)mult_data4, sum_4_0);
										mm_store_si((mm_typei*)(mult_data4 + 8), sum_4_8);
#elif SIMD_TYPE >= SIMDTYPE_SSE
										u_1_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_1));
										u_1_4_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_1 + 4));
										u_1_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_1 + 8));
										u_1_12_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_1 + 12));
										u_1_0 = mm_cvtepi16_epi32(u_1_0_int16);
										u_1_4 = mm_cvtepi16_epi32(u_1_4_int16);
										u_1_8 = mm_cvtepi16_epi32(u_1_8_int16);
										u_1_12 = mm_cvtepi16_epi32(u_1_12_int16);

										u_2_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_2));
										u_2_4_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_2 + 4));
										u_2_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_2 + 8));
										u_2_12_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_2 + 12));
										u_2_0 = mm_cvtepi16_epi32(u_2_0_int16);
										u_2_4 = mm_cvtepi16_epi32(u_2_4_int16);
										u_2_8 = mm_cvtepi16_epi32(u_2_8_int16);
										u_2_12 = mm_cvtepi16_epi32(u_2_12_int16);

										u_3_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_3));
										u_3_4_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_3 + 4));
										u_3_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_3 + 8));
										u_3_12_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_3 + 12));
										u_3_0 = mm_cvtepi16_epi32(u_3_0_int16);
										u_3_4 = mm_cvtepi16_epi32(u_3_4_int16);
										u_3_8 = mm_cvtepi16_epi32(u_3_8_int16);
										u_3_12 = mm_cvtepi16_epi32(u_3_12_int16);

										u_4_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_4));
										u_4_4_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_4 + 4));
										u_4_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_4 + 8));
										u_4_12_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_4 + 12));
										u_4_0 = mm_cvtepi16_epi32(u_4_0_int16);
										u_4_4 = mm_cvtepi16_epi32(u_4_4_int16);
										u_4_8 = mm_cvtepi16_epi32(u_4_8_int16);
										u_4_12 = mm_cvtepi16_epi32(u_4_12_int16);


										v_1_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_1));
										v_1_4_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_1 + 4));
										v_1_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_1 + 8));
										v_1_12_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_1 + 12));
										v_1_0 = mm_cvtepi16_epi32(v_1_0_int16);
										v_1_4 = mm_cvtepi16_epi32(v_1_4_int16);
										v_1_8 = mm_cvtepi16_epi32(v_1_8_int16);
										v_1_12 = mm_cvtepi16_epi32(v_1_12_int16);

										v_2_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_2));
										v_2_4_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_2 + 4));
										v_2_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_2 + 8));
										v_2_12_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_2 + 12));
										v_2_0 = mm_cvtepi16_epi32(v_2_0_int16);
										v_2_4 = mm_cvtepi16_epi32(v_2_4_int16);
										v_2_8 = mm_cvtepi16_epi32(v_2_8_int16);
										v_2_12 = mm_cvtepi16_epi32(v_2_12_int16);

										v_3_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_3));
										v_3_4_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_3 + 4));
										v_3_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_3 + 8));
										v_3_12_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_3 + 12));
										v_3_0 = mm_cvtepi16_epi32(v_3_0_int16);
										v_3_4 = mm_cvtepi16_epi32(v_3_4_int16);
										v_3_8 = mm_cvtepi16_epi32(v_3_8_int16);
										v_3_12 = mm_cvtepi16_epi32(v_3_12_int16);

										v_4_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_4));
										v_4_4_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_4 + 4));
										v_4_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_4 + 8));
										v_4_12_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_4 + 12));
										v_4_0 = mm_cvtepi16_epi32(v_4_0_int16);
										v_4_4 = mm_cvtepi16_epi32(v_4_4_int16);
										v_4_8 = mm_cvtepi16_epi32(v_4_8_int16);
										v_4_12 = mm_cvtepi16_epi32(v_4_12_int16);

										sum_1_0 = mm_mullo_epi32(v_1_0, u_1_0);
										sum_1_4 = mm_mullo_epi32(v_1_4, u_1_4);
										sum_1_8 = mm_mullo_epi32(v_1_8, u_1_8);
										sum_1_12 = mm_mullo_epi32(v_1_12, u_1_12);

										sum_2_0 = mm_mullo_epi32(v_2_0, u_2_0);
										sum_2_4 = mm_mullo_epi32(v_2_4, u_2_4);
										sum_2_8 = mm_mullo_epi32(v_2_8, u_2_8);
										sum_2_12 = mm_mullo_epi32(v_2_12, u_2_12);

										sum_3_0 = mm_mullo_epi32(v_3_0, u_3_0);
										sum_3_4 = mm_mullo_epi32(v_3_4, u_3_4);
										sum_3_8 = mm_mullo_epi32(v_3_8, u_3_8);
										sum_3_12 = mm_mullo_epi32(v_3_12, u_3_12);

										sum_4_0 = mm_mullo_epi32(v_4_0, u_4_0);
										sum_4_4 = mm_mullo_epi32(v_4_4, u_4_4);
										sum_4_8 = mm_mullo_epi32(v_4_8, u_4_8);
										sum_4_12 = mm_mullo_epi32(v_4_12, u_4_12);

										mm_store_si((mm_typei*)mult_data1, sum_1_0);
										mm_store_si((mm_typei*)(mult_data1 + 4), sum_1_4);
										mm_store_si((mm_typei*)(mult_data1 + 8), sum_1_8);
										mm_store_si((mm_typei*)(mult_data1 + 12), sum_1_12);
										mm_store_si((mm_typei*)mult_data2, sum_2_0);
										mm_store_si((mm_typei*)(mult_data2 + 4), sum_2_4);
										mm_store_si((mm_typei*)(mult_data2 + 8), sum_2_8);
										mm_store_si((mm_typei*)(mult_data2 + 12), sum_2_12);
										mm_store_si((mm_typei*)mult_data3, sum_3_0);
										mm_store_si((mm_typei*)(mult_data3 + 4), sum_3_4);
										mm_store_si((mm_typei*)(mult_data3 + 8), sum_3_8);
										mm_store_si((mm_typei*)(mult_data3 + 12), sum_3_12);
										mm_store_si((mm_typei*)mult_data4, sum_4_0);
										mm_store_si((mm_typei*)(mult_data4 + 4), sum_4_4);
										mm_store_si((mm_typei*)(mult_data4 + 8), sum_4_8);
										mm_store_si((mm_typei*)(mult_data4 + 12), sum_4_12);
#else
										for (int i = 0; i < tile_length_; i++)
										{
											mult_data1[i] = U_int16_data[U_offset_och_1 + i] * V_int16_data[V_offset_och_row_col_1 + i];
											mult_data2[i] = U_int16_data[U_offset_och_2 + i] * V_int16_data[V_offset_och_row_col_2 + i];
											mult_data3[i] = U_int16_data[U_offset_och_3 + i] * V_int16_data[V_offset_och_row_col_3 + i];
											mult_data4[i] = U_int16_data[U_offset_och_4 + i] * V_int16_data[V_offset_och_row_col_4 + i];
										}
#endif

										calculate_ATmA23(mult_data1, result1);
										calculate_ATmA23(mult_data2, result2);
										calculate_ATmA23(mult_data3, result3);
										calculate_ATmA23(mult_data4, result4);

										int row_in_output_data = i / w_tile_num_ * m_;
										int col_in_output_data = i % w_tile_num_* m_;
										int top_offset_row_col = row_in_output_data * output_dim_w_ + col_in_output_data;
										int top_offset_num_och_row_col_1 = top_offset_num_och_1 + top_offset_row_col;
										int top_offset_num_och_row_col_2 = top_offset_num_och_2 + top_offset_row_col;
										int top_offset_num_och_row_col_3 = top_offset_num_och_3 + top_offset_row_col;
										int top_offset_num_och_row_col_4 = top_offset_num_och_4 + top_offset_row_col;

										for (int row = 0; row < m_; row++)
										{
											int result_offset_row = row * m_;
											for (int col = 0; col < m_; col++)
											{
												top_int32_data[top_offset_num_och_row_col_1 + col] = result1[result_offset_row + col];
												top_int32_data[top_offset_num_och_row_col_2 + col] = result2[result_offset_row + col];
												top_int32_data[top_offset_num_och_row_col_3 + col] = result3[result_offset_row + col];
												top_int32_data[top_offset_num_och_row_col_4 + col] = result4[result_offset_row + col];
											}
											top_offset_num_och_row_col_1 += output_dim_w_;
											top_offset_num_och_row_col_2 += output_dim_w_;
											top_offset_num_och_row_col_3 += output_dim_w_;
											top_offset_num_och_row_col_4 += output_dim_w_;
										}
									}
								}

								for (int och = remain_outch_start; och < output_Channel_; och++)
								{
#if SIMD_TYPE >= SIMDTYPE_AVX
									mm_typei sum_1_0;
									mm_typei sum_1_8;
									__m128i u_1_0_int16;
									__m128i u_1_8_int16;
									__m128i v_1_0_int16;
									__m128i v_1_8_int16;
									mm_typei u_1_0;
									mm_typei u_1_8;
									mm_typei v_1_0;
									mm_typei v_1_8;
#elif SIMD_TYPE >= SIMDTYPE_SSE
									mm_typei sum_1_0;
									mm_typei sum_1_4;
									mm_typei sum_1_8;
									mm_typei sum_1_12;
									mm_typei u_1_0;
									mm_typei u_1_4;
									mm_typei u_1_8;
									mm_typei u_1_12;
									mm_typei u_1_0_int16;
									mm_typei u_1_4_int16;
									mm_typei u_1_8_int16;
									mm_typei u_1_12_int16;
									mm_typei v_1_0;
									mm_typei v_1_4;
									mm_typei v_1_8;
									mm_typei v_1_12;
									mm_typei v_1_0_int16;
									mm_typei v_1_4_int16;
									mm_typei v_1_8_int16;
									mm_typei v_1_12_int16;
#endif

									int U_offset_och_1 = och * tile_length_;
									int V_offset_och_1 = och * h_w_tile_stride;

									float bias1 = bias_data[och];
									int result1[4];
									int top_offset_num_och_1 = top_offset_num + och * output_spatial_dim_;

									for (int i = 0; i < total_tile_num; i++)
									{
										int mult_data1[16] = { 0 };
										int V_offset_row_col = i * tile_length_;

										int V_offset_och_row_col_1 = V_offset_och_1 + V_offset_row_col;

#if SIMD_TYPE >= SIMDTYPE_AVX
										u_1_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_1));
										u_1_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_1 + 8));
										u_1_0 = mm_cvtepi16_epi32(u_1_0_int16);
										u_1_8 = mm_cvtepi16_epi32(u_1_8_int16);

										v_1_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_1));
										v_1_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_1 + 8));
										v_1_0 = mm_cvtepi16_epi32(v_1_0_int16);
										v_1_8 = mm_cvtepi16_epi32(v_1_8_int16);

										sum_1_0 = mm_mullo_epi32(v_1_0, u_1_0);
										sum_1_8 = mm_mullo_epi32(v_1_8, u_1_8);

										mm_store_si((mm_typei*)mult_data1, sum_1_0);
										mm_store_si((mm_typei*)(mult_data1 + 8), sum_1_8);
#elif SIMD_TYPE >= SIMDTYPE_SSE
										u_1_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_1));
										u_1_4_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_1 + 4));
										u_1_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_1 + 8));
										u_1_12_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_1 + 12));
										u_1_0 = mm_cvtepi16_epi32(u_1_0_int16);
										u_1_4 = mm_cvtepi16_epi32(u_1_4_int16);
										u_1_8 = mm_cvtepi16_epi32(u_1_8_int16);
										u_1_12 = mm_cvtepi16_epi32(u_1_12_int16);

										v_1_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_1));
										v_1_4_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_1 + 4));
										v_1_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_1 + 8));
										v_1_12_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_och_row_col_1 + 12));
										v_1_0 = mm_cvtepi16_epi32(v_1_0_int16);
										v_1_4 = mm_cvtepi16_epi32(v_1_4_int16);
										v_1_8 = mm_cvtepi16_epi32(v_1_8_int16);
										v_1_12 = mm_cvtepi16_epi32(v_1_12_int16);

										sum_1_0 = mm_mullo_epi32(v_1_0, u_1_0);
										sum_1_4 = mm_mullo_epi32(v_1_4, u_1_4);
										sum_1_8 = mm_mullo_epi32(v_1_8, u_1_8);
										sum_1_12 = mm_mullo_epi32(v_1_12, u_1_12);

										mm_store_si((mm_typei*)mult_data1, sum_1_0);
										mm_store_si((mm_typei*)(mult_data1 + 4), sum_1_4);
										mm_store_si((mm_typei*)(mult_data1 + 8), sum_1_8);
										mm_store_si((mm_typei*)(mult_data1 + 12), sum_1_12);
#else
										for (int i = 0; i < tile_length_; i++)
										{
											mult_data1[i] = U_int16_data[U_offset_och_1 + i] * V_int16_data[V_offset_och_row_col_1 + i];
										}
#endif

										calculate_ATmA23(mult_data1, result1);

										int row_in_output_data = i / w_tile_num_ * m_;
										int col_in_output_data = i % w_tile_num_* m_;
										int top_offset_row_col = row_in_output_data * output_dim_w_ + col_in_output_data;
										int top_offset_num_och_row_col_1 = top_offset_num_och_1 + top_offset_row_col;

										for (int row = 0; row < m_; row++)
										{
											int result_offset_row = row * m_;
											for (int col = 0; col < m_; col++)
											{
												top_int32_data[top_offset_num_och_row_col_1 + col] = result1[result_offset_row + col];
											}
											top_offset_num_och_row_col_1 += output_dim_w_;
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
							}
						}

						if ((add_h != 0) || (add_w != 0))
						{
							tensor_operation_cpu::cut_border_cpu(top, top, 0, add_h, 0, add_w);
						}
					}
					else if (group_ == 1)
					{
						int U_offset_single_och = input_Channel_ * tile_length_;

						for (int n = 0; n < num_; n++)
						{
							int bottom_offset_num = n * bottom_dim_;
							int top_offset_num = n * top_dim_;

							//get transformed bottom_data: V
							{
								int nn_inch = input_Channel_ >> 2;
								int remain_inch_start = nn_inch << 2;

#ifdef _OPENMP
#pragma omp parallel for
#endif
								for (int nn = 0; nn < nn_inch; nn++)
								{
									int ich = nn * 4;
									int bottom_offset_num_ich_1 = bottom_offset_num + ich * input_spatial_dim_;
									int bottom_offset_num_ich_2 = bottom_offset_num_ich_1 + input_spatial_dim_;
									int bottom_offset_num_ich_3 = bottom_offset_num_ich_2 + input_spatial_dim_;
									int bottom_offset_num_ich_4 = bottom_offset_num_ich_3 + input_spatial_dim_;
									int V_offset_ich_1 = ich * h_w_tile_stride;
									int V_offset_ich_2 = V_offset_ich_1 + h_w_tile_stride;
									int V_offset_ich_3 = V_offset_ich_2 + h_w_tile_stride;
									int V_offset_ich_4 = V_offset_ich_3 + h_w_tile_stride;

									for (int i = 0; i < total_tile_num; i++)
									{
										int row_in_input_data = (i / w_tile_num_) * m_;
										int col_in_input_data = (i % w_tile_num_) * m_;
										int bottom_offset_row_col = row_in_input_data * input_dim_w_ + col_in_input_data;
										int bottom_offset_num_ich_row_col_1 = bottom_offset_num_ich_1 + bottom_offset_row_col;
										int bottom_offset_num_ich_row_col_2 = bottom_offset_num_ich_2 + bottom_offset_row_col;
										int bottom_offset_num_ich_row_col_3 = bottom_offset_num_ich_3 + bottom_offset_row_col;
										int bottom_offset_num_ich_row_col_4 = bottom_offset_num_ich_4 + bottom_offset_row_col;

										int tile_offset = i * tile_length_;
										int V_offset_ich_row_col_1 = V_offset_ich_1 + tile_offset;
										int V_offset_ich_row_col_2 = V_offset_ich_2 + tile_offset;
										int V_offset_ich_row_col_3 = V_offset_ich_3 + tile_offset;
										int V_offset_ich_row_col_4 = V_offset_ich_4 + tile_offset;

										const signed char *row_data1_1 = bottom_int8_data + bottom_offset_num_ich_row_col_1;
										const signed char *row_data1_2 = row_data1_1 + input_dim_w_;
										const signed char *row_data1_3 = row_data1_2 + input_dim_w_;
										const signed char *row_data1_4 = row_data1_3 + input_dim_w_;
										calculate_BTdB23(row_data1_1, row_data1_2, row_data1_3, row_data1_4, V_int16_data + V_offset_ich_row_col_1);

										const signed char *row_data2_1 = bottom_int8_data + bottom_offset_num_ich_row_col_2;
										const signed char *row_data2_2 = row_data2_1 + input_dim_w_;
										const signed char *row_data2_3 = row_data2_2 + input_dim_w_;
										const signed char *row_data2_4 = row_data2_3 + input_dim_w_;
										calculate_BTdB23(row_data2_1, row_data2_2, row_data2_3, row_data2_4, V_int16_data + V_offset_ich_row_col_2);

										const signed char *row_data3_1 = bottom_int8_data + bottom_offset_num_ich_row_col_3;
										const signed char *row_data3_2 = row_data3_1 + input_dim_w_;
										const signed char *row_data3_3 = row_data3_2 + input_dim_w_;
										const signed char *row_data3_4 = row_data3_3 + input_dim_w_;
										calculate_BTdB23(row_data3_1, row_data3_2, row_data3_3, row_data3_4, V_int16_data + V_offset_ich_row_col_3);

										const signed char *row_data4_1 = bottom_int8_data + bottom_offset_num_ich_row_col_4;
										const signed char *row_data4_2 = row_data4_1 + input_dim_w_;
										const signed char *row_data4_3 = row_data4_2 + input_dim_w_;
										const signed char *row_data4_4 = row_data4_3 + input_dim_w_;
										calculate_BTdB23(row_data4_1, row_data4_2, row_data4_3, row_data4_4, V_int16_data + V_offset_ich_row_col_4);
									}
								}

								for (int ich = remain_inch_start; ich < input_Channel_; ich++)
								{
									int bottom_offset_num_ich_1 = bottom_offset_num + ich * input_spatial_dim_;
									int V_offset_ich_1 = ich * h_w_tile_stride;

									for (int i = 0; i < total_tile_num; i++)
									{
										int row_in_input_data = (i / w_tile_num_) * m_;
										int col_in_input_data = (i % w_tile_num_) * m_;
										int bottom_offset_row_col = row_in_input_data * input_dim_w_ + col_in_input_data;
										int bottom_offset_num_ich_row_col_1 = bottom_offset_num_ich_1 + bottom_offset_row_col;

										int tile_offset = i * tile_length_;
										int V_offset_ich_row_col_1 = V_offset_ich_1 + tile_offset;

										const signed char *row_data1_1 = bottom_int8_data + bottom_offset_num_ich_row_col_1;
										const signed char *row_data1_2 = row_data1_1 + input_dim_w_;
										const signed char *row_data1_3 = row_data1_2 + input_dim_w_;
										const signed char *row_data1_4 = row_data1_3 + input_dim_w_;
										calculate_BTdB23(row_data1_1, row_data1_2, row_data1_3, row_data1_4, V_int16_data + V_offset_ich_row_col_1);
									}
								}
							}

							//multiply
							{
								int nn_outch = output_Channel_ >> 2;
								int remain_outch_start = nn_outch << 2;
#ifdef _OPENMP
#pragma omp parallel for
#endif
								for (int nn = 0; nn < nn_outch; nn++)
								{
#if SIMD_TYPE >= SIMDTYPE_AVX
									mm_typei sum_1_0;
									mm_typei sum_1_8;
									mm_typei sum_2_0;
									mm_typei sum_2_8;
									mm_typei sum_3_0;
									mm_typei sum_3_8;
									mm_typei sum_4_0;
									mm_typei sum_4_8;
									mm_typei u_1_0;
									mm_typei u_1_8;
									mm_typei u_1_16;
									mm_typei u_1_24;
									mm_typei u_1_32;
									mm_typei u_1_40;
									mm_typei u_1_48;
									mm_typei u_1_56;
									mm_typei u_2_0;
									mm_typei u_2_8;
									mm_typei u_2_16;
									mm_typei u_2_24;
									mm_typei u_2_32;
									mm_typei u_2_40;
									mm_typei u_2_48;
									mm_typei u_2_56;
									mm_typei u_3_0;
									mm_typei u_3_8;
									mm_typei u_3_16;
									mm_typei u_3_24;
									mm_typei u_3_32;
									mm_typei u_3_40;
									mm_typei u_3_48;
									mm_typei u_3_56;
									mm_typei u_4_0;
									mm_typei u_4_8;
									mm_typei u_4_16;
									mm_typei u_4_24;
									mm_typei u_4_32;
									mm_typei u_4_40;
									mm_typei u_4_48;
									mm_typei u_4_56;
									__m128i u_1_0_int16;
									__m128i u_1_8_int16;
									__m128i u_1_16_int16;
									__m128i u_1_24_int16;
									__m128i u_1_32_int16;
									__m128i u_1_40_int16;
									__m128i u_1_48_int16;
									__m128i u_1_56_int16;
									__m128i u_2_0_int16;
									__m128i u_2_8_int16;
									__m128i u_2_16_int16;
									__m128i u_2_24_int16;
									__m128i u_2_32_int16;
									__m128i u_2_40_int16;
									__m128i u_2_48_int16;
									__m128i u_2_56_int16;
									__m128i u_3_0_int16;
									__m128i u_3_8_int16;
									__m128i u_3_16_int16;
									__m128i u_3_24_int16;
									__m128i u_3_32_int16;
									__m128i u_3_40_int16;
									__m128i u_3_48_int16;
									__m128i u_3_56_int16;
									__m128i u_4_0_int16;
									__m128i u_4_8_int16;
									__m128i u_4_16_int16;
									__m128i u_4_24_int16;
									__m128i u_4_32_int16;
									__m128i u_4_40_int16;
									__m128i u_4_48_int16;
									__m128i u_4_56_int16;
									mm_typei v_1_0;
									mm_typei v_1_8;
									mm_typei v_2_0;
									mm_typei v_2_8;
									mm_typei v_3_0;
									mm_typei v_3_8;
									mm_typei v_4_0;
									mm_typei v_4_8;
									__m128i v_1_0_int16;
									__m128i v_1_8_int16;
									__m128i v_2_0_int16;
									__m128i v_2_8_int16;
									__m128i v_3_0_int16;
									__m128i v_3_8_int16;
									__m128i v_4_0_int16;
									__m128i v_4_8_int16;
#elif SIMD_TYPE >= SIMDTYPE_SSE
									mm_typei sum_1_0;
									mm_typei sum_1_4;
									mm_typei sum_1_8;
									mm_typei sum_1_12;
									mm_typei sum_2_0;
									mm_typei sum_2_4;
									mm_typei sum_2_8;
									mm_typei sum_2_12;
									mm_typei sum_3_0;
									mm_typei sum_3_4;
									mm_typei sum_3_8;
									mm_typei sum_3_12;
									mm_typei sum_4_0;
									mm_typei sum_4_4;
									mm_typei sum_4_8;
									mm_typei sum_4_12;
									mm_typei u_1_0;
									mm_typei u_1_4;
									mm_typei u_1_8;
									mm_typei u_1_12;
									mm_typei u_1_16;
									mm_typei u_1_20;
									mm_typei u_1_24;
									mm_typei u_1_28;
									mm_typei u_1_32;
									mm_typei u_1_36;
									mm_typei u_1_40;
									mm_typei u_1_44;
									mm_typei u_1_48;
									mm_typei u_1_52;
									mm_typei u_1_56;
									mm_typei u_1_60;
									mm_typei u_2_0;
									mm_typei u_2_4;
									mm_typei u_2_8;
									mm_typei u_2_12;
									mm_typei u_2_16;
									mm_typei u_2_20;
									mm_typei u_2_24;
									mm_typei u_2_28;
									mm_typei u_2_32;
									mm_typei u_2_36;
									mm_typei u_2_40;
									mm_typei u_2_44;
									mm_typei u_2_48;
									mm_typei u_2_52;
									mm_typei u_2_56;
									mm_typei u_2_60;
									mm_typei u_3_0;
									mm_typei u_3_4;
									mm_typei u_3_8;
									mm_typei u_3_12;
									mm_typei u_3_16;
									mm_typei u_3_20;
									mm_typei u_3_24;
									mm_typei u_3_28;
									mm_typei u_3_32;
									mm_typei u_3_36;
									mm_typei u_3_40;
									mm_typei u_3_44;
									mm_typei u_3_48;
									mm_typei u_3_52;
									mm_typei u_3_56;
									mm_typei u_3_60;
									mm_typei u_4_0;
									mm_typei u_4_4;
									mm_typei u_4_8;
									mm_typei u_4_12;
									mm_typei u_4_16;
									mm_typei u_4_20;
									mm_typei u_4_24;
									mm_typei u_4_28;
									mm_typei u_4_32;
									mm_typei u_4_36;
									mm_typei u_4_40;
									mm_typei u_4_44;
									mm_typei u_4_48;
									mm_typei u_4_52;
									mm_typei u_4_56;
									mm_typei u_4_60;
									mm_typei u_1_0_int16;
									mm_typei u_1_4_int16;
									mm_typei u_1_8_int16;
									mm_typei u_1_12_int16;
									mm_typei u_1_16_int16;
									mm_typei u_1_20_int16;
									mm_typei u_1_24_int16;
									mm_typei u_1_28_int16;
									mm_typei u_1_32_int16;
									mm_typei u_1_36_int16;
									mm_typei u_1_40_int16;
									mm_typei u_1_44_int16;
									mm_typei u_1_48_int16;
									mm_typei u_1_52_int16;
									mm_typei u_1_56_int16;
									mm_typei u_1_60_int16;
									mm_typei u_2_0_int16;
									mm_typei u_2_4_int16;
									mm_typei u_2_8_int16;
									mm_typei u_2_12_int16;
									mm_typei u_2_16_int16;
									mm_typei u_2_20_int16;
									mm_typei u_2_24_int16;
									mm_typei u_2_28_int16;
									mm_typei u_2_32_int16;
									mm_typei u_2_36_int16;
									mm_typei u_2_40_int16;
									mm_typei u_2_44_int16;
									mm_typei u_2_48_int16;
									mm_typei u_2_52_int16;
									mm_typei u_2_56_int16;
									mm_typei u_2_60_int16;
									mm_typei u_3_0_int16;
									mm_typei u_3_4_int16;
									mm_typei u_3_8_int16;
									mm_typei u_3_12_int16;
									mm_typei u_3_16_int16;
									mm_typei u_3_20_int16;
									mm_typei u_3_24_int16;
									mm_typei u_3_28_int16;
									mm_typei u_3_32_int16;
									mm_typei u_3_36_int16;
									mm_typei u_3_40_int16;
									mm_typei u_3_44_int16;
									mm_typei u_3_48_int16;
									mm_typei u_3_52_int16;
									mm_typei u_3_56_int16;
									mm_typei u_3_60_int16;
									mm_typei u_4_0_int16;
									mm_typei u_4_4_int16;
									mm_typei u_4_8_int16;
									mm_typei u_4_12_int16;
									mm_typei u_4_16_int16;
									mm_typei u_4_20_int16;
									mm_typei u_4_24_int16;
									mm_typei u_4_28_int16;
									mm_typei u_4_32_int16;
									mm_typei u_4_36_int16;
									mm_typei u_4_40_int16;
									mm_typei u_4_44_int16;
									mm_typei u_4_48_int16;
									mm_typei u_4_52_int16;
									mm_typei u_4_56_int16;
									mm_typei u_4_60_int16;
									mm_typei v_1_0;
									mm_typei v_1_4;
									mm_typei v_1_8;
									mm_typei v_1_12;
									mm_typei v_2_0;
									mm_typei v_2_4;
									mm_typei v_2_8;
									mm_typei v_2_12;
									mm_typei v_3_0;
									mm_typei v_3_4;
									mm_typei v_3_8;
									mm_typei v_3_12;
									mm_typei v_4_0;
									mm_typei v_4_4;
									mm_typei v_4_8;
									mm_typei v_4_12;
									mm_typei v_1_0_int16;
									mm_typei v_1_4_int16;
									mm_typei v_1_8_int16;
									mm_typei v_1_12_int16;
									mm_typei v_2_0_int16;
									mm_typei v_2_4_int16;
									mm_typei v_2_8_int16;
									mm_typei v_2_12_int16;
									mm_typei v_3_0_int16;
									mm_typei v_3_4_int16;
									mm_typei v_3_8_int16;
									mm_typei v_3_12_int16;
									mm_typei v_4_0_int16;
									mm_typei v_4_4_int16;
									mm_typei v_4_8_int16;
									mm_typei v_4_12_int16;
#endif

									int och = nn * 4;
									int U_offset_och_1 = och * U_offset_single_och;
									int U_offset_och_2 = U_offset_och_1 + U_offset_single_och;
									int U_offset_och_3 = U_offset_och_2 + U_offset_single_och;
									int U_offset_och_4 = U_offset_och_3 + U_offset_single_och;

									float bias1 = bias_data[och];
									float bias2 = bias_data[och + 1];
									float bias3 = bias_data[och + 2];
									float bias4 = bias_data[och + 3];
									int result1[4], result2[4], result3[4], result4[4];

									int top_offset_num_och_1 = top_offset_num + och * output_spatial_dim_;
									int top_offset_num_och_2 = top_offset_num_och_1 + output_spatial_dim_;
									int top_offset_num_och_3 = top_offset_num_och_2 + output_spatial_dim_;
									int top_offset_num_och_4 = top_offset_num_och_3 + output_spatial_dim_;

									for (int i = 0; i < total_tile_num; i++)
									{
										int mult_data1[16] = { 0 };
										int mult_data2[16] = { 0 };
										int mult_data3[16] = { 0 };
										int mult_data4[16] = { 0 };

										int V_offset_row_col = i * tile_length_;

#if SIMD_TYPE >= SIMDTYPE_AVX
										sum_1_0 = mm_setzero_si();
										sum_1_8 = mm_setzero_si();
										sum_2_0 = mm_setzero_si();
										sum_2_8 = mm_setzero_si();
										sum_3_0 = mm_setzero_si();
										sum_3_8 = mm_setzero_si();
										sum_4_0 = mm_setzero_si();
										sum_4_8 = mm_setzero_si();
#elif SIMD_TYPE >= SIMDTYPE_SSE
										sum_1_0 = mm_setzero_si();
										sum_1_4 = mm_setzero_si();
										sum_1_8 = mm_setzero_si();
										sum_1_12 = mm_setzero_si();
										sum_2_0 = mm_setzero_si();
										sum_2_4 = mm_setzero_si();
										sum_2_8 = mm_setzero_si();
										sum_2_12 = mm_setzero_si();
										sum_3_0 = mm_setzero_si();
										sum_3_4 = mm_setzero_si();
										sum_3_8 = mm_setzero_si();
										sum_3_12 = mm_setzero_si();
										sum_4_0 = mm_setzero_si();
										sum_4_4 = mm_setzero_si();
										sum_4_8 = mm_setzero_si();
										sum_4_12 = mm_setzero_si();
#endif // SIMD_TYPE >= SIMDTYPE_AVX
										int ich = 0;
										for (; ich + 3 < input_Channel_; ich += 4)
										{
											int offset = ich * tile_length_;
											int U_offset_och_ich_1 = U_offset_och_1 + offset;
											int U_offset_och_ich_2 = U_offset_och_2 + offset;
											int U_offset_och_ich_3 = U_offset_och_3 + offset;
											int U_offset_och_ich_4 = U_offset_och_4 + offset;

											int V_offset_ich_row_col_1 = V_offset_row_col + ich * h_w_tile_stride;
											int V_offset_ich_row_col_2 = V_offset_ich_row_col_1 + h_w_tile_stride;
											int V_offset_ich_row_col_3 = V_offset_ich_row_col_2 + h_w_tile_stride;
											int V_offset_ich_row_col_4 = V_offset_ich_row_col_3 + h_w_tile_stride;

#if SIMD_TYPE >= SIMDTYPE_AVX
											u_1_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1));
											u_1_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 8));
											u_1_16_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 16));
											u_1_24_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 24));
											u_1_32_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 32));
											u_1_40_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 40));
											u_1_48_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 48));
											u_1_56_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 56));
											u_1_0 = mm_cvtepi16_epi32(u_1_0_int16);
											u_1_8 = mm_cvtepi16_epi32(u_1_8_int16);
											u_1_16 = mm_cvtepi16_epi32(u_1_16_int16);
											u_1_24 = mm_cvtepi16_epi32(u_1_24_int16);
											u_1_32 = mm_cvtepi16_epi32(u_1_32_int16);
											u_1_40 = mm_cvtepi16_epi32(u_1_40_int16);
											u_1_48 = mm_cvtepi16_epi32(u_1_48_int16);
											u_1_56 = mm_cvtepi16_epi32(u_1_56_int16);

											u_2_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2));
											u_2_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 8));
											u_2_16_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 16));
											u_2_24_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 24));
											u_2_32_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 32));
											u_2_40_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 40));
											u_2_48_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 48));
											u_2_56_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 56));
											u_2_0 = mm_cvtepi16_epi32(u_2_0_int16);
											u_2_8 = mm_cvtepi16_epi32(u_2_8_int16);
											u_2_16 = mm_cvtepi16_epi32(u_2_16_int16);
											u_2_24 = mm_cvtepi16_epi32(u_2_24_int16);
											u_2_32 = mm_cvtepi16_epi32(u_2_32_int16);
											u_2_40 = mm_cvtepi16_epi32(u_2_40_int16);
											u_2_48 = mm_cvtepi16_epi32(u_2_48_int16);
											u_2_56 = mm_cvtepi16_epi32(u_2_56_int16);

											u_3_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3));
											u_3_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 8));
											u_3_16_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 16));
											u_3_24_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 24));
											u_3_32_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 32));
											u_3_40_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 40));
											u_3_48_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 48));
											u_3_56_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 56));
											u_3_0 = mm_cvtepi16_epi32(u_3_0_int16);
											u_3_8 = mm_cvtepi16_epi32(u_3_8_int16);
											u_3_16 = mm_cvtepi16_epi32(u_3_16_int16);
											u_3_24 = mm_cvtepi16_epi32(u_3_24_int16);
											u_3_32 = mm_cvtepi16_epi32(u_3_32_int16);
											u_3_40 = mm_cvtepi16_epi32(u_3_40_int16);
											u_3_48 = mm_cvtepi16_epi32(u_3_48_int16);
											u_3_56 = mm_cvtepi16_epi32(u_3_56_int16);

											u_4_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4));
											u_4_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 8));
											u_4_16_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 16));
											u_4_24_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 24));
											u_4_32_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 32));
											u_4_40_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 40));
											u_4_48_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 48));
											u_4_56_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 56));
											u_4_0 = mm_cvtepi16_epi32(u_4_0_int16);
											u_4_8 = mm_cvtepi16_epi32(u_4_8_int16);
											u_4_16 = mm_cvtepi16_epi32(u_4_16_int16);
											u_4_24 = mm_cvtepi16_epi32(u_4_24_int16);
											u_4_32 = mm_cvtepi16_epi32(u_4_32_int16);
											u_4_40 = mm_cvtepi16_epi32(u_4_40_int16);
											u_4_48 = mm_cvtepi16_epi32(u_4_48_int16);
											u_4_56 = mm_cvtepi16_epi32(u_4_56_int16);

											v_1_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1));
											v_1_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1 + 8));
											v_1_0 = mm_cvtepi16_epi32(v_1_0_int16);
											v_1_8 = mm_cvtepi16_epi32(v_1_8_int16);

											v_2_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_2));
											v_2_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_2 + 8));
											v_2_0 = mm_cvtepi16_epi32(v_2_0_int16);
											v_2_8 = mm_cvtepi16_epi32(v_2_8_int16);

											v_3_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_3));
											v_3_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_3 + 8));
											v_3_0 = mm_cvtepi16_epi32(v_3_0_int16);
											v_3_8 = mm_cvtepi16_epi32(v_3_8_int16);

											v_4_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_4));
											v_4_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_4 + 8));
											v_4_0 = mm_cvtepi16_epi32(v_4_0_int16);
											v_4_8 = mm_cvtepi16_epi32(v_4_8_int16);

											sum_1_0 = mm_add_epi32(mm_mullo_epi32(u_1_0, v_1_0), sum_1_0);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(u_1_8, v_1_8), sum_1_8);
											sum_1_0 = mm_add_epi32(mm_mullo_epi32(u_1_16, v_2_0), sum_1_0);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(u_1_24, v_2_8), sum_1_8);
											sum_1_0 = mm_add_epi32(mm_mullo_epi32(u_1_32, v_3_0), sum_1_0);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(u_1_40, v_3_8), sum_1_8);
											sum_1_0 = mm_add_epi32(mm_mullo_epi32(u_1_48, v_4_0), sum_1_0);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(u_1_56, v_4_8), sum_1_8);

											sum_2_0 = mm_add_epi32(mm_mullo_epi32(u_2_0, v_1_0), sum_2_0);
											sum_2_8 = mm_add_epi32(mm_mullo_epi32(u_2_8, v_1_8), sum_2_8);
											sum_2_0 = mm_add_epi32(mm_mullo_epi32(u_2_16, v_2_0), sum_2_0);
											sum_2_8 = mm_add_epi32(mm_mullo_epi32(u_2_24, v_2_8), sum_2_8);
											sum_2_0 = mm_add_epi32(mm_mullo_epi32(u_2_32, v_3_0), sum_2_0);
											sum_2_8 = mm_add_epi32(mm_mullo_epi32(u_2_40, v_3_8), sum_2_8);
											sum_2_0 = mm_add_epi32(mm_mullo_epi32(u_2_48, v_4_0), sum_2_0);
											sum_2_8 = mm_add_epi32(mm_mullo_epi32(u_2_56, v_4_8), sum_2_8);

											sum_3_0 = mm_add_epi32(mm_mullo_epi32(u_3_0, v_1_0), sum_3_0);
											sum_3_8 = mm_add_epi32(mm_mullo_epi32(u_3_8, v_1_8), sum_3_8);
											sum_3_0 = mm_add_epi32(mm_mullo_epi32(u_3_16, v_2_0), sum_3_0);
											sum_3_8 = mm_add_epi32(mm_mullo_epi32(u_3_24, v_2_8), sum_3_8);
											sum_3_0 = mm_add_epi32(mm_mullo_epi32(u_3_32, v_3_0), sum_3_0);
											sum_3_8 = mm_add_epi32(mm_mullo_epi32(u_3_40, v_3_8), sum_3_8);
											sum_3_0 = mm_add_epi32(mm_mullo_epi32(u_3_48, v_4_0), sum_3_0);
											sum_3_8 = mm_add_epi32(mm_mullo_epi32(u_3_56, v_4_8), sum_3_8);

											sum_4_0 = mm_add_epi32(mm_mullo_epi32(u_4_0, v_1_0), sum_4_0);
											sum_4_8 = mm_add_epi32(mm_mullo_epi32(u_4_8, v_1_8), sum_4_8);
											sum_4_0 = mm_add_epi32(mm_mullo_epi32(u_4_16, v_2_0), sum_4_0);
											sum_4_8 = mm_add_epi32(mm_mullo_epi32(u_4_24, v_2_8), sum_4_8);
											sum_4_0 = mm_add_epi32(mm_mullo_epi32(u_4_32, v_3_0), sum_4_0);
											sum_4_8 = mm_add_epi32(mm_mullo_epi32(u_4_40, v_3_8), sum_4_8);
											sum_4_0 = mm_add_epi32(mm_mullo_epi32(u_4_48, v_4_0), sum_4_0);
											sum_4_8 = mm_add_epi32(mm_mullo_epi32(u_4_56, v_4_8), sum_4_8);

#elif SIMD_TYPE >= SIMDTYPE_SSE
											u_1_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1));
											u_1_4_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 4));
											u_1_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 8));
											u_1_12_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 12));
											u_1_16_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 16));
											u_1_20_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 20));
											u_1_24_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 24));
											u_1_28_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 28));
											u_1_32_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 32));
											u_1_36_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 36));
											u_1_40_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 40));
											u_1_44_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 44));
											u_1_48_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 48));
											u_1_52_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 52));
											u_1_56_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 56));
											u_1_60_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 60));
											u_1_0 = mm_cvtepi16_epi32(u_1_0_int16);
											u_1_4 = mm_cvtepi16_epi32(u_1_4_int16);
											u_1_8 = mm_cvtepi16_epi32(u_1_8_int16);
											u_1_12 = mm_cvtepi16_epi32(u_1_12_int16);
											u_1_16 = mm_cvtepi16_epi32(u_1_16_int16);
											u_1_20 = mm_cvtepi16_epi32(u_1_20_int16);
											u_1_24 = mm_cvtepi16_epi32(u_1_24_int16);
											u_1_28 = mm_cvtepi16_epi32(u_1_28_int16);
											u_1_32 = mm_cvtepi16_epi32(u_1_32_int16);
											u_1_36 = mm_cvtepi16_epi32(u_1_36_int16);
											u_1_40 = mm_cvtepi16_epi32(u_1_40_int16);
											u_1_44 = mm_cvtepi16_epi32(u_1_44_int16);
											u_1_48 = mm_cvtepi16_epi32(u_1_48_int16);
											u_1_52 = mm_cvtepi16_epi32(u_1_52_int16);
											u_1_56 = mm_cvtepi16_epi32(u_1_56_int16);
											u_1_60 = mm_cvtepi16_epi32(u_1_60_int16);

											u_2_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2));
											u_2_4_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 4));
											u_2_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 8));
											u_2_12_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 12));
											u_2_16_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 16));
											u_2_20_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 20));
											u_2_24_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 24));
											u_2_28_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 28));
											u_2_32_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 32));
											u_2_36_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 36));
											u_2_40_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 40));
											u_2_44_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 44));
											u_2_48_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 48));
											u_2_52_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 52));
											u_2_56_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 56));
											u_2_60_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 60));
											u_2_0 = mm_cvtepi16_epi32(u_2_0_int16);
											u_2_4 = mm_cvtepi16_epi32(u_2_4_int16);
											u_2_8 = mm_cvtepi16_epi32(u_2_8_int16);
											u_2_12 = mm_cvtepi16_epi32(u_2_12_int16);
											u_2_16 = mm_cvtepi16_epi32(u_2_16_int16);
											u_2_20 = mm_cvtepi16_epi32(u_2_20_int16);
											u_2_24 = mm_cvtepi16_epi32(u_2_24_int16);
											u_2_28 = mm_cvtepi16_epi32(u_2_28_int16);
											u_2_32 = mm_cvtepi16_epi32(u_2_32_int16);
											u_2_36 = mm_cvtepi16_epi32(u_2_36_int16);
											u_2_40 = mm_cvtepi16_epi32(u_2_40_int16);
											u_2_44 = mm_cvtepi16_epi32(u_2_44_int16);
											u_2_48 = mm_cvtepi16_epi32(u_2_48_int16);
											u_2_52 = mm_cvtepi16_epi32(u_2_52_int16);
											u_2_56 = mm_cvtepi16_epi32(u_2_56_int16);
											u_2_60 = mm_cvtepi16_epi32(u_2_60_int16);

											u_3_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3));
											u_3_4_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 4));
											u_3_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 8));
											u_3_12_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 12));
											u_3_16_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 16));
											u_3_20_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 20));
											u_3_24_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 24));
											u_3_28_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 28));
											u_3_32_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 32));
											u_3_36_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 36));
											u_3_40_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 40));
											u_3_44_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 44));
											u_3_48_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 48));
											u_3_52_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 52));
											u_3_56_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 56));
											u_3_60_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 60));
											u_3_0 = mm_cvtepi16_epi32(u_3_0_int16);
											u_3_4 = mm_cvtepi16_epi32(u_3_4_int16);
											u_3_8 = mm_cvtepi16_epi32(u_3_8_int16);
											u_3_12 = mm_cvtepi16_epi32(u_3_12_int16);
											u_3_16 = mm_cvtepi16_epi32(u_3_16_int16);
											u_3_20 = mm_cvtepi16_epi32(u_3_20_int16);
											u_3_24 = mm_cvtepi16_epi32(u_3_24_int16);
											u_3_28 = mm_cvtepi16_epi32(u_3_28_int16);
											u_3_32 = mm_cvtepi16_epi32(u_3_32_int16);
											u_3_36 = mm_cvtepi16_epi32(u_3_36_int16);
											u_3_40 = mm_cvtepi16_epi32(u_3_40_int16);
											u_3_44 = mm_cvtepi16_epi32(u_3_44_int16);
											u_3_48 = mm_cvtepi16_epi32(u_3_48_int16);
											u_3_52 = mm_cvtepi16_epi32(u_3_52_int16);
											u_3_56 = mm_cvtepi16_epi32(u_3_56_int16);
											u_3_60 = mm_cvtepi16_epi32(u_3_60_int16);

											u_4_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4));
											u_4_4_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 4));
											u_4_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 8));
											u_4_12_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 12));
											u_4_16_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 16));
											u_4_20_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 20));
											u_4_24_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 24));
											u_4_28_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 28));
											u_4_32_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 32));
											u_4_36_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 36));
											u_4_40_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 40));
											u_4_44_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 44));
											u_4_48_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 48));
											u_4_52_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 52));
											u_4_56_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 56));
											u_4_60_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 60));
											u_4_0 = mm_cvtepi16_epi32(u_4_0_int16);
											u_4_4 = mm_cvtepi16_epi32(u_4_4_int16);
											u_4_8 = mm_cvtepi16_epi32(u_4_8_int16);
											u_4_12 = mm_cvtepi16_epi32(u_4_12_int16);
											u_4_16 = mm_cvtepi16_epi32(u_4_16_int16);
											u_4_20 = mm_cvtepi16_epi32(u_4_20_int16);
											u_4_24 = mm_cvtepi16_epi32(u_4_24_int16);
											u_4_28 = mm_cvtepi16_epi32(u_4_28_int16);
											u_4_32 = mm_cvtepi16_epi32(u_4_32_int16);
											u_4_36 = mm_cvtepi16_epi32(u_4_36_int16);
											u_4_40 = mm_cvtepi16_epi32(u_4_40_int16);
											u_4_44 = mm_cvtepi16_epi32(u_4_44_int16);
											u_4_48 = mm_cvtepi16_epi32(u_4_48_int16);
											u_4_52 = mm_cvtepi16_epi32(u_4_52_int16);
											u_4_56 = mm_cvtepi16_epi32(u_4_56_int16);
											u_4_60 = mm_cvtepi16_epi32(u_4_60_int16);

											v_1_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1));
											v_1_4_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1 + 4));
											v_1_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1 + 8));
											v_1_12_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1 + 12));
											v_1_0 = mm_cvtepi16_epi32(v_1_0_int16);
											v_1_4 = mm_cvtepi16_epi32(v_1_4_int16);
											v_1_8 = mm_cvtepi16_epi32(v_1_8_int16);
											v_1_12 = mm_cvtepi16_epi32(v_1_12_int16);

											v_2_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_2));
											v_2_4_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_2 + 4));
											v_2_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_2 + 8));
											v_2_12_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_2 + 12));
											v_2_0 = mm_cvtepi16_epi32(v_2_0_int16);
											v_2_4 = mm_cvtepi16_epi32(v_2_4_int16);
											v_2_8 = mm_cvtepi16_epi32(v_2_8_int16);
											v_2_12 = mm_cvtepi16_epi32(v_2_12_int16);

											v_3_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_3));
											v_3_4_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_3 + 4));
											v_3_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_3 + 8));
											v_3_12_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_3 + 12));
											v_3_0 = mm_cvtepi16_epi32(v_3_0_int16);
											v_3_4 = mm_cvtepi16_epi32(v_3_4_int16);
											v_3_8 = mm_cvtepi16_epi32(v_3_8_int16);
											v_3_12 = mm_cvtepi16_epi32(v_3_12_int16);

											v_4_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_4));
											v_4_4_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_4 + 4));
											v_4_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_4 + 8));
											v_4_12_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_4 + 12));
											v_4_0 = mm_cvtepi16_epi32(v_4_0_int16);
											v_4_4 = mm_cvtepi16_epi32(v_4_4_int16);
											v_4_8 = mm_cvtepi16_epi32(v_4_8_int16);
											v_4_12 = mm_cvtepi16_epi32(v_4_12_int16);

											sum_1_0 = mm_add_epi32(mm_mullo_epi32(v_1_0, u_1_0), sum_1_0);
											sum_1_4 = mm_add_epi32(mm_mullo_epi32(v_1_4, u_1_4), sum_1_4);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(v_1_8, u_1_8), sum_1_8);
											sum_1_12 = mm_add_epi32(mm_mullo_epi32(v_1_12, u_1_12), sum_1_12);
											sum_1_0 = mm_add_epi32(mm_mullo_epi32(v_2_0, u_1_16), sum_1_0);
											sum_1_4 = mm_add_epi32(mm_mullo_epi32(v_2_4, u_1_20), sum_1_4);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(v_2_8, u_1_24), sum_1_8);
											sum_1_12 = mm_add_epi32(mm_mullo_epi32(v_2_12, u_1_28), sum_1_12);
											sum_1_0 = mm_add_epi32(mm_mullo_epi32(v_3_0, u_1_32), sum_1_0);
											sum_1_4 = mm_add_epi32(mm_mullo_epi32(v_3_4, u_1_36), sum_1_4);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(v_3_8, u_1_40), sum_1_8);
											sum_1_12 = mm_add_epi32(mm_mullo_epi32(v_3_12, u_1_44), sum_1_12);
											sum_1_0 = mm_add_epi32(mm_mullo_epi32(v_4_0, u_1_48), sum_1_0);
											sum_1_4 = mm_add_epi32(mm_mullo_epi32(v_4_4, u_1_52), sum_1_4);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(v_4_8, u_1_56), sum_1_8);
											sum_1_12 = mm_add_epi32(mm_mullo_epi32(v_4_12, u_1_60), sum_1_12);

											sum_2_0 = mm_add_epi32(mm_mullo_epi32(v_1_0, u_2_0), sum_2_0);
											sum_2_4 = mm_add_epi32(mm_mullo_epi32(v_1_4, u_2_4), sum_2_4);
											sum_2_8 = mm_add_epi32(mm_mullo_epi32(v_1_8, u_2_8), sum_2_8);
											sum_2_12 = mm_add_epi32(mm_mullo_epi32(v_1_12, u_2_12), sum_2_12);
											sum_2_0 = mm_add_epi32(mm_mullo_epi32(v_2_0, u_2_16), sum_2_0);
											sum_2_4 = mm_add_epi32(mm_mullo_epi32(v_2_4, u_2_20), sum_2_4);
											sum_2_8 = mm_add_epi32(mm_mullo_epi32(v_2_8, u_2_24), sum_2_8);
											sum_2_12 = mm_add_epi32(mm_mullo_epi32(v_2_12, u_2_28), sum_2_12);
											sum_2_0 = mm_add_epi32(mm_mullo_epi32(v_3_0, u_2_32), sum_2_0);
											sum_2_4 = mm_add_epi32(mm_mullo_epi32(v_3_4, u_2_36), sum_2_4);
											sum_2_8 = mm_add_epi32(mm_mullo_epi32(v_3_8, u_2_40), sum_2_8);
											sum_2_12 = mm_add_epi32(mm_mullo_epi32(v_3_12, u_2_44), sum_2_12);
											sum_2_0 = mm_add_epi32(mm_mullo_epi32(v_4_0, u_2_48), sum_2_0);
											sum_2_4 = mm_add_epi32(mm_mullo_epi32(v_4_4, u_2_52), sum_2_4);
											sum_2_8 = mm_add_epi32(mm_mullo_epi32(v_4_8, u_2_56), sum_2_8);
											sum_2_12 = mm_add_epi32(mm_mullo_epi32(v_4_12, u_2_60), sum_2_12);

											sum_3_0 = mm_add_epi32(mm_mullo_epi32(v_1_0, u_3_0), sum_3_0);
											sum_3_4 = mm_add_epi32(mm_mullo_epi32(v_1_4, u_3_4), sum_3_4);
											sum_3_8 = mm_add_epi32(mm_mullo_epi32(v_1_8, u_3_8), sum_3_8);
											sum_3_12 = mm_add_epi32(mm_mullo_epi32(v_1_12, u_3_12), sum_3_12);
											sum_3_0 = mm_add_epi32(mm_mullo_epi32(v_2_0, u_3_16), sum_3_0);
											sum_3_4 = mm_add_epi32(mm_mullo_epi32(v_2_4, u_3_20), sum_3_4);
											sum_3_8 = mm_add_epi32(mm_mullo_epi32(v_2_8, u_3_24), sum_3_8);
											sum_3_12 = mm_add_epi32(mm_mullo_epi32(v_2_12, u_3_28), sum_3_12);
											sum_3_0 = mm_add_epi32(mm_mullo_epi32(v_3_0, u_3_32), sum_3_0);
											sum_3_4 = mm_add_epi32(mm_mullo_epi32(v_3_4, u_3_36), sum_3_4);
											sum_3_8 = mm_add_epi32(mm_mullo_epi32(v_3_8, u_3_40), sum_3_8);
											sum_3_12 = mm_add_epi32(mm_mullo_epi32(v_3_12, u_3_44), sum_3_12);
											sum_3_0 = mm_add_epi32(mm_mullo_epi32(v_4_0, u_3_48), sum_3_0);
											sum_3_4 = mm_add_epi32(mm_mullo_epi32(v_4_4, u_3_52), sum_3_4);
											sum_3_8 = mm_add_epi32(mm_mullo_epi32(v_4_8, u_3_56), sum_3_8);
											sum_3_12 = mm_add_epi32(mm_mullo_epi32(v_4_12, u_3_60), sum_3_12);

											sum_4_0 = mm_add_epi32(mm_mullo_epi32(v_1_0, u_4_0), sum_4_0);
											sum_4_4 = mm_add_epi32(mm_mullo_epi32(v_1_4, u_4_4), sum_4_4);
											sum_4_8 = mm_add_epi32(mm_mullo_epi32(v_1_8, u_4_8), sum_4_8);
											sum_4_12 = mm_add_epi32(mm_mullo_epi32(v_1_12, u_4_12), sum_4_12);
											sum_4_0 = mm_add_epi32(mm_mullo_epi32(v_2_0, u_4_16), sum_4_0);
											sum_4_4 = mm_add_epi32(mm_mullo_epi32(v_2_4, u_4_20), sum_4_4);
											sum_4_8 = mm_add_epi32(mm_mullo_epi32(v_2_8, u_4_24), sum_4_8);
											sum_4_12 = mm_add_epi32(mm_mullo_epi32(v_2_12, u_4_28), sum_4_12);
											sum_4_0 = mm_add_epi32(mm_mullo_epi32(v_3_0, u_4_32), sum_4_0);
											sum_4_4 = mm_add_epi32(mm_mullo_epi32(v_3_4, u_4_36), sum_4_4);
											sum_4_8 = mm_add_epi32(mm_mullo_epi32(v_3_8, u_4_40), sum_4_8);
											sum_4_12 = mm_add_epi32(mm_mullo_epi32(v_3_12, u_4_44), sum_4_12);
											sum_4_0 = mm_add_epi32(mm_mullo_epi32(v_4_0, u_4_48), sum_4_0);
											sum_4_4 = mm_add_epi32(mm_mullo_epi32(v_4_4, u_4_52), sum_4_4);
											sum_4_8 = mm_add_epi32(mm_mullo_epi32(v_4_8, u_4_56), sum_4_8);
											sum_4_12 = mm_add_epi32(mm_mullo_epi32(v_4_12, u_4_60), sum_4_12);
#else
											for (int i = 0; i < tile_length_; i++)
											{
												mult_data1[i] += U_int16_data[U_offset_och_ich_1 + i] * V_int16_data[V_offset_ich_row_col_1 + i];
												mult_data1[i] += U_int16_data[U_offset_och_ich_1 + tile_length_ + i] * V_int16_data[V_offset_ich_row_col_2 + i];
												mult_data1[i] += U_int16_data[U_offset_och_ich_1 + 2 * tile_length_ + i] * V_int16_data[V_offset_ich_row_col_3 + i];
												mult_data1[i] += U_int16_data[U_offset_och_ich_1 + 3 * tile_length_ + i] * V_int16_data[V_offset_ich_row_col_4 + i];

												mult_data2[i] += U_int16_data[U_offset_och_ich_2 + i] * V_int16_data[V_offset_ich_row_col_1 + i];
												mult_data2[i] += U_int16_data[U_offset_och_ich_2 + tile_length_ + i] * V_int16_data[V_offset_ich_row_col_2 + i];
												mult_data2[i] += U_int16_data[U_offset_och_ich_2 + 2 * tile_length_ + i] * V_int16_data[V_offset_ich_row_col_3 + i];
												mult_data2[i] += U_int16_data[U_offset_och_ich_2 + 3 * tile_length_ + i] * V_int16_data[V_offset_ich_row_col_4 + i];

												mult_data3[i] += U_int16_data[U_offset_och_ich_3 + i] * V_int16_data[V_offset_ich_row_col_1 + i];
												mult_data3[i] += U_int16_data[U_offset_och_ich_3 + tile_length_ + i] * V_int16_data[V_offset_ich_row_col_2 + i];
												mult_data3[i] += U_int16_data[U_offset_och_ich_3 + 2 * tile_length_ + i] * V_int16_data[V_offset_ich_row_col_3 + i];
												mult_data3[i] += U_int16_data[U_offset_och_ich_3 + 3 * tile_length_ + i] * V_int16_data[V_offset_ich_row_col_4 + i];

												mult_data4[i] += U_int16_data[U_offset_och_ich_4 + i] * V_int16_data[V_offset_ich_row_col_1 + i];
												mult_data4[i] += U_int16_data[U_offset_och_ich_4 + tile_length_ + i] * V_int16_data[V_offset_ich_row_col_2 + i];
												mult_data4[i] += U_int16_data[U_offset_och_ich_4 + 2 * tile_length_ + i] * V_int16_data[V_offset_ich_row_col_3 + i];
												mult_data4[i] += U_int16_data[U_offset_och_ich_4 + 3 * tile_length_ + i] * V_int16_data[V_offset_ich_row_col_4 + i];
											}
#endif
										}

										for (; ich < input_Channel_; ich++)
										{
											int offset = ich * tile_length_;
											int U_offset_och_ich_1 = U_offset_och_1 + offset;
											int U_offset_och_ich_2 = U_offset_och_2 + offset;
											int U_offset_och_ich_3 = U_offset_och_3 + offset;
											int U_offset_och_ich_4 = U_offset_och_4 + offset;

											int V_offset_ich_row_col_1 = V_offset_row_col + ich * h_w_tile_stride;

#if SIMD_TYPE >= SIMDTYPE_AVX
											u_1_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1));
											u_1_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 8));
											u_1_0 = mm_cvtepi16_epi32(u_1_0_int16);
											u_1_8 = mm_cvtepi16_epi32(u_1_8_int16);

											u_2_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2));
											u_2_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 8));
											u_2_0 = mm_cvtepi16_epi32(u_2_0_int16);
											u_2_8 = mm_cvtepi16_epi32(u_2_8_int16);

											u_3_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3));
											u_3_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 8));
											u_3_0 = mm_cvtepi16_epi32(u_3_0_int16);
											u_3_8 = mm_cvtepi16_epi32(u_3_8_int16);

											u_4_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4));
											u_4_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 8));
											u_4_0 = mm_cvtepi16_epi32(u_4_0_int16);
											u_4_8 = mm_cvtepi16_epi32(u_4_8_int16);

											v_1_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1));
											v_1_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1 + 8));
											v_1_0 = mm_cvtepi16_epi32(v_1_0_int16);
											v_1_8 = mm_cvtepi16_epi32(v_1_8_int16);

											sum_1_0 = mm_add_epi32(mm_mullo_epi32(v_1_0, u_1_0), sum_1_0);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(v_1_8, u_1_8), sum_1_8);
											sum_2_0 = mm_add_epi32(mm_mullo_epi32(v_1_0, u_2_0), sum_2_0);
											sum_2_8 = mm_add_epi32(mm_mullo_epi32(v_1_8, u_2_8), sum_2_8);
											sum_3_0 = mm_add_epi32(mm_mullo_epi32(v_1_0, u_3_0), sum_3_0);
											sum_3_8 = mm_add_epi32(mm_mullo_epi32(v_1_8, u_3_8), sum_3_8);
											sum_4_0 = mm_add_epi32(mm_mullo_epi32(v_1_0, u_4_0), sum_4_0);
											sum_4_8 = mm_add_epi32(mm_mullo_epi32(v_1_8, u_4_8), sum_4_8);
#elif SIMD_TYPE >= SIMDTYPE_SSE
											u_1_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1));
											u_1_4_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 4));
											u_1_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 8));
											u_1_12_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 12));
											u_1_0 = mm_cvtepi16_epi32(u_1_0_int16);
											u_1_4 = mm_cvtepi16_epi32(u_1_4_int16);
											u_1_8 = mm_cvtepi16_epi32(u_1_8_int16);
											u_1_12 = mm_cvtepi16_epi32(u_1_12_int16);

											u_2_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2));
											u_2_4_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 4));
											u_2_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 8));
											u_2_12_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_2 + 12));
											u_2_0 = mm_cvtepi16_epi32(u_2_0_int16);
											u_2_4 = mm_cvtepi16_epi32(u_2_4_int16);
											u_2_8 = mm_cvtepi16_epi32(u_2_8_int16);
											u_2_12 = mm_cvtepi16_epi32(u_2_12_int16);

											u_3_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3));
											u_3_4_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 4));
											u_3_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 8));
											u_3_12_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_3 + 12));
											u_3_0 = mm_cvtepi16_epi32(u_3_0_int16);
											u_3_4 = mm_cvtepi16_epi32(u_3_4_int16);
											u_3_8 = mm_cvtepi16_epi32(u_3_8_int16);
											u_3_12 = mm_cvtepi16_epi32(u_3_12_int16);

											u_4_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4));
											u_4_4_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 4));
											u_4_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 8));
											u_4_12_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_4 + 12));
											u_4_0 = mm_cvtepi16_epi32(u_4_0_int16);
											u_4_4 = mm_cvtepi16_epi32(u_4_4_int16);
											u_4_8 = mm_cvtepi16_epi32(u_4_8_int16);
											u_4_12 = mm_cvtepi16_epi32(u_4_12_int16);

											v_1_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1));
											v_1_4_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1 + 4));
											v_1_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1 + 8));
											v_1_12_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1 + 12));
											v_1_0 = mm_cvtepi16_epi32(v_1_0_int16);
											v_1_4 = mm_cvtepi16_epi32(v_1_4_int16);
											v_1_8 = mm_cvtepi16_epi32(v_1_8_int16);
											v_1_12 = mm_cvtepi16_epi32(v_1_12_int16);

											sum_1_0 = mm_add_epi32(mm_mullo_epi32(v_1_0, u_1_0), sum_1_0);
											sum_1_4 = mm_add_epi32(mm_mullo_epi32(v_1_4, u_1_4), sum_1_4);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(v_1_8, u_1_8), sum_1_8);
											sum_1_12 = mm_add_epi32(mm_mullo_epi32(v_1_12, u_1_12), sum_1_12);

											sum_2_0 = mm_add_epi32(mm_mullo_epi32(v_1_0, u_2_0), sum_2_0);
											sum_2_4 = mm_add_epi32(mm_mullo_epi32(v_1_4, u_2_4), sum_2_4);
											sum_2_8 = mm_add_epi32(mm_mullo_epi32(v_1_8, u_2_8), sum_2_8);
											sum_2_12 = mm_add_epi32(mm_mullo_epi32(v_1_12, u_2_12), sum_2_12);

											sum_3_0 = mm_add_epi32(mm_mullo_epi32(v_1_0, u_3_0), sum_3_0);
											sum_3_4 = mm_add_epi32(mm_mullo_epi32(v_1_4, u_3_4), sum_3_4);
											sum_3_8 = mm_add_epi32(mm_mullo_epi32(v_1_8, u_3_8), sum_3_8);
											sum_3_12 = mm_add_epi32(mm_mullo_epi32(v_1_12, u_3_12), sum_3_12);

											sum_4_0 = mm_add_epi32(mm_mullo_epi32(v_1_0, u_4_0), sum_4_0);
											sum_4_4 = mm_add_epi32(mm_mullo_epi32(v_1_4, u_4_4), sum_4_4);
											sum_4_8 = mm_add_epi32(mm_mullo_epi32(v_1_8, u_4_8), sum_4_8);
											sum_4_12 = mm_add_epi32(mm_mullo_epi32(v_1_12, u_4_12), sum_4_12);
#else
											for (int i = 0; i < tile_length_; i++)
											{
												mult_data1[i] += U_int16_data[U_offset_och_ich_1 + i] * V_int16_data[V_offset_ich_row_col_1 + i];
												mult_data2[i] += U_int16_data[U_offset_och_ich_2 + i] * V_int16_data[V_offset_ich_row_col_1 + i];
												mult_data3[i] += U_int16_data[U_offset_och_ich_3 + i] * V_int16_data[V_offset_ich_row_col_1 + i];
												mult_data4[i] += U_int16_data[U_offset_och_ich_4 + i] * V_int16_data[V_offset_ich_row_col_1 + i];
											}
#endif
										}		

#if SIMD_TYPE >= SIMDTYPE_AVX
										mm_store_si((mm_typei*)mult_data1, sum_1_0);
										mm_store_si((mm_typei*)(mult_data1 + 8), sum_1_8);
										mm_store_si((mm_typei*)mult_data2, sum_2_0);
										mm_store_si((mm_typei*)(mult_data2 + 8), sum_2_8);
										mm_store_si((mm_typei*)mult_data3, sum_3_0);
										mm_store_si((mm_typei*)(mult_data3 + 8), sum_3_8);
										mm_store_si((mm_typei*)mult_data4, sum_4_0);
										mm_store_si((mm_typei*)(mult_data4 + 8), sum_4_8);
#elif SIMD_TYPE >= SIMDTYPE_SSE
										mm_store_si((mm_typei*)mult_data1, sum_1_0);
										mm_store_si((mm_typei*)(mult_data1 + 4), sum_1_4);
										mm_store_si((mm_typei*)(mult_data1 + 8), sum_1_8);
										mm_store_si((mm_typei*)(mult_data1 + 12), sum_1_12);
										mm_store_si((mm_typei*)mult_data2, sum_2_0);
										mm_store_si((mm_typei*)(mult_data2 + 4), sum_2_4);
										mm_store_si((mm_typei*)(mult_data2 + 8), sum_2_8);
										mm_store_si((mm_typei*)(mult_data2 + 12), sum_2_12);
										mm_store_si((mm_typei*)mult_data3, sum_3_0);
										mm_store_si((mm_typei*)(mult_data3 + 4), sum_3_4);
										mm_store_si((mm_typei*)(mult_data3 + 8), sum_3_8);
										mm_store_si((mm_typei*)(mult_data3 + 12), sum_3_12);
										mm_store_si((mm_typei*)mult_data4, sum_4_0);
										mm_store_si((mm_typei*)(mult_data4 + 4), sum_4_4);
										mm_store_si((mm_typei*)(mult_data4 + 8), sum_4_8);
										mm_store_si((mm_typei*)(mult_data4 + 12), sum_4_12);
#endif

										calculate_ATmA23(mult_data1, result1);
										calculate_ATmA23(mult_data2, result2);
										calculate_ATmA23(mult_data3, result3);
										calculate_ATmA23(mult_data4, result4);

										int row_in_output_data = i / w_tile_num_ * m_;
										int col_in_output_data = i % w_tile_num_* m_;
										int top_offset_row_col = row_in_output_data * output_dim_w_ + col_in_output_data;
										int top_offset_num_och_row_col_1 = top_offset_num_och_1 + top_offset_row_col;
										int top_offset_num_och_row_col_2 = top_offset_num_och_2 + top_offset_row_col;
										int top_offset_num_och_row_col_3 = top_offset_num_och_3 + top_offset_row_col;
										int top_offset_num_och_row_col_4 = top_offset_num_och_4 + top_offset_row_col;

										for (int row = 0; row < m_; row++)
										{
											int result_offset_row = row * m_;
											for (int col = 0; col < m_; col++)
											{
												top_int32_data[top_offset_num_och_row_col_1 + col] = result1[result_offset_row + col];
												top_int32_data[top_offset_num_och_row_col_2 + col] = result2[result_offset_row + col];
												top_int32_data[top_offset_num_och_row_col_3 + col] = result3[result_offset_row + col];
												top_int32_data[top_offset_num_och_row_col_4 + col] = result4[result_offset_row + col];
											}
											top_offset_num_och_row_col_1 += output_dim_w_;
											top_offset_num_och_row_col_2 += output_dim_w_;
											top_offset_num_och_row_col_3 += output_dim_w_;
											top_offset_num_och_row_col_4 += output_dim_w_;
										}
									}
								}

								for (int och = remain_outch_start; och < output_Channel_; och++)
								{
#if SIMD_TYPE >= SIMDTYPE_AVX
									mm_typei sum_1_0;
									mm_typei sum_1_8;
									mm_typei u_1_0;
									mm_typei u_1_8;
									mm_typei u_1_16;
									mm_typei u_1_24;
									mm_typei u_1_32;
									mm_typei u_1_40;
									mm_typei u_1_48;
									mm_typei u_1_56;
									__m128i u_1_0_int16;
									__m128i u_1_8_int16;
									__m128i u_1_16_int16;
									__m128i u_1_24_int16;
									__m128i u_1_32_int16;
									__m128i u_1_40_int16;
									__m128i u_1_48_int16;
									__m128i u_1_56_int16;									
									mm_typei v_1_0;
									mm_typei v_1_8;
									mm_typei v_2_0;
									mm_typei v_2_8;
									mm_typei v_3_0;
									mm_typei v_3_8;
									mm_typei v_4_0;
									mm_typei v_4_8;
									__m128i v_1_0_int16;
									__m128i v_1_8_int16;
									__m128i v_2_0_int16;
									__m128i v_2_8_int16;
									__m128i v_3_0_int16;
									__m128i v_3_8_int16;
									__m128i v_4_0_int16;
									__m128i v_4_8_int16;
#elif SIMD_TYPE >= SIMDTYPE_SSE
									mm_typei sum_1_0;
									mm_typei sum_1_4;
									mm_typei sum_1_8;
									mm_typei sum_1_12;
									mm_typei u_1_0;
									mm_typei u_1_4;
									mm_typei u_1_8;
									mm_typei u_1_12;
									mm_typei u_1_16;
									mm_typei u_1_20;
									mm_typei u_1_24;
									mm_typei u_1_28;
									mm_typei u_1_32;
									mm_typei u_1_36;
									mm_typei u_1_40;
									mm_typei u_1_44;
									mm_typei u_1_48;
									mm_typei u_1_52;
									mm_typei u_1_56;
									mm_typei u_1_60;
									mm_typei u_1_0_int16;
									mm_typei u_1_4_int16;
									mm_typei u_1_8_int16;
									mm_typei u_1_12_int16;
									mm_typei u_1_16_int16;
									mm_typei u_1_20_int16;
									mm_typei u_1_24_int16;
									mm_typei u_1_28_int16;
									mm_typei u_1_32_int16;
									mm_typei u_1_36_int16;
									mm_typei u_1_40_int16;
									mm_typei u_1_44_int16;
									mm_typei u_1_48_int16;
									mm_typei u_1_52_int16;
									mm_typei u_1_56_int16;
									mm_typei u_1_60_int16;
									mm_typei v_1_0;
									mm_typei v_1_4;
									mm_typei v_1_8;
									mm_typei v_1_12;
									mm_typei v_2_0;
									mm_typei v_2_4;
									mm_typei v_2_8;
									mm_typei v_2_12;
									mm_typei v_3_0;
									mm_typei v_3_4;
									mm_typei v_3_8;
									mm_typei v_3_12;
									mm_typei v_4_0;
									mm_typei v_4_4;
									mm_typei v_4_8;
									mm_typei v_4_12;
									mm_typei v_1_0_int16;
									mm_typei v_1_4_int16;
									mm_typei v_1_8_int16;
									mm_typei v_1_12_int16;
									mm_typei v_2_0_int16;
									mm_typei v_2_4_int16;
									mm_typei v_2_8_int16;
									mm_typei v_2_12_int16;
									mm_typei v_3_0_int16;
									mm_typei v_3_4_int16;
									mm_typei v_3_8_int16;
									mm_typei v_3_12_int16;
									mm_typei v_4_0_int16;
									mm_typei v_4_4_int16;
									mm_typei v_4_8_int16;
									mm_typei v_4_12_int16;
#endif

									int U_offset_och_1 = och * U_offset_single_och;

									float bias1 = bias_data[och];
									int result1[4];
									int top_offset_num_och_1 = top_offset_num + och * output_spatial_dim_;

									for (int i = 0; i < total_tile_num; i++)
									{
										int mult_data1[16] = { 0 };
										int V_offset_row_col = i * tile_length_;

#if SIMD_TYPE >= SIMDTYPE_AVX
										sum_1_0 = mm_setzero_si();
										sum_1_8 = mm_setzero_si();
#elif SIMD_TYPE >= SIMDTYPE_SSE
										sum_1_0 = mm_setzero_si();
										sum_1_4 = mm_setzero_si();
										sum_1_8 = mm_setzero_si();
										sum_1_12 = mm_setzero_si();
#endif // SIMD_TYPE >= SIMDTYPE_AVX
										int ich = 0;
										for (; ich + 3 < input_Channel_; ich += 4)
										{
											int U_offset_och_ich_1 = U_offset_och_1 + ich * tile_length_;

											int V_offset_ich_row_col_1 = V_offset_row_col + ich * h_w_tile_stride;
											int V_offset_ich_row_col_2 = V_offset_ich_row_col_1 + h_w_tile_stride;
											int V_offset_ich_row_col_3 = V_offset_ich_row_col_2 + h_w_tile_stride;
											int V_offset_ich_row_col_4 = V_offset_ich_row_col_3 + h_w_tile_stride;

#if SIMD_TYPE >= SIMDTYPE_AVX
											u_1_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1));
											u_1_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 8));
											u_1_16_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 16));
											u_1_24_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 24));
											u_1_32_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 32));
											u_1_40_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 40));
											u_1_48_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 48));
											u_1_56_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 56));
											u_1_0 = mm_cvtepi16_epi32(u_1_0_int16);
											u_1_8 = mm_cvtepi16_epi32(u_1_8_int16);
											u_1_16 = mm_cvtepi16_epi32(u_1_16_int16);
											u_1_24 = mm_cvtepi16_epi32(u_1_24_int16);
											u_1_32 = mm_cvtepi16_epi32(u_1_32_int16);
											u_1_40 = mm_cvtepi16_epi32(u_1_40_int16);
											u_1_48 = mm_cvtepi16_epi32(u_1_48_int16);
											u_1_56 = mm_cvtepi16_epi32(u_1_56_int16);

											v_1_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1));
											v_1_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1 + 8));
											v_1_0 = mm_cvtepi16_epi32(v_1_0_int16);
											v_1_8 = mm_cvtepi16_epi32(v_1_8_int16);
											v_2_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_2));
											v_2_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_2 + 8));
											v_2_0 = mm_cvtepi16_epi32(v_2_0_int16);
											v_2_8 = mm_cvtepi16_epi32(v_2_8_int16);
											v_3_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_3));
											v_3_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_3 + 8));
											v_3_0 = mm_cvtepi16_epi32(v_3_0_int16);
											v_3_8 = mm_cvtepi16_epi32(v_3_8_int16);
											v_4_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_4));
											v_4_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_4 + 8));
											v_4_0 = mm_cvtepi16_epi32(v_4_0_int16);
											v_4_8 = mm_cvtepi16_epi32(v_4_8_int16);

											sum_1_0 = mm_add_epi32(mm_mullo_epi32(v_1_0, u_1_0), sum_1_0);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(v_1_8, u_1_8), sum_1_8);
											sum_1_0 = mm_add_epi32(mm_mullo_epi32(v_2_0, u_1_16), sum_1_0);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(v_2_8, u_1_24), sum_1_8);
											sum_1_0 = mm_add_epi32(mm_mullo_epi32(v_3_0, u_1_32), sum_1_0);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(v_3_8, u_1_40), sum_1_8);
											sum_1_0 = mm_add_epi32(mm_mullo_epi32(v_4_0, u_1_48), sum_1_0);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(v_4_8, u_1_56), sum_1_8);

#elif SIMD_TYPE >= SIMDTYPE_SSE
											u_1_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1));
											u_1_4_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 4));
											u_1_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 8));
											u_1_12_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 12));
											u_1_16_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 16));
											u_1_20_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 20));
											u_1_24_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 24));
											u_1_28_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 28));
											u_1_32_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 32));
											u_1_36_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 36));
											u_1_40_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 40));
											u_1_44_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 44));
											u_1_48_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 48));
											u_1_52_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 52));
											u_1_56_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 56));
											u_1_60_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 60));
											u_1_0 = mm_cvtepi16_epi32(u_1_0_int16);
											u_1_4 = mm_cvtepi16_epi32(u_1_4_int16);
											u_1_8 = mm_cvtepi16_epi32(u_1_8_int16);
											u_1_12 = mm_cvtepi16_epi32(u_1_12_int16);
											u_1_16 = mm_cvtepi16_epi32(u_1_16_int16);
											u_1_20 = mm_cvtepi16_epi32(u_1_20_int16);
											u_1_24 = mm_cvtepi16_epi32(u_1_24_int16);
											u_1_28 = mm_cvtepi16_epi32(u_1_28_int16);
											u_1_32 = mm_cvtepi16_epi32(u_1_32_int16);
											u_1_36 = mm_cvtepi16_epi32(u_1_36_int16);
											u_1_40 = mm_cvtepi16_epi32(u_1_40_int16);
											u_1_44 = mm_cvtepi16_epi32(u_1_44_int16);
											u_1_48 = mm_cvtepi16_epi32(u_1_48_int16);
											u_1_52 = mm_cvtepi16_epi32(u_1_52_int16);
											u_1_56 = mm_cvtepi16_epi32(u_1_56_int16);
											u_1_60 = mm_cvtepi16_epi32(u_1_60_int16);

											v_1_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1));
											v_1_4_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1 + 4));
											v_1_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1 + 8));
											v_1_12_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1 + 12));
											v_1_0 = mm_cvtepi16_epi32(v_1_0_int16);
											v_1_4 = mm_cvtepi16_epi32(v_1_4_int16);
											v_1_8 = mm_cvtepi16_epi32(v_1_8_int16);
											v_1_12 = mm_cvtepi16_epi32(v_1_12_int16);
											v_2_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_2));
											v_2_4_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_2 + 4));
											v_2_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_2 + 8));
											v_2_12_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_2 + 12));
											v_2_0 = mm_cvtepi16_epi32(v_2_0_int16);
											v_2_4 = mm_cvtepi16_epi32(v_2_4_int16);
											v_2_8 = mm_cvtepi16_epi32(v_2_8_int16);
											v_2_12 = mm_cvtepi16_epi32(v_2_12_int16);
											v_3_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_3));
											v_3_4_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_3 + 4));
											v_3_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_3 + 8));
											v_3_12_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_3 + 12));
											v_3_0 = mm_cvtepi16_epi32(v_3_0_int16);
											v_3_4 = mm_cvtepi16_epi32(v_3_4_int16);
											v_3_8 = mm_cvtepi16_epi32(v_3_8_int16);
											v_3_12 = mm_cvtepi16_epi32(v_3_12_int16);
											v_4_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_4));
											v_4_4_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_4 + 4));
											v_4_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_4 + 8));
											v_4_12_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_4 + 12));
											v_4_0 = mm_cvtepi16_epi32(v_4_0_int16);
											v_4_4 = mm_cvtepi16_epi32(v_4_4_int16);
											v_4_8 = mm_cvtepi16_epi32(v_4_8_int16);
											v_4_12 = mm_cvtepi16_epi32(v_4_12_int16);

											sum_1_0 = mm_add_epi32(mm_mullo_epi32(v_1_0, u_1_0), sum_1_0);
											sum_1_4 = mm_add_epi32(mm_mullo_epi32(v_1_4, u_1_4), sum_1_4);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(v_1_8, u_1_8), sum_1_8);
											sum_1_12 = mm_add_epi32(mm_mullo_epi32(v_1_12, u_1_12), sum_1_12);
											sum_1_0 = mm_add_epi32(mm_mullo_epi32(v_2_0, u_1_16), sum_1_0);
											sum_1_4 = mm_add_epi32(mm_mullo_epi32(v_2_4, u_1_20), sum_1_4);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(v_2_8, u_1_24), sum_1_8);
											sum_1_12 = mm_add_epi32(mm_mullo_epi32(v_2_12, u_1_28), sum_1_12);
											sum_1_0 = mm_add_epi32(mm_mullo_epi32(v_3_0, u_1_32), sum_1_0);
											sum_1_4 = mm_add_epi32(mm_mullo_epi32(v_3_4, u_1_36), sum_1_4);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(v_3_8, u_1_40), sum_1_8);
											sum_1_12 = mm_add_epi32(mm_mullo_epi32(v_3_12, u_1_44), sum_1_12);
											sum_1_0 = mm_add_epi32(mm_mullo_epi32(v_4_0, u_1_48), sum_1_0);
											sum_1_4 = mm_add_epi32(mm_mullo_epi32(v_4_4, u_1_52), sum_1_4);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(v_4_8, u_1_56), sum_1_8);
											sum_1_12 = mm_add_epi32(mm_mullo_epi32(v_4_12, u_1_60), sum_1_12);
#else
											for (int i = 0; i < tile_length_; i++)
											{
												mult_data1[i] += U_int16_data[U_offset_och_ich_1 + i] * V_int16_data[V_offset_ich_row_col_1 + i];
												mult_data1[i] += U_int16_data[U_offset_och_ich_1 + tile_length_ + i] * V_int16_data[V_offset_ich_row_col_2 + i];
												mult_data1[i] += U_int16_data[U_offset_och_ich_1 + 2 * tile_length_ + i] * V_int16_data[V_offset_ich_row_col_3 + i];
												mult_data1[i] += U_int16_data[U_offset_och_ich_1 + 3 * tile_length_ + i] * V_int16_data[V_offset_ich_row_col_4 + i];
											}
#endif
										}

										for (; ich < input_Channel_; ich++)
										{
											int U_offset_och_ich_1 = U_offset_och_1 + ich * tile_length_;
											int V_offset_ich_row_col_1 = V_offset_row_col + ich * h_w_tile_stride;

#if SIMD_TYPE >= SIMDTYPE_AVX
											u_1_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1));
											u_1_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 8));
											u_1_0 = mm_cvtepi16_epi32(u_1_0_int16);
											u_1_8 = mm_cvtepi16_epi32(u_1_8_int16);

											v_1_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1));
											v_1_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1 + 8));
											v_1_0 = mm_cvtepi16_epi32(v_1_0_int16);
											v_1_8 = mm_cvtepi16_epi32(v_1_8_int16);

											sum_1_0 = mm_add_epi32(mm_mullo_epi32(v_1_0, u_1_0), sum_1_0);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(v_1_8, u_1_8), sum_1_8);
#elif SIMD_TYPE >= SIMDTYPE_SSE
											u_1_0_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1));
											u_1_4_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 4));
											u_1_8_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 8));
											u_1_12_int16 = _mm_load_si128((__m128i*)(U_int16_data + U_offset_och_ich_1 + 12));
											u_1_0 = mm_cvtepi16_epi32(u_1_0_int16);
											u_1_4 = mm_cvtepi16_epi32(u_1_4_int16);
											u_1_8 = mm_cvtepi16_epi32(u_1_8_int16);
											u_1_12 = mm_cvtepi16_epi32(u_1_12_int16);

											v_1_0_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1));
											v_1_4_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1 + 4));
											v_1_8_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1 + 8));
											v_1_12_int16 = _mm_load_si128((__m128i*)(V_int16_data + V_offset_ich_row_col_1 + 12));
											v_1_0 = mm_cvtepi16_epi32(v_1_0_int16);
											v_1_4 = mm_cvtepi16_epi32(v_1_4_int16);
											v_1_8 = mm_cvtepi16_epi32(v_1_8_int16);
											v_1_12 = mm_cvtepi16_epi32(v_1_12_int16);

											sum_1_0 = mm_add_epi32(mm_mullo_epi32(v_1_0, u_1_0), sum_1_0);
											sum_1_4 = mm_add_epi32(mm_mullo_epi32(v_1_4, u_1_4), sum_1_4);
											sum_1_8 = mm_add_epi32(mm_mullo_epi32(v_1_8, u_1_8), sum_1_8);
											sum_1_12 = mm_add_epi32(mm_mullo_epi32(v_1_12, u_1_12), sum_1_12);
#else
											for (int i = 0; i < tile_length_; i++)
											{
												mult_data1[i] += U_int16_data[U_offset_och_ich_1 + i] * V_int16_data[V_offset_ich_row_col_1 + i];
											}
#endif
										}


#if SIMD_TYPE >= SIMDTYPE_AVX
										mm_store_si((mm_typei*)mult_data1, sum_1_0);
										mm_store_si((mm_typei*)(mult_data1 + 8), sum_1_8);
#elif SIMD_TYPE >= SIMDTYPE_SSE
										mm_store_si((mm_typei*)mult_data1, sum_1_0);
										mm_store_si((mm_typei*)(mult_data1 + 4), sum_1_4);
										mm_store_si((mm_typei*)(mult_data1 + 8), sum_1_8);
										mm_store_si((mm_typei*)(mult_data1 + 12), sum_1_12);
#endif // SIMD_TYPE >= SIMDTYPE_AVX

										calculate_ATmA23(mult_data1, result1);

										int row_in_output_data = i / w_tile_num_ * m_;
										int col_in_output_data = i % w_tile_num_* m_;
										int top_offset_row_col = row_in_output_data * output_dim_w_ + col_in_output_data;
										int top_offset_num_och_row_col_1 = top_offset_num_och_1 + top_offset_row_col;

										for (int row = 0; row < m_; row++)
										{
											int result_offset_row = row * m_;
											for (int col = 0; col < m_; col++)
											{
												top_int32_data[top_offset_num_och_row_col_1 + col] = result1[result_offset_row + col];
											}
											top_offset_num_och_row_col_1 += output_dim_w_;
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
							}
						}

						if ((add_h != 0) || (add_w != 0))
						{
							tensor_operation_cpu::cut_border_cpu(top, top, 0, add_h, 0, add_w);
						}
					}
					else
					{
						LOG(FATAL) << "group wrong!!!";
					}
				}
				else if (order_ == NHWC)
				{
					NOT_IMPLEMENTED;
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				delete V_int16_data;
			}
			else
			{
				//V=BT*d*B,so V has the same number as data tile, there are tile_length_ elements in single V
				//V_.reset(new tensor<float>(std::vector<int>{1, input_Channel_, total_tile_num, tile_length_}));
				//V_data = V_->mutable_cpu_data();
				V_data = new float[input_Channel_ * total_tile_num * tile_length_];

				if (order_ == NCHW)
				{
					if (group_ > 1)
					{
						for (int n = 0; n < num_; n++)
						{
							int bottom_offset_num = n * bottom_dim_;
							int top_offset_num = n * top_dim_;

							//get transformed bottom_data: V
							{
								int nn_inch = input_Channel_ >> 2;
								int remain_inch_start = nn_inch << 2;

#ifdef _OPENMP
#pragma omp parallel for
#endif
								for (int nn = 0; nn < nn_inch; nn++)
								{
									int ich = nn * 4;
									int bottom_offset_num_ich_1 = bottom_offset_num + ich * input_spatial_dim_;
									int bottom_offset_num_ich_2 = bottom_offset_num_ich_1 + input_spatial_dim_;
									int bottom_offset_num_ich_3 = bottom_offset_num_ich_2 + input_spatial_dim_;
									int bottom_offset_num_ich_4 = bottom_offset_num_ich_3 + input_spatial_dim_;
									int V_offset_ich_1 = ich * h_w_tile_stride;
									int V_offset_ich_2 = V_offset_ich_1 + h_w_tile_stride;
									int V_offset_ich_3 = V_offset_ich_2 + h_w_tile_stride;
									int V_offset_ich_4 = V_offset_ich_3 + h_w_tile_stride;

									for (int i = 0; i < total_tile_num; i++)
									{
										int row_in_input_data = (i / w_tile_num_) * m_;
										int col_in_input_data = (i % w_tile_num_) * m_;
										int bottom_offset_row_col = row_in_input_data * input_dim_w_ + col_in_input_data;
										int bottom_offset_num_ich_row_col_1 = bottom_offset_num_ich_1 + bottom_offset_row_col;
										int bottom_offset_num_ich_row_col_2 = bottom_offset_num_ich_2 + bottom_offset_row_col;
										int bottom_offset_num_ich_row_col_3 = bottom_offset_num_ich_3 + bottom_offset_row_col;
										int bottom_offset_num_ich_row_col_4 = bottom_offset_num_ich_4 + bottom_offset_row_col;

										int tile_offset = i * tile_length_;
										int V_offset_ich_row_col_1 = V_offset_ich_1 + tile_offset;
										int V_offset_ich_row_col_2 = V_offset_ich_2 + tile_offset;
										int V_offset_ich_row_col_3 = V_offset_ich_3 + tile_offset;
										int V_offset_ich_row_col_4 = V_offset_ich_4 + tile_offset;

										const float *row_data1_1 = bottom_data + bottom_offset_num_ich_row_col_1;
										const float *row_data1_2 = row_data1_1 + input_dim_w_;
										const float *row_data1_3 = row_data1_2 + input_dim_w_;
										const float *row_data1_4 = row_data1_3 + input_dim_w_;
										calculate_BTdB23(row_data1_1, row_data1_2, row_data1_3, row_data1_4, V_data + V_offset_ich_row_col_1);

										const float *row_data2_1 = bottom_data + bottom_offset_num_ich_row_col_2;
										const float *row_data2_2 = row_data2_1 + input_dim_w_;
										const float *row_data2_3 = row_data2_2 + input_dim_w_;
										const float *row_data2_4 = row_data2_3 + input_dim_w_;
										calculate_BTdB23(row_data2_1, row_data2_2, row_data2_3, row_data2_4, V_data + V_offset_ich_row_col_2);

										const float *row_data3_1 = bottom_data + bottom_offset_num_ich_row_col_3;
										const float *row_data3_2 = row_data3_1 + input_dim_w_;
										const float *row_data3_3 = row_data3_2 + input_dim_w_;
										const float *row_data3_4 = row_data3_3 + input_dim_w_;
										calculate_BTdB23(row_data3_1, row_data3_2, row_data3_3, row_data3_4, V_data + V_offset_ich_row_col_3);

										const float *row_data4_1 = bottom_data + bottom_offset_num_ich_row_col_4;
										const float *row_data4_2 = row_data4_1 + input_dim_w_;
										const float *row_data4_3 = row_data4_2 + input_dim_w_;
										const float *row_data4_4 = row_data4_3 + input_dim_w_;
										calculate_BTdB23(row_data4_1, row_data4_2, row_data4_3, row_data4_4, V_data + V_offset_ich_row_col_4);
									}
								}

								for (int ich = remain_inch_start; ich < input_Channel_; ich++)
								{
									int bottom_offset_num_ich_1 = bottom_offset_num + ich * input_spatial_dim_;
									int V_offset_ich_1 = ich * h_w_tile_stride;

									for (int i = 0; i < total_tile_num; i++)
									{
										int row_in_input_data = (i / w_tile_num_) * m_;
										int col_in_input_data = (i % w_tile_num_) * m_;
										int bottom_offset_row_col = row_in_input_data * input_dim_w_ + col_in_input_data;
										int bottom_offset_num_ich_row_col_1 = bottom_offset_num_ich_1 + bottom_offset_row_col;

										int tile_offset = i * tile_length_;
										int V_offset_ich_row_col_1 = V_offset_ich_1 + tile_offset;

										const float *row_data1_1 = bottom_data + bottom_offset_num_ich_row_col_1;
										const float *row_data1_2 = row_data1_1 + input_dim_w_;
										const float *row_data1_3 = row_data1_2 + input_dim_w_;
										const float *row_data1_4 = row_data1_3 + input_dim_w_;
										calculate_BTdB23(row_data1_1, row_data1_2, row_data1_3, row_data1_4, V_data + V_offset_ich_row_col_1);
									}
								}
							}

							//multiply
							{
								int nn_outch = output_Channel_ >> 2;
								int remain_outch_start = nn_outch << 2;
#ifdef _OPENMP
#pragma omp parallel for
#endif
								for (int nn = 0; nn < nn_outch; nn++)
								{
#if SIMD_TYPE >= SIMDTYPE_AVX
									mm_type sum_1_0;
									mm_type sum_1_8;
									mm_type sum_2_0;
									mm_type sum_2_8;
									mm_type sum_3_0;
									mm_type sum_3_8;
									mm_type sum_4_0;
									mm_type sum_4_8;
									mm_type u_1_0;
									mm_type u_1_8;
									mm_type u_2_0;
									mm_type u_2_8;
									mm_type u_3_0;
									mm_type u_3_8;
									mm_type u_4_0;
									mm_type u_4_8;
									mm_type v_1_0;
									mm_type v_1_8;
									mm_type v_2_0;
									mm_type v_2_8;
									mm_type v_3_0;
									mm_type v_3_8;
									mm_type v_4_0;
									mm_type v_4_8;
#elif SIMD_TYPE >= SIMDTYPE_SSE
									mm_type sum_1_0;
									mm_type sum_1_4;
									mm_type sum_1_8;
									mm_type sum_1_12;
									mm_type sum_2_0;
									mm_type sum_2_4;
									mm_type sum_2_8;
									mm_type sum_2_12;
									mm_type sum_3_0;
									mm_type sum_3_4;
									mm_type sum_3_8;
									mm_type sum_3_12;
									mm_type sum_4_0;
									mm_type sum_4_4;
									mm_type sum_4_8;
									mm_type sum_4_12;
									mm_type u_1_0;
									mm_type u_1_4;
									mm_type u_1_8;
									mm_type u_1_12;
									mm_type u_2_0;
									mm_type u_2_4;
									mm_type u_2_8;
									mm_type u_2_12;
									mm_type u_3_0;
									mm_type u_3_4;
									mm_type u_3_8;
									mm_type u_3_12;
									mm_type u_4_0;
									mm_type u_4_4;
									mm_type u_4_8;
									mm_type u_4_12;
									mm_type v_1_0;
									mm_type v_1_4;
									mm_type v_1_8;
									mm_type v_1_12;
									mm_type v_2_0;
									mm_type v_2_4;
									mm_type v_2_8;
									mm_type v_2_12;
									mm_type v_3_0;
									mm_type v_3_4;
									mm_type v_3_8;
									mm_type v_3_12;
									mm_type v_4_0;
									mm_type v_4_4;
									mm_type v_4_8;
									mm_type v_4_12;
#endif

									int och = nn * 4;
									int U_offset_och_1 = och * tile_length_;
									int U_offset_och_2 = U_offset_och_1 + tile_length_;
									int U_offset_och_3 = U_offset_och_2 + tile_length_;
									int U_offset_och_4 = U_offset_och_3 + tile_length_;

									int V_offset_och_1 = och * h_w_tile_stride;
									int V_offset_och_2 = V_offset_och_1 + h_w_tile_stride;
									int V_offset_och_3 = V_offset_och_2 + h_w_tile_stride;
									int V_offset_och_4 = V_offset_och_3 + h_w_tile_stride;

									float bias1 = bias_data[och];
									float bias2 = bias_data[och + 1];
									float bias3 = bias_data[och + 2];
									float bias4 = bias_data[och + 3];
									float result1[4], result2[4], result3[4], result4[4];

									int top_offset_num_och_1 = top_offset_num + och * output_spatial_dim_;
									int top_offset_num_och_2 = top_offset_num_och_1 + output_spatial_dim_;
									int top_offset_num_och_3 = top_offset_num_och_2 + output_spatial_dim_;
									int top_offset_num_och_4 = top_offset_num_och_3 + output_spatial_dim_;

									for (int i = 0; i < total_tile_num; i++)
									{
										float mult_data1[16] = { 0 };
										float mult_data2[16] = { 0 };
										float mult_data3[16] = { 0 };
										float mult_data4[16] = { 0 };

										int V_offset_row_col = i * tile_length_;
										int V_offset_och_row_col_1 = V_offset_och_1 + V_offset_row_col;
										int V_offset_och_row_col_2 = V_offset_och_2 + V_offset_row_col;
										int V_offset_och_row_col_3 = V_offset_och_3 + V_offset_row_col;
										int V_offset_och_row_col_4 = V_offset_och_4 + V_offset_row_col;

#if SIMD_TYPE >= SIMDTYPE_AVX
										u_1_0 = mm_load_ps(U_data + U_offset_och_1);
										u_1_8 = mm_load_ps(U_data + U_offset_och_1 + 8);
										u_2_0 = mm_load_ps(U_data + U_offset_och_2);
										u_2_8 = mm_load_ps(U_data + U_offset_och_2 + 8);
										u_3_0 = mm_load_ps(U_data + U_offset_och_3);
										u_3_8 = mm_load_ps(U_data + U_offset_och_3 + 8);
										u_4_0 = mm_load_ps(U_data + U_offset_och_4);
										u_4_8 = mm_load_ps(U_data + U_offset_och_4 + 8);
										v_1_0 = mm_load_ps(V_data + V_offset_och_row_col_1);
										v_1_8 = mm_load_ps(V_data + V_offset_och_row_col_1 + 8);
										v_2_0 = mm_load_ps(V_data + V_offset_och_row_col_2);
										v_2_8 = mm_load_ps(V_data + V_offset_och_row_col_2 + 8);
										v_3_0 = mm_load_ps(V_data + V_offset_och_row_col_3);
										v_3_8 = mm_load_ps(V_data + V_offset_och_row_col_3 + 8);
										v_4_0 = mm_load_ps(V_data + V_offset_och_row_col_4);
										v_4_8 = mm_load_ps(V_data + V_offset_och_row_col_4 + 8);

										sum_1_0 = mm_mul_ps(v_1_0, u_1_0);
										sum_1_8 = mm_mul_ps(v_1_8, u_1_8);
										sum_2_0 = mm_mul_ps(v_2_0, u_2_0);
										sum_2_8 = mm_mul_ps(v_2_8, u_2_8);
										sum_3_0 = mm_mul_ps(v_3_0, u_3_0);
										sum_3_8 = mm_mul_ps(v_3_8, u_3_8);
										sum_4_0 = mm_mul_ps(v_4_0, u_4_0);
										sum_4_8 = mm_mul_ps(v_4_8, u_4_8);

										mm_store_ps(mult_data1, sum_1_0);
										mm_store_ps(mult_data1 + 8, sum_1_8);
										mm_store_ps(mult_data2, sum_2_0);
										mm_store_ps(mult_data2 + 8, sum_2_8);
										mm_store_ps(mult_data3, sum_3_0);
										mm_store_ps(mult_data3 + 8, sum_3_8);
										mm_store_ps(mult_data4, sum_4_0);
										mm_store_ps(mult_data4 + 8, sum_4_8);
#elif SIMD_TYPE >= SIMDTYPE_SSE
										u_1_0 = mm_load_ps(U_data + U_offset_och_1);
										u_1_4 = mm_load_ps(U_data + U_offset_och_1 + 4);
										u_1_8 = mm_load_ps(U_data + U_offset_och_1 + 8);
										u_1_12 = mm_load_ps(U_data + U_offset_och_1 + 12);

										u_2_0 = mm_load_ps(U_data + U_offset_och_2);
										u_2_4 = mm_load_ps(U_data + U_offset_och_2 + 4);
										u_2_8 = mm_load_ps(U_data + U_offset_och_2 + 8);
										u_2_12 = mm_load_ps(U_data + U_offset_och_2 + 12);

										u_3_0 = mm_load_ps(U_data + U_offset_och_3);
										u_3_4 = mm_load_ps(U_data + U_offset_och_3 + 4);
										u_3_8 = mm_load_ps(U_data + U_offset_och_3 + 8);
										u_3_12 = mm_load_ps(U_data + U_offset_och_3 + 12);

										u_4_0 = mm_load_ps(U_data + U_offset_och_4);
										u_4_4 = mm_load_ps(U_data + U_offset_och_4 + 4);
										u_4_8 = mm_load_ps(U_data + U_offset_och_4 + 8);
										u_4_12 = mm_load_ps(U_data + U_offset_och_4 + 12);

										v_1_0 = mm_load_ps(V_data + V_offset_och_row_col_1);
										v_1_4 = mm_load_ps(V_data + V_offset_och_row_col_1 + 4);
										v_1_8 = mm_load_ps(V_data + V_offset_och_row_col_1 + 8);
										v_1_12 = mm_load_ps(V_data + V_offset_och_row_col_1 + 12);

										v_2_0 = mm_load_ps(V_data + V_offset_och_row_col_2);
										v_2_4 = mm_load_ps(V_data + V_offset_och_row_col_2 + 4);
										v_2_8 = mm_load_ps(V_data + V_offset_och_row_col_2 + 8);
										v_2_12 = mm_load_ps(V_data + V_offset_och_row_col_2 + 12);

										v_3_0 = mm_load_ps(V_data + V_offset_och_row_col_3);
										v_3_4 = mm_load_ps(V_data + V_offset_och_row_col_3 + 4);
										v_3_8 = mm_load_ps(V_data + V_offset_och_row_col_3 + 8);
										v_3_12 = mm_load_ps(V_data + V_offset_och_row_col_3 + 12);

										v_4_0 = mm_load_ps(V_data + V_offset_och_row_col_4);
										v_4_4 = mm_load_ps(V_data + V_offset_och_row_col_4 + 4);
										v_4_8 = mm_load_ps(V_data + V_offset_och_row_col_4 + 8);
										v_4_12 = mm_load_ps(V_data + V_offset_och_row_col_4 + 12);

										sum_1_0 = mm_mul_ps(v_1_0, u_1_0);
										sum_1_4 = mm_mul_ps(v_1_4, u_1_4);
										sum_1_8 = mm_mul_ps(v_1_8, u_1_8);
										sum_1_12 = mm_mul_ps(v_1_12, u_1_12);

										sum_2_0 = mm_mul_ps(v_2_0, u_2_0);
										sum_2_4 = mm_mul_ps(v_2_4, u_2_4);
										sum_2_8 = mm_mul_ps(v_2_8, u_2_8);
										sum_2_12 = mm_mul_ps(v_2_12, u_2_12);

										sum_3_0 = mm_mul_ps(v_3_0, u_3_0);
										sum_3_4 = mm_mul_ps(v_3_4, u_3_4);
										sum_3_8 = mm_mul_ps(v_3_8, u_3_8);
										sum_3_12 = mm_mul_ps(v_3_12, u_3_12);

										sum_4_0 = mm_mul_ps(v_4_0, u_4_0);
										sum_4_4 = mm_mul_ps(v_4_4, u_4_4);
										sum_4_8 = mm_mul_ps(v_4_8, u_4_8);
										sum_4_12 = mm_mul_ps(v_4_12, u_4_12);

										mm_store_ps(mult_data1, sum_1_0);
										mm_store_ps(mult_data1 + 4, sum_1_4);
										mm_store_ps(mult_data1 + 8, sum_1_8);
										mm_store_ps(mult_data1 + 12, sum_1_12);
										mm_store_ps(mult_data2, sum_2_0);
										mm_store_ps(mult_data2 + 4, sum_2_4);
										mm_store_ps(mult_data2 + 8, sum_2_8);
										mm_store_ps(mult_data2 + 12, sum_2_12);
										mm_store_ps(mult_data3, sum_3_0);
										mm_store_ps(mult_data3 + 4, sum_3_4);
										mm_store_ps(mult_data3 + 8, sum_3_8);
										mm_store_ps(mult_data3 + 12, sum_3_12);
										mm_store_ps(mult_data4, sum_4_0);
										mm_store_ps(mult_data4 + 4, sum_4_4);
										mm_store_ps(mult_data4 + 8, sum_4_8);
										mm_store_ps(mult_data4 + 12, sum_4_12);
#else
										for (int i = 0; i < tile_length_; i++)
										{
											mult_data1[i] = U_data[U_offset_och_1 + i] * V_data[V_offset_och_row_col_1 + i];
											mult_data2[i] = U_data[U_offset_och_2 + i] * V_data[V_offset_och_row_col_2 + i];
											mult_data3[i] = U_data[U_offset_och_3 + i] * V_data[V_offset_och_row_col_3 + i];
											mult_data4[i] = U_data[U_offset_och_4 + i] * V_data[V_offset_och_row_col_4 + i];
										}
#endif

										calculate_ATmA23(mult_data1, result1);
										calculate_ATmA23(mult_data2, result2);
										calculate_ATmA23(mult_data3, result3);
										calculate_ATmA23(mult_data4, result4);

										int row_in_output_data = i / w_tile_num_ * m_;
										int col_in_output_data = i % w_tile_num_* m_;
										int top_offset_row_col = row_in_output_data * output_dim_w_ + col_in_output_data;
										int top_offset_num_och_row_col_1 = top_offset_num_och_1 + top_offset_row_col;
										int top_offset_num_och_row_col_2 = top_offset_num_och_2 + top_offset_row_col;
										int top_offset_num_och_row_col_3 = top_offset_num_och_3 + top_offset_row_col;
										int top_offset_num_och_row_col_4 = top_offset_num_och_4 + top_offset_row_col;

										for (int row = 0; row < m_; row++)
										{
											int result_offset_row = row * m_;
											for (int col = 0; col < m_; col++)
											{
												top_data[top_offset_num_och_row_col_1 + col] = result1[result_offset_row + col] + bias1;
												top_data[top_offset_num_och_row_col_2 + col] = result2[result_offset_row + col] + bias2;
												top_data[top_offset_num_och_row_col_3 + col] = result3[result_offset_row + col] + bias3;
												top_data[top_offset_num_och_row_col_4 + col] = result4[result_offset_row + col] + bias4;
											}
											top_offset_num_och_row_col_1 += output_dim_w_;
											top_offset_num_och_row_col_2 += output_dim_w_;
											top_offset_num_och_row_col_3 += output_dim_w_;
											top_offset_num_och_row_col_4 += output_dim_w_;
										}
									}
								}

								for (int och = remain_outch_start; och < output_Channel_; och++)
								{
#if SIMD_TYPE >= SIMDTYPE_AVX
									mm_type sum_1_0;
									mm_type sum_1_8;
									mm_type u_1_0;
									mm_type u_1_8;
									mm_type v_1_0;
									mm_type v_1_8;
#elif SIMD_TYPE >= SIMDTYPE_SSE
									mm_type sum_1_0;
									mm_type sum_1_4;
									mm_type sum_1_8;
									mm_type sum_1_12;
									mm_type u_1_0;
									mm_type u_1_4;
									mm_type u_1_8;
									mm_type u_1_12;
									mm_type v_1_0;
									mm_type v_1_4;
									mm_type v_1_8;
									mm_type v_1_12;
#endif

									int U_offset_och_1 = och * tile_length_;
									int V_offset_och_1 = och * h_w_tile_stride;

									float bias1 = bias_data[och];
									float result1[4];
									int top_offset_num_och_1 = top_offset_num + och * output_spatial_dim_;

									for (int i = 0; i < total_tile_num; i++)
									{
										float mult_data1[16] = { 0 };
										int V_offset_row_col = i * tile_length_;

										int V_offset_och_row_col_1 = V_offset_och_1 + V_offset_row_col;

#if SIMD_TYPE >= SIMDTYPE_AVX
										u_1_0 = mm_load_ps(U_data + U_offset_och_1);
										u_1_8 = mm_load_ps(U_data + U_offset_och_1 + 8);

										v_1_0 = mm_load_ps(V_data + V_offset_och_row_col_1);
										v_1_8 = mm_load_ps(V_data + V_offset_och_row_col_1 + 8);

										sum_1_0 = mm_mul_ps(v_1_0, u_1_0);
										sum_1_8 = mm_mul_ps(v_1_8, u_1_8);

										mm_store_ps(mult_data1, sum_1_0);
										mm_store_ps(mult_data1 + 8, sum_1_8);
#elif SIMD_TYPE >= SIMDTYPE_SSE
										u_1_0 = mm_load_ps(U_data + U_offset_och_1);
										u_1_4 = mm_load_ps(U_data + U_offset_och_1 + 4);
										u_1_8 = mm_load_ps(U_data + U_offset_och_1 + 8);
										u_1_12 = mm_load_ps(U_data + U_offset_och_1 + 12);

										v_1_0 = mm_load_ps(V_data + V_offset_och_row_col_1);
										v_1_4 = mm_load_ps(V_data + V_offset_och_row_col_1 + 4);
										v_1_8 = mm_load_ps(V_data + V_offset_och_row_col_1 + 8);
										v_1_12 = mm_load_ps(V_data + V_offset_och_row_col_1 + 12);

										sum_1_0 = mm_mul_ps(v_1_0, u_1_0);
										sum_1_4 = mm_mul_ps(v_1_4, u_1_4);
										sum_1_8 = mm_mul_ps(v_1_8, u_1_8);
										sum_1_12 = mm_mul_ps(v_1_12, u_1_12);

										mm_store_ps(mult_data1, sum_1_0);
										mm_store_ps(mult_data1 + 4, sum_1_4);
										mm_store_ps(mult_data1 + 8, sum_1_8);
										mm_store_ps(mult_data1 + 12, sum_1_12);
#else
										for (int i = 0; i < tile_length_; i++)
										{
											mult_data1[i] = U_data[U_offset_och_1 + i] * V_data[V_offset_och_row_col_1 + i];
										}
#endif

										calculate_ATmA23(mult_data1, result1);

										int row_in_output_data = i / w_tile_num_ * m_;
										int col_in_output_data = i % w_tile_num_* m_;
										int top_offset_row_col = row_in_output_data * output_dim_w_ + col_in_output_data;
										int top_offset_num_och_row_col_1 = top_offset_num_och_1 + top_offset_row_col;

										for (int row = 0; row < m_; row++)
										{
											int result_offset_row = row * m_;
											for (int col = 0; col < m_; col++)
											{
												top_data[top_offset_num_och_row_col_1 + col] = result1[result_offset_row + col] + bias1;
											}
											top_offset_num_och_row_col_1 += output_dim_w_;
										}
									}
								}
							}
						}

						if ((add_h != 0) || (add_w != 0))
						{
							tensor_operation_cpu::cut_border_cpu(top, top, 0, add_h, 0, add_w);
						}
					}
					else if (group_ == 1)
					{
						int U_offset_single_och = input_Channel_ * tile_length_;

						for (int n = 0; n < num_; n++)
						{
							int bottom_offset_num = n * bottom_dim_;
							int top_offset_num = n * top_dim_;

							//get transformed bottom_data: V
							{
								int nn_inch = input_Channel_ >> 2;
								int remain_inch_start = nn_inch << 2;

#ifdef _OPENMP
#pragma omp parallel for
#endif
								for (int nn = 0; nn < nn_inch; nn++)
								{
									int ich = nn * 4;
									int bottom_offset_num_ich_1 = bottom_offset_num + ich * input_spatial_dim_;
									int bottom_offset_num_ich_2 = bottom_offset_num_ich_1 + input_spatial_dim_;
									int bottom_offset_num_ich_3 = bottom_offset_num_ich_2 + input_spatial_dim_;
									int bottom_offset_num_ich_4 = bottom_offset_num_ich_3 + input_spatial_dim_;
									int V_offset_ich_1 = ich * h_w_tile_stride;
									int V_offset_ich_2 = V_offset_ich_1 + h_w_tile_stride;
									int V_offset_ich_3 = V_offset_ich_2 + h_w_tile_stride;
									int V_offset_ich_4 = V_offset_ich_3 + h_w_tile_stride;

									for (int i = 0; i < total_tile_num; i++)
									{
										int row_in_input_data = (i / w_tile_num_) * m_;
										int col_in_input_data = (i % w_tile_num_) * m_;
										int bottom_offset_row_col = row_in_input_data * input_dim_w_ + col_in_input_data;
										int bottom_offset_num_ich_row_col_1 = bottom_offset_num_ich_1 + bottom_offset_row_col;
										int bottom_offset_num_ich_row_col_2 = bottom_offset_num_ich_2 + bottom_offset_row_col;
										int bottom_offset_num_ich_row_col_3 = bottom_offset_num_ich_3 + bottom_offset_row_col;
										int bottom_offset_num_ich_row_col_4 = bottom_offset_num_ich_4 + bottom_offset_row_col;

										int tile_offset = i * tile_length_;
										int V_offset_ich_row_col_1 = V_offset_ich_1 + tile_offset;
										int V_offset_ich_row_col_2 = V_offset_ich_2 + tile_offset;
										int V_offset_ich_row_col_3 = V_offset_ich_3 + tile_offset;
										int V_offset_ich_row_col_4 = V_offset_ich_4 + tile_offset;

										const float *row_data1_1 = bottom_data + bottom_offset_num_ich_row_col_1;
										const float *row_data1_2 = row_data1_1 + input_dim_w_;
										const float *row_data1_3 = row_data1_2 + input_dim_w_;
										const float *row_data1_4 = row_data1_3 + input_dim_w_;
										calculate_BTdB23(row_data1_1, row_data1_2, row_data1_3, row_data1_4, V_data + V_offset_ich_row_col_1);

										const float *row_data2_1 = bottom_data + bottom_offset_num_ich_row_col_2;
										const float *row_data2_2 = row_data2_1 + input_dim_w_;
										const float *row_data2_3 = row_data2_2 + input_dim_w_;
										const float *row_data2_4 = row_data2_3 + input_dim_w_;
										calculate_BTdB23(row_data2_1, row_data2_2, row_data2_3, row_data2_4, V_data + V_offset_ich_row_col_2);

										const float *row_data3_1 = bottom_data + bottom_offset_num_ich_row_col_3;
										const float *row_data3_2 = row_data3_1 + input_dim_w_;
										const float *row_data3_3 = row_data3_2 + input_dim_w_;
										const float *row_data3_4 = row_data3_3 + input_dim_w_;
										calculate_BTdB23(row_data3_1, row_data3_2, row_data3_3, row_data3_4, V_data + V_offset_ich_row_col_3);

										const float *row_data4_1 = bottom_data + bottom_offset_num_ich_row_col_4;
										const float *row_data4_2 = row_data4_1 + input_dim_w_;
										const float *row_data4_3 = row_data4_2 + input_dim_w_;
										const float *row_data4_4 = row_data4_3 + input_dim_w_;
										calculate_BTdB23(row_data4_1, row_data4_2, row_data4_3, row_data4_4, V_data + V_offset_ich_row_col_4);
									}
								}

								for (int ich = remain_inch_start; ich < input_Channel_; ich++)
								{
									int bottom_offset_num_ich_1 = bottom_offset_num + ich * input_spatial_dim_;
									int V_offset_ich_1 = ich * h_w_tile_stride;

									for (int i = 0; i < total_tile_num; i++)
									{
										int row_in_input_data = (i / w_tile_num_) * m_;
										int col_in_input_data = (i % w_tile_num_) * m_;
										int bottom_offset_row_col = row_in_input_data * input_dim_w_ + col_in_input_data;
										int bottom_offset_num_ich_row_col_1 = bottom_offset_num_ich_1 + bottom_offset_row_col;

										int tile_offset = i * tile_length_;
										int V_offset_ich_row_col_1 = V_offset_ich_1 + tile_offset;

										const float *row_data1_1 = bottom_data + bottom_offset_num_ich_row_col_1;
										const float *row_data1_2 = row_data1_1 + input_dim_w_;
										const float *row_data1_3 = row_data1_2 + input_dim_w_;
										const float *row_data1_4 = row_data1_3 + input_dim_w_;
										calculate_BTdB23(row_data1_1, row_data1_2, row_data1_3, row_data1_4, V_data + V_offset_ich_row_col_1);
									}
								}
							}

							//multiply
							{
								int nn_outch = output_Channel_ >> 2;
								int remain_outch_start = nn_outch << 2;
#ifdef _OPENMP
#pragma omp parallel for
#endif
								for (int nn = 0; nn < nn_outch; nn++)
								{
#if SIMD_TYPE >= SIMDTYPE_AVX
									mm_type sum_1_0;
									mm_type sum_1_8;
									mm_type sum_2_0;
									mm_type sum_2_8;
									mm_type sum_3_0;
									mm_type sum_3_8;
									mm_type sum_4_0;
									mm_type sum_4_8;
									mm_type u_1_0;
									mm_type u_1_8;
									mm_type u_1_16;
									mm_type u_1_24;
									mm_type u_1_32;
									mm_type u_1_40;
									mm_type u_1_48;
									mm_type u_1_56;
									mm_type u_2_0;
									mm_type u_2_8;
									mm_type u_2_16;
									mm_type u_2_24;
									mm_type u_2_32;
									mm_type u_2_40;
									mm_type u_2_48;
									mm_type u_2_56;
									mm_type u_3_0;
									mm_type u_3_8;
									mm_type u_3_16;
									mm_type u_3_24;
									mm_type u_3_32;
									mm_type u_3_40;
									mm_type u_3_48;
									mm_type u_3_56;
									mm_type u_4_0;
									mm_type u_4_8;
									mm_type u_4_16;
									mm_type u_4_24;
									mm_type u_4_32;
									mm_type u_4_40;
									mm_type u_4_48;
									mm_type u_4_56;
									mm_type v_1_0;
									mm_type v_1_8;
									mm_type v_2_0;
									mm_type v_2_8;
									mm_type v_3_0;
									mm_type v_3_8;
									mm_type v_4_0;
									mm_type v_4_8;
#elif SIMD_TYPE >= SIMDTYPE_SSE
									mm_type sum_1_0;
									mm_type sum_1_4;
									mm_type sum_1_8;
									mm_type sum_1_12;
									mm_type sum_2_0;
									mm_type sum_2_4;
									mm_type sum_2_8;
									mm_type sum_2_12;
									mm_type sum_3_0;
									mm_type sum_3_4;
									mm_type sum_3_8;
									mm_type sum_3_12;
									mm_type sum_4_0;
									mm_type sum_4_4;
									mm_type sum_4_8;
									mm_type sum_4_12;
									mm_type u_1_0;
									mm_type u_1_4;
									mm_type u_1_8;
									mm_type u_1_12;
									mm_type u_1_16;
									mm_type u_1_20;
									mm_type u_1_24;
									mm_type u_1_28;
									mm_type u_1_32;
									mm_type u_1_36;
									mm_type u_1_40;
									mm_type u_1_44;
									mm_type u_1_48;
									mm_type u_1_52;
									mm_type u_1_56;
									mm_type u_1_60;
									mm_type u_2_0;
									mm_type u_2_4;
									mm_type u_2_8;
									mm_type u_2_12;
									mm_type u_2_16;
									mm_type u_2_20;
									mm_type u_2_24;
									mm_type u_2_28;
									mm_type u_2_32;
									mm_type u_2_36;
									mm_type u_2_40;
									mm_type u_2_44;
									mm_type u_2_48;
									mm_type u_2_52;
									mm_type u_2_56;
									mm_type u_2_60;
									mm_type u_3_0;
									mm_type u_3_4;
									mm_type u_3_8;
									mm_type u_3_12;
									mm_type u_3_16;
									mm_type u_3_20;
									mm_type u_3_24;
									mm_type u_3_28;
									mm_type u_3_32;
									mm_type u_3_36;
									mm_type u_3_40;
									mm_type u_3_44;
									mm_type u_3_48;
									mm_type u_3_52;
									mm_type u_3_56;
									mm_type u_3_60;
									mm_type u_4_0;
									mm_type u_4_4;
									mm_type u_4_8;
									mm_type u_4_12;
									mm_type u_4_16;
									mm_type u_4_20;
									mm_type u_4_24;
									mm_type u_4_28;
									mm_type u_4_32;
									mm_type u_4_36;
									mm_type u_4_40;
									mm_type u_4_44;
									mm_type u_4_48;
									mm_type u_4_52;
									mm_type u_4_56;
									mm_type u_4_60;
									mm_type v_1_0;
									mm_type v_1_4;
									mm_type v_1_8;
									mm_type v_1_12;
									mm_type v_2_0;
									mm_type v_2_4;
									mm_type v_2_8;
									mm_type v_2_12;
									mm_type v_3_0;
									mm_type v_3_4;
									mm_type v_3_8;
									mm_type v_3_12;
									mm_type v_4_0;
									mm_type v_4_4;
									mm_type v_4_8;
									mm_type v_4_12;
#endif

									int och = nn * 4;
									int U_offset_och_1 = och * U_offset_single_och;
									int U_offset_och_2 = U_offset_och_1 + U_offset_single_och;
									int U_offset_och_3 = U_offset_och_2 + U_offset_single_och;
									int U_offset_och_4 = U_offset_och_3 + U_offset_single_och;

									float bias1 = bias_data[och];
									float bias2 = bias_data[och + 1];
									float bias3 = bias_data[och + 2];
									float bias4 = bias_data[och + 3];
									float result1[4], result2[4], result3[4], result4[4];

									int top_offset_num_och_1 = top_offset_num + och * output_spatial_dim_;
									int top_offset_num_och_2 = top_offset_num_och_1 + output_spatial_dim_;
									int top_offset_num_och_3 = top_offset_num_och_2 + output_spatial_dim_;
									int top_offset_num_och_4 = top_offset_num_och_3 + output_spatial_dim_;

									for (int i = 0; i < total_tile_num; i++)
									{
										float mult_data1[16] = { 0 };
										float mult_data2[16] = { 0 };
										float mult_data3[16] = { 0 };
										float mult_data4[16] = { 0 };

										int V_offset_row_col = i * tile_length_;

#if SIMD_TYPE >= SIMDTYPE_AVX
										sum_1_0 = mm_setzero_ps();
										sum_1_8 = mm_setzero_ps();
										sum_2_0 = mm_setzero_ps();
										sum_2_8 = mm_setzero_ps();
										sum_3_0 = mm_setzero_ps();
										sum_3_8 = mm_setzero_ps();
										sum_4_0 = mm_setzero_ps();
										sum_4_8 = mm_setzero_ps();
#elif SIMD_TYPE >= SIMDTYPE_SSE
										sum_1_0 = mm_setzero_ps();
										sum_1_4 = mm_setzero_ps();
										sum_1_8 = mm_setzero_ps();
										sum_1_12 = mm_setzero_ps();
										sum_2_0 = mm_setzero_ps();
										sum_2_4 = mm_setzero_ps();
										sum_2_8 = mm_setzero_ps();
										sum_2_12 = mm_setzero_ps();
										sum_3_0 = mm_setzero_ps();
										sum_3_4 = mm_setzero_ps();
										sum_3_8 = mm_setzero_ps();
										sum_3_12 = mm_setzero_ps();
										sum_4_0 = mm_setzero_ps();
										sum_4_4 = mm_setzero_ps();
										sum_4_8 = mm_setzero_ps();
										sum_4_12 = mm_setzero_ps();
#endif // SIMD_TYPE >= SIMDTYPE_AVX
										int ich = 0;
										for (; ich + 3 < input_Channel_; ich += 4)
										{
											int offset = ich * tile_length_;
											int U_offset_och_ich_1 = U_offset_och_1 + offset;
											int U_offset_och_ich_2 = U_offset_och_2 + offset;
											int U_offset_och_ich_3 = U_offset_och_3 + offset;
											int U_offset_och_ich_4 = U_offset_och_4 + offset;

											int V_offset_ich_row_col_1 = V_offset_row_col + ich * h_w_tile_stride;
											int V_offset_ich_row_col_2 = V_offset_ich_row_col_1 + h_w_tile_stride;
											int V_offset_ich_row_col_3 = V_offset_ich_row_col_2 + h_w_tile_stride;
											int V_offset_ich_row_col_4 = V_offset_ich_row_col_3 + h_w_tile_stride;

#if SIMD_TYPE >= SIMDTYPE_AVX
											u_1_0 = mm_load_ps(U_data + U_offset_och_ich_1);
											u_1_8 = mm_load_ps(U_data + U_offset_och_ich_1 + 8);
											u_1_16 = mm_load_ps(U_data + U_offset_och_ich_1 + 16);
											u_1_24 = mm_load_ps(U_data + U_offset_och_ich_1 + 24);
											u_1_32 = mm_load_ps(U_data + U_offset_och_ich_1 + 32);
											u_1_40 = mm_load_ps(U_data + U_offset_och_ich_1 + 40);
											u_1_48 = mm_load_ps(U_data + U_offset_och_ich_1 + 48);
											u_1_56 = mm_load_ps(U_data + U_offset_och_ich_1 + 56);

											u_2_0 = mm_load_ps(U_data + U_offset_och_ich_2);
											u_2_8 = mm_load_ps(U_data + U_offset_och_ich_2 + 8);
											u_2_16 = mm_load_ps(U_data + U_offset_och_ich_2 + 16);
											u_2_24 = mm_load_ps(U_data + U_offset_och_ich_2 + 24);
											u_2_32 = mm_load_ps(U_data + U_offset_och_ich_2 + 32);
											u_2_40 = mm_load_ps(U_data + U_offset_och_ich_2 + 40);
											u_2_48 = mm_load_ps(U_data + U_offset_och_ich_2 + 48);
											u_2_56 = mm_load_ps(U_data + U_offset_och_ich_2 + 56);

											u_3_0 = mm_load_ps(U_data + U_offset_och_ich_3);
											u_3_8 = mm_load_ps(U_data + U_offset_och_ich_3 + 8);
											u_3_16 = mm_load_ps(U_data + U_offset_och_ich_3 + 16);
											u_3_24 = mm_load_ps(U_data + U_offset_och_ich_3 + 24);
											u_3_32 = mm_load_ps(U_data + U_offset_och_ich_3 + 32);
											u_3_40 = mm_load_ps(U_data + U_offset_och_ich_3 + 40);
											u_3_48 = mm_load_ps(U_data + U_offset_och_ich_3 + 48);
											u_3_56 = mm_load_ps(U_data + U_offset_och_ich_3 + 56);

											u_4_0 = mm_load_ps(U_data + U_offset_och_ich_4);
											u_4_8 = mm_load_ps(U_data + U_offset_och_ich_4 + 8);
											u_4_16 = mm_load_ps(U_data + U_offset_och_ich_4 + 16);
											u_4_24 = mm_load_ps(U_data + U_offset_och_ich_4 + 24);
											u_4_32 = mm_load_ps(U_data + U_offset_och_ich_4 + 32);
											u_4_40 = mm_load_ps(U_data + U_offset_och_ich_4 + 40);
											u_4_48 = mm_load_ps(U_data + U_offset_och_ich_4 + 48);
											u_4_56 = mm_load_ps(U_data + U_offset_och_ich_4 + 56);


											v_1_0 = mm_load_ps(V_data + V_offset_ich_row_col_1);
											v_1_8 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 8);
											v_2_0 = mm_load_ps(V_data + V_offset_ich_row_col_2);
											v_2_8 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 8);
											v_3_0 = mm_load_ps(V_data + V_offset_ich_row_col_3);
											v_3_8 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 8);
											v_4_0 = mm_load_ps(V_data + V_offset_ich_row_col_4);
											v_4_8 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 8);

											sum_1_0 = mm_fmadd_ps(v_1_0, u_1_0, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_1_8, u_1_8, sum_1_8);
											sum_1_0 = mm_fmadd_ps(v_2_0, u_1_16, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_2_8, u_1_24, sum_1_8);
											sum_1_0 = mm_fmadd_ps(v_3_0, u_1_32, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_3_8, u_1_40, sum_1_8);
											sum_1_0 = mm_fmadd_ps(v_4_0, u_1_48, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_4_8, u_1_56, sum_1_8);

											sum_2_0 = mm_fmadd_ps(v_1_0, u_2_0, sum_2_0);
											sum_2_8 = mm_fmadd_ps(v_1_8, u_2_8, sum_2_8);
											sum_2_0 = mm_fmadd_ps(v_2_0, u_2_16, sum_2_0);
											sum_2_8 = mm_fmadd_ps(v_2_8, u_2_24, sum_2_8);
											sum_2_0 = mm_fmadd_ps(v_3_0, u_2_32, sum_2_0);
											sum_2_8 = mm_fmadd_ps(v_3_8, u_2_40, sum_2_8);
											sum_2_0 = mm_fmadd_ps(v_4_0, u_2_48, sum_2_0);
											sum_2_8 = mm_fmadd_ps(v_4_8, u_2_56, sum_2_8);

											sum_3_0 = mm_fmadd_ps(v_1_0, u_3_0, sum_3_0);
											sum_3_8 = mm_fmadd_ps(v_1_8, u_3_8, sum_3_8);
											sum_3_0 = mm_fmadd_ps(v_2_0, u_3_16, sum_3_0);
											sum_3_8 = mm_fmadd_ps(v_2_8, u_3_24, sum_3_8);
											sum_3_0 = mm_fmadd_ps(v_3_0, u_3_32, sum_3_0);
											sum_3_8 = mm_fmadd_ps(v_3_8, u_3_40, sum_3_8);
											sum_3_0 = mm_fmadd_ps(v_4_0, u_3_48, sum_3_0);
											sum_3_8 = mm_fmadd_ps(v_4_8, u_3_56, sum_3_8);

											sum_4_0 = mm_fmadd_ps(v_1_0, u_4_0, sum_4_0);
											sum_4_8 = mm_fmadd_ps(v_1_8, u_4_8, sum_4_8);
											sum_4_0 = mm_fmadd_ps(v_2_0, u_4_16, sum_4_0);
											sum_4_8 = mm_fmadd_ps(v_2_8, u_4_24, sum_4_8);
											sum_4_0 = mm_fmadd_ps(v_3_0, u_4_32, sum_4_0);
											sum_4_8 = mm_fmadd_ps(v_3_8, u_4_40, sum_4_8);
											sum_4_0 = mm_fmadd_ps(v_4_0, u_4_48, sum_4_0);
											sum_4_8 = mm_fmadd_ps(v_4_8, u_4_56, sum_4_8);

#elif SIMD_TYPE >= SIMDTYPE_SSE
											u_1_0 = mm_load_ps(U_data + U_offset_och_ich_1);
											u_1_4 = mm_load_ps(U_data + U_offset_och_ich_1 + 4);
											u_1_8 = mm_load_ps(U_data + U_offset_och_ich_1 + 8);
											u_1_12 = mm_load_ps(U_data + U_offset_och_ich_1 + 12);
											u_1_16 = mm_load_ps(U_data + U_offset_och_ich_1 + 16);
											u_1_20 = mm_load_ps(U_data + U_offset_och_ich_1 + 20);
											u_1_24 = mm_load_ps(U_data + U_offset_och_ich_1 + 24);
											u_1_28 = mm_load_ps(U_data + U_offset_och_ich_1 + 28);
											u_1_32 = mm_load_ps(U_data + U_offset_och_ich_1 + 32);
											u_1_36 = mm_load_ps(U_data + U_offset_och_ich_1 + 36);
											u_1_40 = mm_load_ps(U_data + U_offset_och_ich_1 + 40);
											u_1_44 = mm_load_ps(U_data + U_offset_och_ich_1 + 44);
											u_1_48 = mm_load_ps(U_data + U_offset_och_ich_1 + 48);
											u_1_52 = mm_load_ps(U_data + U_offset_och_ich_1 + 52);
											u_1_56 = mm_load_ps(U_data + U_offset_och_ich_1 + 56);
											u_1_60 = mm_load_ps(U_data + U_offset_och_ich_1 + 60);

											u_2_0 = mm_load_ps(U_data + U_offset_och_ich_2);
											u_2_4 = mm_load_ps(U_data + U_offset_och_ich_2 + 4);
											u_2_8 = mm_load_ps(U_data + U_offset_och_ich_2 + 8);
											u_2_12 = mm_load_ps(U_data + U_offset_och_ich_2 + 12);
											u_2_16 = mm_load_ps(U_data + U_offset_och_ich_2 + 16);
											u_2_20 = mm_load_ps(U_data + U_offset_och_ich_2 + 20);
											u_2_24 = mm_load_ps(U_data + U_offset_och_ich_2 + 24);
											u_2_28 = mm_load_ps(U_data + U_offset_och_ich_2 + 28);
											u_2_32 = mm_load_ps(U_data + U_offset_och_ich_2 + 32);
											u_2_36 = mm_load_ps(U_data + U_offset_och_ich_2 + 36);
											u_2_40 = mm_load_ps(U_data + U_offset_och_ich_2 + 40);
											u_2_44 = mm_load_ps(U_data + U_offset_och_ich_2 + 44);
											u_2_48 = mm_load_ps(U_data + U_offset_och_ich_2 + 48);
											u_2_52 = mm_load_ps(U_data + U_offset_och_ich_2 + 52);
											u_2_56 = mm_load_ps(U_data + U_offset_och_ich_2 + 56);
											u_2_60 = mm_load_ps(U_data + U_offset_och_ich_2 + 60);

											u_3_0 = mm_load_ps(U_data + U_offset_och_ich_3);
											u_3_4 = mm_load_ps(U_data + U_offset_och_ich_3 + 4);
											u_3_8 = mm_load_ps(U_data + U_offset_och_ich_3 + 8);
											u_3_12 = mm_load_ps(U_data + U_offset_och_ich_3 + 12);
											u_3_16 = mm_load_ps(U_data + U_offset_och_ich_3 + 16);
											u_3_20 = mm_load_ps(U_data + U_offset_och_ich_3 + 20);
											u_3_24 = mm_load_ps(U_data + U_offset_och_ich_3 + 24);
											u_3_28 = mm_load_ps(U_data + U_offset_och_ich_3 + 28);
											u_3_32 = mm_load_ps(U_data + U_offset_och_ich_3 + 32);
											u_3_36 = mm_load_ps(U_data + U_offset_och_ich_3 + 36);
											u_3_40 = mm_load_ps(U_data + U_offset_och_ich_3 + 40);
											u_3_44 = mm_load_ps(U_data + U_offset_och_ich_3 + 44);
											u_3_48 = mm_load_ps(U_data + U_offset_och_ich_3 + 48);
											u_3_52 = mm_load_ps(U_data + U_offset_och_ich_3 + 52);
											u_3_56 = mm_load_ps(U_data + U_offset_och_ich_3 + 56);
											u_3_60 = mm_load_ps(U_data + U_offset_och_ich_3 + 60);

											u_4_0 = mm_load_ps(U_data + U_offset_och_ich_4);
											u_4_4 = mm_load_ps(U_data + U_offset_och_ich_4 + 4);
											u_4_8 = mm_load_ps(U_data + U_offset_och_ich_4 + 8);
											u_4_12 = mm_load_ps(U_data + U_offset_och_ich_4 + 12);
											u_4_16 = mm_load_ps(U_data + U_offset_och_ich_4 + 16);
											u_4_20 = mm_load_ps(U_data + U_offset_och_ich_4 + 20);
											u_4_24 = mm_load_ps(U_data + U_offset_och_ich_4 + 24);
											u_4_28 = mm_load_ps(U_data + U_offset_och_ich_4 + 28);
											u_4_32 = mm_load_ps(U_data + U_offset_och_ich_4 + 32);
											u_4_36 = mm_load_ps(U_data + U_offset_och_ich_4 + 36);
											u_4_40 = mm_load_ps(U_data + U_offset_och_ich_4 + 40);
											u_4_44 = mm_load_ps(U_data + U_offset_och_ich_4 + 44);
											u_4_48 = mm_load_ps(U_data + U_offset_och_ich_4 + 48);
											u_4_52 = mm_load_ps(U_data + U_offset_och_ich_4 + 52);
											u_4_56 = mm_load_ps(U_data + U_offset_och_ich_4 + 56);
											u_4_60 = mm_load_ps(U_data + U_offset_och_ich_4 + 60);

											v_1_0 = mm_load_ps(V_data + V_offset_ich_row_col_1);
											v_1_4 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 4);
											v_1_8 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 8);
											v_1_12 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 12);
											v_2_0 = mm_load_ps(V_data + V_offset_ich_row_col_2);
											v_2_4 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 4);
											v_2_8 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 8);
											v_2_12 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 12);
											v_3_0 = mm_load_ps(V_data + V_offset_ich_row_col_3);
											v_3_4 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 4);
											v_3_8 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 8);
											v_3_12 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 12);
											v_4_0 = mm_load_ps(V_data + V_offset_ich_row_col_4);
											v_4_4 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 4);
											v_4_8 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 8);
											v_4_12 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 12);

											sum_1_0 = mm_fmadd_ps(v_1_0, u_1_0, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_1_4, u_1_4, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_1_8, u_1_8, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_1_12, u_1_12, sum_1_12);
											sum_1_0 = mm_fmadd_ps(v_2_0, u_1_16, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_2_4, u_1_20, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_2_8, u_1_24, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_2_12, u_1_28, sum_1_12);
											sum_1_0 = mm_fmadd_ps(v_3_0, u_1_32, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_3_4, u_1_36, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_3_8, u_1_40, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_3_12, u_1_44, sum_1_12);
											sum_1_0 = mm_fmadd_ps(v_4_0, u_1_48, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_4_4, u_1_52, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_4_8, u_1_56, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_4_12, u_1_60, sum_1_12);

											sum_2_0 = mm_fmadd_ps(v_1_0, u_2_0, sum_2_0);
											sum_2_4 = mm_fmadd_ps(v_1_4, u_2_4, sum_2_4);
											sum_2_8 = mm_fmadd_ps(v_1_8, u_2_8, sum_2_8);
											sum_2_12 = mm_fmadd_ps(v_1_12, u_2_12, sum_2_12);
											sum_2_0 = mm_fmadd_ps(v_2_0, u_2_16, sum_2_0);
											sum_2_4 = mm_fmadd_ps(v_2_4, u_2_20, sum_2_4);
											sum_2_8 = mm_fmadd_ps(v_2_8, u_2_24, sum_2_8);
											sum_2_12 = mm_fmadd_ps(v_2_12, u_2_28, sum_2_12);
											sum_2_0 = mm_fmadd_ps(v_3_0, u_2_32, sum_2_0);
											sum_2_4 = mm_fmadd_ps(v_3_4, u_2_36, sum_2_4);
											sum_2_8 = mm_fmadd_ps(v_3_8, u_2_40, sum_2_8);
											sum_2_12 = mm_fmadd_ps(v_3_12, u_2_44, sum_2_12);
											sum_2_0 = mm_fmadd_ps(v_4_0, u_2_48, sum_2_0);
											sum_2_4 = mm_fmadd_ps(v_4_4, u_2_52, sum_2_4);
											sum_2_8 = mm_fmadd_ps(v_4_8, u_2_56, sum_2_8);
											sum_2_12 = mm_fmadd_ps(v_4_12, u_2_60, sum_2_12);

											sum_3_0 = mm_fmadd_ps(v_1_0, u_3_0, sum_3_0);
											sum_3_4 = mm_fmadd_ps(v_1_4, u_3_4, sum_3_4);
											sum_3_8 = mm_fmadd_ps(v_1_8, u_3_8, sum_3_8);
											sum_3_12 = mm_fmadd_ps(v_1_12, u_3_12, sum_3_12);
											sum_3_0 = mm_fmadd_ps(v_2_0, u_3_16, sum_3_0);
											sum_3_4 = mm_fmadd_ps(v_2_4, u_3_20, sum_3_4);
											sum_3_8 = mm_fmadd_ps(v_2_8, u_3_24, sum_3_8);
											sum_3_12 = mm_fmadd_ps(v_2_12, u_3_28, sum_3_12);
											sum_3_0 = mm_fmadd_ps(v_3_0, u_3_32, sum_3_0);
											sum_3_4 = mm_fmadd_ps(v_3_4, u_3_36, sum_3_4);
											sum_3_8 = mm_fmadd_ps(v_3_8, u_3_40, sum_3_8);
											sum_3_12 = mm_fmadd_ps(v_3_12, u_3_44, sum_3_12);
											sum_3_0 = mm_fmadd_ps(v_4_0, u_3_48, sum_3_0);
											sum_3_4 = mm_fmadd_ps(v_4_4, u_3_52, sum_3_4);
											sum_3_8 = mm_fmadd_ps(v_4_8, u_3_56, sum_3_8);
											sum_3_12 = mm_fmadd_ps(v_4_12, u_3_60, sum_3_12);

											sum_4_0 = mm_fmadd_ps(v_1_0, u_4_0, sum_4_0);
											sum_4_4 = mm_fmadd_ps(v_1_4, u_4_4, sum_4_4);
											sum_4_8 = mm_fmadd_ps(v_1_8, u_4_8, sum_4_8);
											sum_4_12 = mm_fmadd_ps(v_1_12, u_4_12, sum_4_12);
											sum_4_0 = mm_fmadd_ps(v_2_0, u_4_16, sum_4_0);
											sum_4_4 = mm_fmadd_ps(v_2_4, u_4_20, sum_4_4);
											sum_4_8 = mm_fmadd_ps(v_2_8, u_4_24, sum_4_8);
											sum_4_12 = mm_fmadd_ps(v_2_12, u_4_28, sum_4_12);
											sum_4_0 = mm_fmadd_ps(v_3_0, u_4_32, sum_4_0);
											sum_4_4 = mm_fmadd_ps(v_3_4, u_4_36, sum_4_4);
											sum_4_8 = mm_fmadd_ps(v_3_8, u_4_40, sum_4_8);
											sum_4_12 = mm_fmadd_ps(v_3_12, u_4_44, sum_4_12);
											sum_4_0 = mm_fmadd_ps(v_4_0, u_4_48, sum_4_0);
											sum_4_4 = mm_fmadd_ps(v_4_4, u_4_52, sum_4_4);
											sum_4_8 = mm_fmadd_ps(v_4_8, u_4_56, sum_4_8);
											sum_4_12 = mm_fmadd_ps(v_4_12, u_4_60, sum_4_12);
#else
											for (int i = 0; i < tile_length_; i++)
											{
												mult_data1[i] += U_data[U_offset_och_ich_1 + i] * V_data[V_offset_ich_row_col_1 + i];
												mult_data1[i] += U_data[U_offset_och_ich_1 + tile_length_ + i] * V_data[V_offset_ich_row_col_2 + i];
												mult_data1[i] += U_data[U_offset_och_ich_1 + 2 * tile_length_ + i] * V_data[V_offset_ich_row_col_3 + i];
												mult_data1[i] += U_data[U_offset_och_ich_1 + 3 * tile_length_ + i] * V_data[V_offset_ich_row_col_4 + i];

												mult_data2[i] += U_data[U_offset_och_ich_2 + i] * V_data[V_offset_ich_row_col_1 + i];
												mult_data2[i] += U_data[U_offset_och_ich_2 + tile_length_ + i] * V_data[V_offset_ich_row_col_2 + i];
												mult_data2[i] += U_data[U_offset_och_ich_2 + 2 * tile_length_ + i] * V_data[V_offset_ich_row_col_3 + i];
												mult_data2[i] += U_data[U_offset_och_ich_2 + 3 * tile_length_ + i] * V_data[V_offset_ich_row_col_4 + i];

												mult_data3[i] += U_data[U_offset_och_ich_3 + i] * V_data[V_offset_ich_row_col_1 + i];
												mult_data3[i] += U_data[U_offset_och_ich_3 + tile_length_ + i] * V_data[V_offset_ich_row_col_2 + i];
												mult_data3[i] += U_data[U_offset_och_ich_3 + 2 * tile_length_ + i] * V_data[V_offset_ich_row_col_3 + i];
												mult_data3[i] += U_data[U_offset_och_ich_3 + 3 * tile_length_ + i] * V_data[V_offset_ich_row_col_4 + i];

												mult_data4[i] += U_data[U_offset_och_ich_4 + i] * V_data[V_offset_ich_row_col_1 + i];
												mult_data4[i] += U_data[U_offset_och_ich_4 + tile_length_ + i] * V_data[V_offset_ich_row_col_2 + i];
												mult_data4[i] += U_data[U_offset_och_ich_4 + 2 * tile_length_ + i] * V_data[V_offset_ich_row_col_3 + i];
												mult_data4[i] += U_data[U_offset_och_ich_4 + 3 * tile_length_ + i] * V_data[V_offset_ich_row_col_4 + i];
											}
#endif
										}

										for (; ich < input_Channel_; ich++)
										{
											int offset = ich * tile_length_;
											int U_offset_och_ich_1 = U_offset_och_1 + offset;
											int U_offset_och_ich_2 = U_offset_och_2 + offset;
											int U_offset_och_ich_3 = U_offset_och_3 + offset;
											int U_offset_och_ich_4 = U_offset_och_4 + offset;

											int V_offset_ich_row_col_1 = V_offset_row_col + ich * h_w_tile_stride;

#if SIMD_TYPE >= SIMDTYPE_AVX
											u_1_0 = mm_load_ps(U_data + U_offset_och_ich_1);
											u_1_8 = mm_load_ps(U_data + U_offset_och_ich_1 + 8);
											u_2_0 = mm_load_ps(U_data + U_offset_och_ich_2);
											u_2_8 = mm_load_ps(U_data + U_offset_och_ich_2 + 8);
											u_3_0 = mm_load_ps(U_data + U_offset_och_ich_3);
											u_3_8 = mm_load_ps(U_data + U_offset_och_ich_3 + 8);
											u_4_0 = mm_load_ps(U_data + U_offset_och_ich_4);
											u_4_8 = mm_load_ps(U_data + U_offset_och_ich_4 + 8);
											v_1_0 = mm_load_ps(V_data + V_offset_ich_row_col_1);
											v_1_8 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 8);

											sum_1_0 = mm_fmadd_ps(v_1_0, u_1_0, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_1_8, u_1_8, sum_1_8);
											sum_2_0 = mm_fmadd_ps(v_1_0, u_2_0, sum_2_0);
											sum_2_8 = mm_fmadd_ps(v_1_8, u_2_8, sum_2_8);
											sum_3_0 = mm_fmadd_ps(v_1_0, u_3_0, sum_3_0);
											sum_3_8 = mm_fmadd_ps(v_1_8, u_3_8, sum_3_8);
											sum_4_0 = mm_fmadd_ps(v_1_0, u_4_0, sum_4_0);
											sum_4_8 = mm_fmadd_ps(v_1_8, u_4_8, sum_4_8);
#elif SIMD_TYPE >= SIMDTYPE_SSE
											u_1_0 = mm_load_ps(U_data + U_offset_och_ich_1);
											u_1_4 = mm_load_ps(U_data + U_offset_och_ich_1 + 4);
											u_1_8 = mm_load_ps(U_data + U_offset_och_ich_1 + 8);
											u_1_12 = mm_load_ps(U_data + U_offset_och_ich_1 + 12);

											u_2_0 = mm_load_ps(U_data + U_offset_och_ich_2);
											u_2_4 = mm_load_ps(U_data + U_offset_och_ich_2 + 4);
											u_2_8 = mm_load_ps(U_data + U_offset_och_ich_2 + 8);
											u_2_12 = mm_load_ps(U_data + U_offset_och_ich_2 + 12);

											u_3_0 = mm_load_ps(U_data + U_offset_och_ich_3);
											u_3_4 = mm_load_ps(U_data + U_offset_och_ich_3 + 4);
											u_3_8 = mm_load_ps(U_data + U_offset_och_ich_3 + 8);
											u_3_12 = mm_load_ps(U_data + U_offset_och_ich_3 + 12);

											u_4_0 = mm_load_ps(U_data + U_offset_och_ich_4);
											u_4_4 = mm_load_ps(U_data + U_offset_och_ich_4 + 4);
											u_4_8 = mm_load_ps(U_data + U_offset_och_ich_4 + 8);
											u_4_12 = mm_load_ps(U_data + U_offset_och_ich_4 + 12);

											v_1_0 = mm_load_ps(V_data + V_offset_ich_row_col_1);
											v_1_4 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 4);
											v_1_8 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 8);
											v_1_12 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 12);

											sum_1_0 = mm_fmadd_ps(v_1_0, u_1_0, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_1_4, u_1_4, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_1_8, u_1_8, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_1_12, u_1_12, sum_1_12);

											sum_2_0 = mm_fmadd_ps(v_1_0, u_2_0, sum_2_0);
											sum_2_4 = mm_fmadd_ps(v_1_4, u_2_4, sum_2_4);
											sum_2_8 = mm_fmadd_ps(v_1_8, u_2_8, sum_2_8);
											sum_2_12 = mm_fmadd_ps(v_1_12, u_2_12, sum_2_12);

											sum_3_0 = mm_fmadd_ps(v_1_0, u_3_0, sum_3_0);
											sum_3_4 = mm_fmadd_ps(v_1_4, u_3_4, sum_3_4);
											sum_3_8 = mm_fmadd_ps(v_1_8, u_3_8, sum_3_8);
											sum_3_12 = mm_fmadd_ps(v_1_12, u_3_12, sum_3_12);

											sum_4_0 = mm_fmadd_ps(v_1_0, u_4_0, sum_4_0);
											sum_4_4 = mm_fmadd_ps(v_1_4, u_4_4, sum_4_4);
											sum_4_8 = mm_fmadd_ps(v_1_8, u_4_8, sum_4_8);
											sum_4_12 = mm_fmadd_ps(v_1_12, u_4_12, sum_4_12);
#else
											for (int i = 0; i < tile_length_; i++)
											{
												mult_data1[i] += U_data[U_offset_och_ich_1 + i] * V_data[V_offset_ich_row_col_1 + i];
												mult_data2[i] += U_data[U_offset_och_ich_2 + i] * V_data[V_offset_ich_row_col_1 + i];
												mult_data3[i] += U_data[U_offset_och_ich_3 + i] * V_data[V_offset_ich_row_col_1 + i];
												mult_data4[i] += U_data[U_offset_och_ich_4 + i] * V_data[V_offset_ich_row_col_1 + i];
											}
#endif
										}

#if SIMD_TYPE >= SIMDTYPE_AVX
										mm_store_ps(mult_data1, sum_1_0);
										mm_store_ps(mult_data1 + 8, sum_1_8);
										mm_store_ps(mult_data2, sum_2_0);
										mm_store_ps(mult_data2 + 8, sum_2_8);
										mm_store_ps(mult_data3, sum_3_0);
										mm_store_ps(mult_data3 + 8, sum_3_8);
										mm_store_ps(mult_data4, sum_4_0);
										mm_store_ps(mult_data4 + 8, sum_4_8);
#elif SIMD_TYPE >= SIMDTYPE_SSE
										mm_store_ps(mult_data1, sum_1_0);
										mm_store_ps(mult_data1 + 4, sum_1_4);
										mm_store_ps(mult_data1 + 8, sum_1_8);
										mm_store_ps(mult_data1 + 12, sum_1_12);
										mm_store_ps(mult_data2, sum_2_0);
										mm_store_ps(mult_data2 + 4, sum_2_4);
										mm_store_ps(mult_data2 + 8, sum_2_8);
										mm_store_ps(mult_data2 + 12, sum_2_12);
										mm_store_ps(mult_data3, sum_3_0);
										mm_store_ps(mult_data3 + 4, sum_3_4);
										mm_store_ps(mult_data3 + 8, sum_3_8);
										mm_store_ps(mult_data3 + 12, sum_3_12);
										mm_store_ps(mult_data4, sum_4_0);
										mm_store_ps(mult_data4 + 4, sum_4_4);
										mm_store_ps(mult_data4 + 8, sum_4_8);
										mm_store_ps(mult_data4 + 12, sum_4_12);
#endif

										calculate_ATmA23(mult_data1, result1);
										calculate_ATmA23(mult_data2, result2);
										calculate_ATmA23(mult_data3, result3);
										calculate_ATmA23(mult_data4, result4);

										int row_in_output_data = i / w_tile_num_ * m_;
										int col_in_output_data = i % w_tile_num_* m_;
										int top_offset_row_col = row_in_output_data * output_dim_w_ + col_in_output_data;
										int top_offset_num_och_row_col_1 = top_offset_num_och_1 + top_offset_row_col;
										int top_offset_num_och_row_col_2 = top_offset_num_och_2 + top_offset_row_col;
										int top_offset_num_och_row_col_3 = top_offset_num_och_3 + top_offset_row_col;
										int top_offset_num_och_row_col_4 = top_offset_num_och_4 + top_offset_row_col;

										for (int row = 0; row < m_; row++)
										{
											int result_offset_row = row * m_;
											for (int col = 0; col < m_; col++)
											{
												top_data[top_offset_num_och_row_col_1 + col] = result1[result_offset_row + col] + bias1;
												top_data[top_offset_num_och_row_col_2 + col] = result2[result_offset_row + col] + bias2;
												top_data[top_offset_num_och_row_col_3 + col] = result3[result_offset_row + col] + bias3;
												top_data[top_offset_num_och_row_col_4 + col] = result4[result_offset_row + col] + bias4;
											}
											top_offset_num_och_row_col_1 += output_dim_w_;
											top_offset_num_och_row_col_2 += output_dim_w_;
											top_offset_num_och_row_col_3 += output_dim_w_;
											top_offset_num_och_row_col_4 += output_dim_w_;
										}
									}
								}

								for (int och = remain_outch_start; och < output_Channel_; och++)
								{
#if SIMD_TYPE >= SIMDTYPE_AVX
									mm_type sum_1_0;
									mm_type sum_1_8;
									mm_type u_1_0;
									mm_type u_1_8;
									mm_type u_1_16;
									mm_type u_1_24;
									mm_type u_1_32;
									mm_type u_1_40;
									mm_type u_1_48;
									mm_type u_1_56;
									mm_type v_1_0;
									mm_type v_1_8;
									mm_type v_2_0;
									mm_type v_2_8;
									mm_type v_3_0;
									mm_type v_3_8;
									mm_type v_4_0;
									mm_type v_4_8;
#elif SIMD_TYPE >= SIMDTYPE_SSE
									mm_type sum_1_0;
									mm_type sum_1_4;
									mm_type sum_1_8;
									mm_type sum_1_12;
									mm_type u_1_0;
									mm_type u_1_4;
									mm_type u_1_8;
									mm_type u_1_12;
									mm_type u_1_16;
									mm_type u_1_20;
									mm_type u_1_24;
									mm_type u_1_28;
									mm_type u_1_32;
									mm_type u_1_36;
									mm_type u_1_40;
									mm_type u_1_44;
									mm_type u_1_48;
									mm_type u_1_52;
									mm_type u_1_56;
									mm_type u_1_60;
									mm_type v_1_0;
									mm_type v_1_4;
									mm_type v_1_8;
									mm_type v_1_12;
									mm_type v_2_0;
									mm_type v_2_4;
									mm_type v_2_8;
									mm_type v_2_12;
									mm_type v_3_0;
									mm_type v_3_4;
									mm_type v_3_8;
									mm_type v_3_12;
									mm_type v_4_0;
									mm_type v_4_4;
									mm_type v_4_8;
									mm_type v_4_12;
#endif

									int U_offset_och_1 = och * U_offset_single_och;

									float bias1 = bias_data[och];
									float result1[4];
									int top_offset_num_och_1 = top_offset_num + och * output_spatial_dim_;

									for (int i = 0; i < total_tile_num; i++)
									{
										float mult_data1[16] = { 0 };
										int V_offset_row_col = i * tile_length_;

#if SIMD_TYPE >= SIMDTYPE_AVX
										sum_1_0 = mm_setzero_ps();
										sum_1_8 = mm_setzero_ps();
#elif SIMD_TYPE >= SIMDTYPE_SSE
										sum_1_0 = mm_setzero_ps();
										sum_1_4 = mm_setzero_ps();
										sum_1_8 = mm_setzero_ps();
										sum_1_12 = mm_setzero_ps();
#endif // SIMD_TYPE >= SIMDTYPE_AVX
										int ich = 0;
										for (; ich + 3 < input_Channel_; ich += 4)
										{
											int U_offset_och_ich_1 = U_offset_och_1 + ich * tile_length_;

											int V_offset_ich_row_col_1 = V_offset_row_col + ich * h_w_tile_stride;
											int V_offset_ich_row_col_2 = V_offset_ich_row_col_1 + h_w_tile_stride;
											int V_offset_ich_row_col_3 = V_offset_ich_row_col_2 + h_w_tile_stride;
											int V_offset_ich_row_col_4 = V_offset_ich_row_col_3 + h_w_tile_stride;

#if SIMD_TYPE >= SIMDTYPE_AVX
											u_1_0 = mm_load_ps(U_data + U_offset_och_ich_1);
											u_1_8 = mm_load_ps(U_data + U_offset_och_ich_1 + 8);
											u_1_16 = mm_load_ps(U_data + U_offset_och_ich_1 + 16);
											u_1_24 = mm_load_ps(U_data + U_offset_och_ich_1 + 24);
											u_1_32 = mm_load_ps(U_data + U_offset_och_ich_1 + 32);
											u_1_40 = mm_load_ps(U_data + U_offset_och_ich_1 + 40);
											u_1_48 = mm_load_ps(U_data + U_offset_och_ich_1 + 48);
											u_1_56 = mm_load_ps(U_data + U_offset_och_ich_1 + 56);

											v_1_0 = mm_load_ps(V_data + V_offset_ich_row_col_1);
											v_1_8 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 8);
											v_2_0 = mm_load_ps(V_data + V_offset_ich_row_col_2);
											v_2_8 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 8);
											v_3_0 = mm_load_ps(V_data + V_offset_ich_row_col_3);
											v_3_8 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 8);
											v_4_0 = mm_load_ps(V_data + V_offset_ich_row_col_4);
											v_4_8 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 8);

											sum_1_0 = mm_fmadd_ps(v_1_0, u_1_0, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_1_8, u_1_8, sum_1_8);
											sum_1_0 = mm_fmadd_ps(v_2_0, u_1_16, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_2_8, u_1_24, sum_1_8);
											sum_1_0 = mm_fmadd_ps(v_3_0, u_1_32, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_3_8, u_1_40, sum_1_8);
											sum_1_0 = mm_fmadd_ps(v_4_0, u_1_48, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_4_8, u_1_56, sum_1_8);

#elif SIMD_TYPE >= SIMDTYPE_SSE
											u_1_0 = mm_load_ps(U_data + U_offset_och_ich_1);
											u_1_4 = mm_load_ps(U_data + U_offset_och_ich_1 + 4);
											u_1_8 = mm_load_ps(U_data + U_offset_och_ich_1 + 8);
											u_1_12 = mm_load_ps(U_data + U_offset_och_ich_1 + 12);
											u_1_16 = mm_load_ps(U_data + U_offset_och_ich_1 + 16);
											u_1_20 = mm_load_ps(U_data + U_offset_och_ich_1 + 20);
											u_1_24 = mm_load_ps(U_data + U_offset_och_ich_1 + 24);
											u_1_28 = mm_load_ps(U_data + U_offset_och_ich_1 + 28);
											u_1_32 = mm_load_ps(U_data + U_offset_och_ich_1 + 32);
											u_1_36 = mm_load_ps(U_data + U_offset_och_ich_1 + 36);
											u_1_40 = mm_load_ps(U_data + U_offset_och_ich_1 + 40);
											u_1_44 = mm_load_ps(U_data + U_offset_och_ich_1 + 44);
											u_1_48 = mm_load_ps(U_data + U_offset_och_ich_1 + 48);
											u_1_52 = mm_load_ps(U_data + U_offset_och_ich_1 + 52);
											u_1_56 = mm_load_ps(U_data + U_offset_och_ich_1 + 56);
											u_1_60 = mm_load_ps(U_data + U_offset_och_ich_1 + 60);

											v_1_0 = mm_load_ps(V_data + V_offset_ich_row_col_1);
											v_1_4 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 4);
											v_1_8 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 8);
											v_1_12 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 12);
											v_2_0 = mm_load_ps(V_data + V_offset_ich_row_col_2);
											v_2_4 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 4);
											v_2_8 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 8);
											v_2_12 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 12);
											v_3_0 = mm_load_ps(V_data + V_offset_ich_row_col_3);
											v_3_4 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 4);
											v_3_8 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 8);
											v_3_12 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 12);
											v_4_0 = mm_load_ps(V_data + V_offset_ich_row_col_4);
											v_4_4 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 4);
											v_4_8 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 8);
											v_4_12 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 12);

											sum_1_0 = mm_fmadd_ps(v_1_0, u_1_0, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_1_4, u_1_4, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_1_8, u_1_8, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_1_12, u_1_12, sum_1_12);
											sum_1_0 = mm_fmadd_ps(v_2_0, u_1_16, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_2_4, u_1_20, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_2_8, u_1_24, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_2_12, u_1_28, sum_1_12);
											sum_1_0 = mm_fmadd_ps(v_3_0, u_1_32, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_3_4, u_1_36, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_3_8, u_1_40, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_3_12, u_1_44, sum_1_12);
											sum_1_0 = mm_fmadd_ps(v_4_0, u_1_48, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_4_4, u_1_52, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_4_8, u_1_56, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_4_12, u_1_60, sum_1_12);
#else
											for (int i = 0; i < tile_length_; i++)
											{
												mult_data1[i] += U_data[U_offset_och_ich_1 + i] * V_data[V_offset_ich_row_col_1 + i];
												mult_data1[i] += U_data[U_offset_och_ich_1 + tile_length_ + i] * V_data[V_offset_ich_row_col_2 + i];
												mult_data1[i] += U_data[U_offset_och_ich_1 + 2 * tile_length_ + i] * V_data[V_offset_ich_row_col_3 + i];
												mult_data1[i] += U_data[U_offset_och_ich_1 + 3 * tile_length_ + i] * V_data[V_offset_ich_row_col_4 + i];
											}
#endif
										}

										for (; ich < input_Channel_; ich++)
										{
											int U_offset_och_ich_1 = U_offset_och_1 + ich * tile_length_;
											int V_offset_ich_row_col_1 = V_offset_row_col + ich * h_w_tile_stride;

#if SIMD_TYPE >= SIMDTYPE_AVX
											u_1_0 = mm_load_ps(U_data + U_offset_och_ich_1);
											u_1_8 = mm_load_ps(U_data + U_offset_och_ich_1 + 8);

											v_1_0 = mm_load_ps(V_data + V_offset_ich_row_col_1);
											v_1_8 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 8);

											sum_1_0 = mm_fmadd_ps(v_1_0, u_1_0, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_1_8, u_1_8, sum_1_8);
#elif SIMD_TYPE >= SIMDTYPE_SSE
											u_1_0 = mm_load_ps(U_data + U_offset_och_ich_1);
											u_1_4 = mm_load_ps(U_data + U_offset_och_ich_1 + 4);
											u_1_8 = mm_load_ps(U_data + U_offset_och_ich_1 + 8);
											u_1_12 = mm_load_ps(U_data + U_offset_och_ich_1 + 12);

											v_1_0 = mm_load_ps(V_data + V_offset_ich_row_col_1);
											v_1_4 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 4);
											v_1_8 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 8);
											v_1_12 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 12);

											sum_1_0 = mm_fmadd_ps(v_1_0, u_1_0, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_1_4, u_1_4, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_1_8, u_1_8, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_1_12, u_1_12, sum_1_12);
#else
											for (int i = 0; i < tile_length_; i++)
											{
												mult_data1[i] += U_data[U_offset_och_ich_1 + i] * V_data[V_offset_ich_row_col_1 + i];
											}
#endif
										}


#if SIMD_TYPE >= SIMDTYPE_AVX
										mm_store_ps(mult_data1, sum_1_0);
										mm_store_ps(mult_data1 + 8, sum_1_8);
#elif SIMD_TYPE >= SIMDTYPE_SSE
										mm_store_ps(mult_data1, sum_1_0);
										mm_store_ps(mult_data1 + 4, sum_1_4);
										mm_store_ps(mult_data1 + 8, sum_1_8);
										mm_store_ps(mult_data1 + 12, sum_1_12);
#endif // SIMD_TYPE >= SIMDTYPE_AVX

										calculate_ATmA23(mult_data1, result1);

										int row_in_output_data = i / w_tile_num_ * m_;
										int col_in_output_data = i % w_tile_num_* m_;
										int top_offset_row_col = row_in_output_data * output_dim_w_ + col_in_output_data;
										int top_offset_num_och_row_col_1 = top_offset_num_och_1 + top_offset_row_col;

										for (int row = 0; row < m_; row++)
										{
											int result_offset_row = row * m_;
											for (int col = 0; col < m_; col++)
											{
												top_data[top_offset_num_och_row_col_1 + col] = result1[result_offset_row + col] + bias1;
											}
											top_offset_num_och_row_col_1 += output_dim_w_;
										}
									}
								}
							}
						}

						if ((add_h != 0) || (add_w != 0))
						{
							tensor_operation_cpu::cut_border_cpu(top, top, 0, add_h, 0, add_w);
						}
					}
					else
					{
						LOG(FATAL) << "group wrong!!!";
					}
				}
				else if (order_ == NHWC)
				{
					NOT_IMPLEMENTED;
				}
				else
				{
					NOT_IMPLEMENTED;
				}

				delete V_data;
			}
		}

		void conv_winograd_cpu::Forward_F43(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
		{
			CHECK_EQ(kernelSize_, 3);
			CHECK_EQ(stride_, 1);
			order_ = bottom->order();
			num_ = bottom->num();
			input_dim_h_ = bottom->height();
			input_dim_w_ = bottom->width();
			output_dim_h_ = (input_dim_h_ + 2 * pad_ - kernelSize_) / stride_ + 1;
			output_dim_w_ = (input_dim_w_ + 2 * pad_ - kernelSize_) / stride_ + 1;

			int h_subtract_tilesize = input_dim_h_ + 2 * pad_ - tile_size_;
			int w_subtract_tilesize = input_dim_w_ + 2 * pad_ - tile_size_;
			h_tile_num_ = int((float)h_subtract_tilesize / m_ + 0.5f) + 1;//h_tile_num_ = ceil((H-(m+r-1))/m) + 1, H is height after padding
			w_tile_num_ = int((float)w_subtract_tilesize / m_ + 0.5f) + 1;//w_tile_num_ = ceil((W-(m+r-1))/m) + 1, W is width after padding
			int total_tile_num = h_tile_num_ * w_tile_num_;
			int w_tile_stride = w_tile_num_ * tile_length_;
			int h_w_tile_stride = h_tile_num_ * w_tile_stride;
			int h_aligned = (h_subtract_tilesize + m_ - 1) / m_ * m_;
			int w_aligned = (w_subtract_tilesize + m_ - 1) / m_ * m_;
			int add_h = h_aligned - h_subtract_tilesize;
			int add_w = w_aligned - w_subtract_tilesize;

			input_dim_w_ += 2 * pad_ + add_w;
			input_dim_h_ += 2 * pad_ + add_h;
			output_dim_w_ += add_w;
			output_dim_h_ += add_h;
			input_spatial_dim_ = input_dim_w_ * input_dim_h_;
			output_spatial_dim_ = output_dim_w_ * output_dim_h_;

			std::shared_ptr<tensor<float>> bottom_bordered;
			tensor_operation_cpu::make_border_cpu(bottom, bottom_bordered, pad_, pad_ + add_h, pad_, pad_ + add_w);
			bottom_data = bottom_bordered->cpu_data();
			bottom_dim_ = bottom_bordered->count(1, 4);

			top.reset(new tensor<float>(std::vector<int>{num_, output_Channel_, output_dim_h_, output_dim_w_}, device_, order_));
			top_data = top->mutable_cpu_data();
			top_dim_ = top->count(1, 4);

			if (int8_quantization_)
			{
				NOT_IMPLEMENTED;
			}
			else
			{
				if (order_ == NCHW)
				{
					if (group_ > 1)
					{
						NOT_IMPLEMENTED;
					}
					else if (group_ == 1)
					{
						//V=BT*d*B,so V has the same number as data tile, there are tile_length_ elements in single V
						//V_.reset(new tensor<float>(std::vector<int>{1, input_Channel_, total_tile_num, tile_length_}));
						//V_data = V_->mutable_cpu_data();
						V_data = new float[input_Channel_ * total_tile_num * tile_length_];
						int U_offset_single_och = input_Channel_ * tile_length_;

						for (int n = 0; n < num_; n++)
						{
							int bottom_offset_num = n * bottom_dim_;
							int top_offset_num = n * top_dim_;

							//get transformed bottom_data: V
							{
								int nn_inch = input_Channel_ >> 2;
								int remain_inch_start = nn_inch << 2;

#ifdef _OPENMP
#pragma omp parallel for
#endif
								for (int nn = 0; nn < nn_inch; nn++)
								{
									int ich = nn * 4;
									int bottom_offset_num_ich_1 = bottom_offset_num + ich * input_spatial_dim_;
									int bottom_offset_num_ich_2 = bottom_offset_num_ich_1 + input_spatial_dim_;
									int bottom_offset_num_ich_3 = bottom_offset_num_ich_2 + input_spatial_dim_;
									int bottom_offset_num_ich_4 = bottom_offset_num_ich_3 + input_spatial_dim_;
									int V_offset_ich_1 = ich * h_w_tile_stride;
									int V_offset_ich_2 = V_offset_ich_1 + h_w_tile_stride;
									int V_offset_ich_3 = V_offset_ich_2 + h_w_tile_stride;
									int V_offset_ich_4 = V_offset_ich_3 + h_w_tile_stride;

									for (int i = 0; i < total_tile_num; i++)
									{
										int row_in_input_data = (i / w_tile_num_) * m_;
										int col_in_input_data = (i % w_tile_num_) * m_;
										int bottom_offset_row_col = row_in_input_data * input_dim_w_ + col_in_input_data;
										int bottom_offset_num_ich_row_col_1 = bottom_offset_num_ich_1 + bottom_offset_row_col;
										int bottom_offset_num_ich_row_col_2 = bottom_offset_num_ich_2 + bottom_offset_row_col;
										int bottom_offset_num_ich_row_col_3 = bottom_offset_num_ich_3 + bottom_offset_row_col;
										int bottom_offset_num_ich_row_col_4 = bottom_offset_num_ich_4 + bottom_offset_row_col;

										int tile_offset = i * tile_length_;
										int V_offset_ich_row_col_1 = V_offset_ich_1 + tile_offset;
										int V_offset_ich_row_col_2 = V_offset_ich_2 + tile_offset;
										int V_offset_ich_row_col_3 = V_offset_ich_3 + tile_offset;
										int V_offset_ich_row_col_4 = V_offset_ich_4 + tile_offset;

										const float *row_data1_1 = bottom_data + bottom_offset_num_ich_row_col_1;
										const float *row_data1_2 = row_data1_1 + input_dim_w_;
										const float *row_data1_3 = row_data1_2 + input_dim_w_;
										const float *row_data1_4 = row_data1_3 + input_dim_w_;
										const float *row_data1_5 = row_data1_4 + input_dim_w_;
										const float *row_data1_6 = row_data1_5 + input_dim_w_;
										calculate_BTdB43(row_data1_1, row_data1_2, row_data1_3, row_data1_4, row_data1_5, row_data1_6, V_data + V_offset_ich_row_col_1);

										const float *row_data2_1 = bottom_data + bottom_offset_num_ich_row_col_2;
										const float *row_data2_2 = row_data2_1 + input_dim_w_;
										const float *row_data2_3 = row_data2_2 + input_dim_w_;
										const float *row_data2_4 = row_data2_3 + input_dim_w_;
										const float *row_data2_5 = row_data2_4 + input_dim_w_;
										const float *row_data2_6 = row_data2_5 + input_dim_w_;
										calculate_BTdB43(row_data2_1, row_data2_2, row_data2_3, row_data2_4, row_data2_5, row_data2_6, V_data + V_offset_ich_row_col_2);

										const float *row_data3_1 = bottom_data + bottom_offset_num_ich_row_col_3;
										const float *row_data3_2 = row_data3_1 + input_dim_w_;
										const float *row_data3_3 = row_data3_2 + input_dim_w_;
										const float *row_data3_4 = row_data3_3 + input_dim_w_;
										const float *row_data3_5 = row_data3_4 + input_dim_w_;
										const float *row_data3_6 = row_data3_5 + input_dim_w_;
										calculate_BTdB43(row_data3_1, row_data3_2, row_data3_3, row_data3_4, row_data3_5, row_data3_6, V_data + V_offset_ich_row_col_3);

										const float *row_data4_1 = bottom_data + bottom_offset_num_ich_row_col_4;
										const float *row_data4_2 = row_data4_1 + input_dim_w_;
										const float *row_data4_3 = row_data4_2 + input_dim_w_;
										const float *row_data4_4 = row_data4_3 + input_dim_w_;
										const float *row_data4_5 = row_data4_4 + input_dim_w_;
										const float *row_data4_6 = row_data4_5 + input_dim_w_;
										calculate_BTdB43(row_data4_1, row_data4_2, row_data4_3, row_data4_4, row_data4_5, row_data4_6, V_data + V_offset_ich_row_col_4);
									}
								}

								for (int ich = remain_inch_start; ich < input_Channel_; ich++)
								{
									int bottom_offset_num_ich_1 = bottom_offset_num + ich * input_spatial_dim_;
									int V_offset_ich_1 = ich * h_w_tile_stride;

									for (int i = 0; i < total_tile_num; i++)
									{
										int row_in_input_data = (i / w_tile_num_) * m_;
										int col_in_input_data = (i % w_tile_num_) * m_;
										int bottom_offset_row_col = row_in_input_data * input_dim_w_ + col_in_input_data;
										int bottom_offset_num_ich_row_col_1 = bottom_offset_num_ich_1 + bottom_offset_row_col;

										int tile_offset = i * tile_length_;
										int V_offset_ich_row_col_1 = V_offset_ich_1 + tile_offset;

										const float *row_data1_1 = bottom_data + bottom_offset_num_ich_row_col_1;
										const float *row_data1_2 = row_data1_1 + input_dim_w_;
										const float *row_data1_3 = row_data1_2 + input_dim_w_;
										const float *row_data1_4 = row_data1_3 + input_dim_w_;
										const float *row_data1_5 = row_data1_4 + input_dim_w_;
										const float *row_data1_6 = row_data1_5 + input_dim_w_;
										calculate_BTdB43(row_data1_1, row_data1_2, row_data1_3, row_data1_4, row_data1_5, row_data1_6, V_data + V_offset_ich_row_col_1);
									}
								}
							}

							//multiply
							{
								int nn_outch = output_Channel_ >> 2;
								int remain_outch_start = nn_outch << 2;
#ifdef _OPENMP
#pragma omp parallel for
#endif
								for (int nn = 0; nn < nn_outch; nn++)
								{
#if SIMD_TYPE >= SIMDTYPE_AVX
									mm_type sum_1_0;
									mm_type sum_1_8;
									mm_type sum_1_16;
									mm_type sum_1_24;
									__m128 sum_1_32;
									mm_type sum_2_0;
									mm_type sum_2_8;
									mm_type sum_2_16;
									mm_type sum_2_24;
									__m128 sum_2_32;
									mm_type sum_3_0;
									mm_type sum_3_8;
									mm_type sum_3_16;
									mm_type sum_3_24;
									__m128 sum_3_32;
									mm_type sum_4_0;
									mm_type sum_4_8;
									mm_type sum_4_16;
									mm_type sum_4_24;
									__m128 sum_4_32;

									mm_type u_1_0;
									mm_type u_1_8;
									mm_type u_1_16;
									mm_type u_1_24;
									__m128 u_1_32;

									mm_type u_1_36;
									mm_type u_1_44;
									mm_type u_1_52;
									mm_type u_1_60;
									__m128 u_1_68;

									mm_type u_1_72;
									mm_type u_1_80;
									mm_type u_1_88;
									mm_type u_1_96;
									__m128 u_1_104;

									mm_type u_1_108;
									mm_type u_1_116;
									mm_type u_1_124;
									mm_type u_1_132;
									__m128 u_1_140;

									mm_type u_2_0;
									mm_type u_2_8;
									mm_type u_2_16;
									mm_type u_2_24;
									__m128 u_2_32;

									mm_type u_2_36;
									mm_type u_2_44;
									mm_type u_2_52;
									mm_type u_2_60;
									__m128 u_2_68;

									mm_type u_2_72;
									mm_type u_2_80;
									mm_type u_2_88;
									mm_type u_2_96;
									__m128 u_2_104;

									mm_type u_2_108;
									mm_type u_2_116;
									mm_type u_2_124;
									mm_type u_2_132;
									__m128 u_2_140;

									mm_type u_3_0;
									mm_type u_3_8;
									mm_type u_3_16;
									mm_type u_3_24;
									__m128 u_3_32;

									mm_type u_3_36;
									mm_type u_3_44;
									mm_type u_3_52;
									mm_type u_3_60;
									__m128 u_3_68;

									mm_type u_3_72;
									mm_type u_3_80;
									mm_type u_3_88;
									mm_type u_3_96;
									__m128 u_3_104;

									mm_type u_3_108;
									mm_type u_3_116;
									mm_type u_3_124;
									mm_type u_3_132;
									__m128 u_3_140;

									mm_type u_4_0;
									mm_type u_4_8;
									mm_type u_4_16;
									mm_type u_4_24;
									__m128 u_4_32;

									mm_type u_4_36;
									mm_type u_4_44;
									mm_type u_4_52;
									mm_type u_4_60;
									__m128 u_4_68;

									mm_type u_4_72;
									mm_type u_4_80;
									mm_type u_4_88;
									mm_type u_4_96;
									__m128 u_4_104;

									mm_type u_4_108;
									mm_type u_4_116;
									mm_type u_4_124;
									mm_type u_4_132;
									__m128 u_4_140;

									mm_type v_1_0;
									mm_type v_1_8;
									mm_type v_1_16;
									mm_type v_1_24;
									__m128 v_1_32;
									
									mm_type v_2_0;
									mm_type v_2_8;
									mm_type v_2_16;
									mm_type v_2_24;
									__m128 v_2_32;

									mm_type v_3_0;
									mm_type v_3_8;
									mm_type v_3_16;
									mm_type v_3_24;
									__m128 v_3_32;

									mm_type v_4_0;
									mm_type v_4_8;
									mm_type v_4_16;
									mm_type v_4_24;
									__m128 v_4_32;
#elif SIMD_TYPE >= SIMDTYPE_SSE
									mm_type sum_1_0;
									mm_type sum_1_4;
									mm_type sum_1_8;
									mm_type sum_1_12;
									mm_type sum_2_0;
									mm_type sum_2_4;
									mm_type sum_2_8;
									mm_type sum_2_12;
									mm_type sum_3_0;
									mm_type sum_3_4;
									mm_type sum_3_8;
									mm_type sum_3_12;
									mm_type sum_4_0;
									mm_type sum_4_4;
									mm_type sum_4_8;
									mm_type sum_4_12;
									mm_type u_1_0;
									mm_type u_1_4;
									mm_type u_1_8;
									mm_type u_1_12;
									mm_type u_1_16;
									mm_type u_1_20;
									mm_type u_1_24;
									mm_type u_1_28;
									mm_type u_1_32;
									mm_type u_1_36;
									mm_type u_1_40;
									mm_type u_1_44;
									mm_type u_1_48;
									mm_type u_1_52;
									mm_type u_1_56;
									mm_type u_1_60;
									mm_type u_2_0;
									mm_type u_2_4;
									mm_type u_2_8;
									mm_type u_2_12;
									mm_type u_2_16;
									mm_type u_2_20;
									mm_type u_2_24;
									mm_type u_2_28;
									mm_type u_2_32;
									mm_type u_2_36;
									mm_type u_2_40;
									mm_type u_2_44;
									mm_type u_2_48;
									mm_type u_2_52;
									mm_type u_2_56;
									mm_type u_2_60;
									mm_type u_3_0;
									mm_type u_3_4;
									mm_type u_3_8;
									mm_type u_3_12;
									mm_type u_3_16;
									mm_type u_3_20;
									mm_type u_3_24;
									mm_type u_3_28;
									mm_type u_3_32;
									mm_type u_3_36;
									mm_type u_3_40;
									mm_type u_3_44;
									mm_type u_3_48;
									mm_type u_3_52;
									mm_type u_3_56;
									mm_type u_3_60;
									mm_type u_4_0;
									mm_type u_4_4;
									mm_type u_4_8;
									mm_type u_4_12;
									mm_type u_4_16;
									mm_type u_4_20;
									mm_type u_4_24;
									mm_type u_4_28;
									mm_type u_4_32;
									mm_type u_4_36;
									mm_type u_4_40;
									mm_type u_4_44;
									mm_type u_4_48;
									mm_type u_4_52;
									mm_type u_4_56;
									mm_type u_4_60;
									mm_type v_1_0;
									mm_type v_1_4;
									mm_type v_1_8;
									mm_type v_1_12;
									mm_type v_2_0;
									mm_type v_2_4;
									mm_type v_2_8;
									mm_type v_2_12;
									mm_type v_3_0;
									mm_type v_3_4;
									mm_type v_3_8;
									mm_type v_3_12;
									mm_type v_4_0;
									mm_type v_4_4;
									mm_type v_4_8;
									mm_type v_4_12;
#endif

									int och = nn * 4;
									int U_offset_och_1 = och * U_offset_single_och;
									int U_offset_och_2 = U_offset_och_1 + U_offset_single_och;
									int U_offset_och_3 = U_offset_och_2 + U_offset_single_och;
									int U_offset_och_4 = U_offset_och_3 + U_offset_single_och;

									float bias1 = bias_data[och];
									float bias2 = bias_data[och + 1];
									float bias3 = bias_data[och + 2];
									float bias4 = bias_data[och + 3];
									float result1[16], result2[16], result3[16], result4[16];

									int top_offset_num_och_1 = top_offset_num + och * output_spatial_dim_;
									int top_offset_num_och_2 = top_offset_num_och_1 + output_spatial_dim_;
									int top_offset_num_och_3 = top_offset_num_och_2 + output_spatial_dim_;
									int top_offset_num_och_4 = top_offset_num_och_3 + output_spatial_dim_;

									for (int i = 0; i < total_tile_num; i++)
									{
										float mult_data1[36] = { 0 };
										float mult_data2[36] = { 0 };
										float mult_data3[36] = { 0 };
										float mult_data4[36] = { 0 };

										int V_offset_row_col = i * tile_length_;

#if SIMD_TYPE >= SIMDTYPE_AVX
										sum_1_0 = mm_setzero_ps();
										sum_1_8 = mm_setzero_ps();
										sum_1_16 = mm_setzero_ps();
										sum_1_24 = mm_setzero_ps();
										sum_1_32 = _mm_setzero_ps();
										
										sum_2_0 = mm_setzero_ps();
										sum_2_8 = mm_setzero_ps();
										sum_2_16 = mm_setzero_ps();
										sum_2_24 = mm_setzero_ps();
										sum_2_32 = _mm_setzero_ps();

										sum_3_0 = mm_setzero_ps();
										sum_3_8 = mm_setzero_ps();
										sum_3_16 = mm_setzero_ps();
										sum_3_24 = mm_setzero_ps();
										sum_3_32 = _mm_setzero_ps();

										sum_4_0 = mm_setzero_ps();
										sum_4_8 = mm_setzero_ps();
										sum_4_16 = mm_setzero_ps();
										sum_4_24 = mm_setzero_ps();
										sum_4_32 = _mm_setzero_ps();
#elif SIMD_TYPE >= SIMDTYPE_SSE
										sum_1_0 = mm_setzero_ps();
										sum_1_4 = mm_setzero_ps();
										sum_1_8 = mm_setzero_ps();
										sum_1_12 = mm_setzero_ps();
										sum_2_0 = mm_setzero_ps();
										sum_2_4 = mm_setzero_ps();
										sum_2_8 = mm_setzero_ps();
										sum_2_12 = mm_setzero_ps();
										sum_3_0 = mm_setzero_ps();
										sum_3_4 = mm_setzero_ps();
										sum_3_8 = mm_setzero_ps();
										sum_3_12 = mm_setzero_ps();
										sum_4_0 = mm_setzero_ps();
										sum_4_4 = mm_setzero_ps();
										sum_4_8 = mm_setzero_ps();
										sum_4_12 = mm_setzero_ps();
#endif // SIMD_TYPE >= SIMDTYPE_AVX
										int ich = 0;
										for (; ich + 3 < input_Channel_; ich += 4)
										{
											int offset = ich * tile_length_;
											int U_offset_och_ich_1 = U_offset_och_1 + offset;
											int U_offset_och_ich_2 = U_offset_och_2 + offset;
											int U_offset_och_ich_3 = U_offset_och_3 + offset;
											int U_offset_och_ich_4 = U_offset_och_4 + offset;

											int V_offset_ich_row_col_1 = V_offset_row_col + ich * h_w_tile_stride;
											int V_offset_ich_row_col_2 = V_offset_ich_row_col_1 + h_w_tile_stride;
											int V_offset_ich_row_col_3 = V_offset_ich_row_col_2 + h_w_tile_stride;
											int V_offset_ich_row_col_4 = V_offset_ich_row_col_3 + h_w_tile_stride;

#if SIMD_TYPE >= SIMDTYPE_AVX
											//u_1
											u_1_0 = mm_load_ps(U_data + U_offset_och_ich_1);
											u_1_8 = mm_load_ps(U_data + U_offset_och_ich_1 + 8);
											u_1_16 = mm_load_ps(U_data + U_offset_och_ich_1 + 16);
											u_1_24 = mm_load_ps(U_data + U_offset_och_ich_1 + 24);
											u_1_32 = _mm_loadu_ps(U_data + U_offset_och_ich_1 + 32);

											u_1_36 = mm_load_ps(U_data + U_offset_och_ich_1 + 36);
											u_1_44 = mm_load_ps(U_data + U_offset_och_ich_1 + 44);
											u_1_52 = mm_load_ps(U_data + U_offset_och_ich_1 + 52);
											u_1_60 = mm_load_ps(U_data + U_offset_och_ich_1 + 60);
											u_1_68 = _mm_loadu_ps(U_data + U_offset_och_ich_1 + 68);

											u_1_72 = mm_load_ps(U_data + U_offset_och_ich_1 + 72);
											u_1_80 = mm_load_ps(U_data + U_offset_och_ich_1 + 80);
											u_1_88 = mm_load_ps(U_data + U_offset_och_ich_1 + 88);
											u_1_96 = mm_load_ps(U_data + U_offset_och_ich_1 + 96);
											u_1_104 = _mm_loadu_ps(U_data + U_offset_och_ich_1 + 104);

											u_1_108 = mm_load_ps(U_data + U_offset_och_ich_1 + 108);
											u_1_116 = mm_load_ps(U_data + U_offset_och_ich_1 + 116);
											u_1_124 = mm_load_ps(U_data + U_offset_och_ich_1 + 124);
											u_1_132 = mm_load_ps(U_data + U_offset_och_ich_1 + 132);
											u_1_140 = _mm_loadu_ps(U_data + U_offset_och_ich_1 + 140);

											//u_2
											u_2_0 = mm_load_ps(U_data + U_offset_och_ich_2);
											u_2_8 = mm_load_ps(U_data + U_offset_och_ich_2 + 8);
											u_2_16 = mm_load_ps(U_data + U_offset_och_ich_2 + 16);
											u_2_24 = mm_load_ps(U_data + U_offset_och_ich_2 + 24);
											u_2_32 = _mm_loadu_ps(U_data + U_offset_och_ich_2 + 32);

											u_2_36 = mm_load_ps(U_data + U_offset_och_ich_2 + 36);
											u_2_44 = mm_load_ps(U_data + U_offset_och_ich_2 + 44);
											u_2_52 = mm_load_ps(U_data + U_offset_och_ich_2 + 52);
											u_2_60 = mm_load_ps(U_data + U_offset_och_ich_2 + 60);
											u_2_68 = _mm_loadu_ps(U_data + U_offset_och_ich_2 + 68);

											u_2_72 = mm_load_ps(U_data + U_offset_och_ich_2 + 72);
											u_2_80 = mm_load_ps(U_data + U_offset_och_ich_2 + 80);
											u_2_88 = mm_load_ps(U_data + U_offset_och_ich_2 + 88);
											u_2_96 = mm_load_ps(U_data + U_offset_och_ich_2 + 96);
											u_2_104 = _mm_loadu_ps(U_data + U_offset_och_ich_2 + 104);

											u_2_108 = mm_load_ps(U_data + U_offset_och_ich_2 + 108);
											u_2_116 = mm_load_ps(U_data + U_offset_och_ich_2 + 116);
											u_2_124 = mm_load_ps(U_data + U_offset_och_ich_2 + 124);
											u_2_132 = mm_load_ps(U_data + U_offset_och_ich_2 + 132);
											u_2_140 = _mm_loadu_ps(U_data + U_offset_och_ich_2 + 140);

											//u_3
											u_3_0 = mm_load_ps(U_data + U_offset_och_ich_3);
											u_3_8 = mm_load_ps(U_data + U_offset_och_ich_3 + 8);
											u_3_16 = mm_load_ps(U_data + U_offset_och_ich_3 + 16);
											u_3_24 = mm_load_ps(U_data + U_offset_och_ich_3 + 24);
											u_3_32 = _mm_loadu_ps(U_data + U_offset_och_ich_3 + 32);

											u_3_36 = mm_load_ps(U_data + U_offset_och_ich_3 + 36);
											u_3_44 = mm_load_ps(U_data + U_offset_och_ich_3 + 44);
											u_3_52 = mm_load_ps(U_data + U_offset_och_ich_3 + 52);
											u_3_60 = mm_load_ps(U_data + U_offset_och_ich_3 + 60);
											u_3_68 = _mm_loadu_ps(U_data + U_offset_och_ich_3 + 68);

											u_3_72 = mm_load_ps(U_data + U_offset_och_ich_3 + 72);
											u_3_80 = mm_load_ps(U_data + U_offset_och_ich_3 + 80);
											u_3_88 = mm_load_ps(U_data + U_offset_och_ich_3 + 88);
											u_3_96 = mm_load_ps(U_data + U_offset_och_ich_3 + 96);
											u_3_104 = _mm_loadu_ps(U_data + U_offset_och_ich_3 + 104);

											u_3_108 = mm_load_ps(U_data + U_offset_och_ich_3 + 108);
											u_3_116 = mm_load_ps(U_data + U_offset_och_ich_3 + 116);
											u_3_124 = mm_load_ps(U_data + U_offset_och_ich_3 + 124);
											u_3_132 = mm_load_ps(U_data + U_offset_och_ich_3 + 132);
											u_3_140 = _mm_loadu_ps(U_data + U_offset_och_ich_3 + 140);

											//u_4
											u_4_0 = mm_load_ps(U_data + U_offset_och_ich_4);
											u_4_8 = mm_load_ps(U_data + U_offset_och_ich_4 + 8);
											u_4_16 = mm_load_ps(U_data + U_offset_och_ich_4 + 16);
											u_4_24 = mm_load_ps(U_data + U_offset_och_ich_4 + 24);
											u_4_32 = _mm_loadu_ps(U_data + U_offset_och_ich_4 + 32);

											u_4_36 = mm_load_ps(U_data + U_offset_och_ich_4 + 36);
											u_4_44 = mm_load_ps(U_data + U_offset_och_ich_4 + 44);
											u_4_52 = mm_load_ps(U_data + U_offset_och_ich_4 + 52);
											u_4_60 = mm_load_ps(U_data + U_offset_och_ich_4 + 60);
											u_4_68 = _mm_loadu_ps(U_data + U_offset_och_ich_4 + 68);

											u_4_72 = mm_load_ps(U_data + U_offset_och_ich_4 + 72);
											u_4_80 = mm_load_ps(U_data + U_offset_och_ich_4 + 80);
											u_4_88 = mm_load_ps(U_data + U_offset_och_ich_4 + 88);
											u_4_96 = mm_load_ps(U_data + U_offset_och_ich_4 + 96);
											u_4_104 = _mm_loadu_ps(U_data + U_offset_och_ich_4 + 104);

											u_4_108 = mm_load_ps(U_data + U_offset_och_ich_4 + 108);
											u_4_116 = mm_load_ps(U_data + U_offset_och_ich_4 + 116);
											u_4_124 = mm_load_ps(U_data + U_offset_och_ich_4 + 124);
											u_4_132 = mm_load_ps(U_data + U_offset_och_ich_4 + 132);
											u_4_140 = _mm_loadu_ps(U_data + U_offset_och_ich_4 + 140);

											//v
											v_1_0 = mm_load_ps(V_data + V_offset_ich_row_col_1);
											v_1_8 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 8);
											v_1_16 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 16);
											v_1_24 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 24);
											v_1_32 = _mm_loadu_ps(V_data + V_offset_ich_row_col_1 + 32);

											v_2_0 = mm_load_ps(V_data + V_offset_ich_row_col_2);
											v_2_8 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 8);
											v_2_16 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 16);
											v_2_24 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 24);
											v_2_32 = _mm_loadu_ps(V_data + V_offset_ich_row_col_2 + 32);

											v_3_0 = mm_load_ps(V_data + V_offset_ich_row_col_3);
											v_3_8 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 8);
											v_3_16 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 16);
											v_3_24 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 24);
											v_3_32 = _mm_loadu_ps(V_data + V_offset_ich_row_col_3 + 32);

											v_4_0 = mm_load_ps(V_data + V_offset_ich_row_col_4);
											v_4_8 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 8);
											v_4_16 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 16);
											v_4_24 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 24);
											v_4_32 = _mm_loadu_ps(V_data + V_offset_ich_row_col_4 + 32);

											sum_1_0 = mm_fmadd_ps(v_1_0, u_1_0, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_1_8, u_1_8, sum_1_8);
											sum_1_16 = mm_fmadd_ps(v_1_16, u_1_16, sum_1_16);
											sum_1_24 = mm_fmadd_ps(v_1_24, u_1_24, sum_1_24);
											//sum_1_32 = mm_fmadd_ps(v_1_32, u_1_32, sum_1_32);
											sum_1_32 = _mm_add_ps(_mm_mul_ps(v_1_32, u_1_32), sum_1_32);
											sum_1_0 = mm_fmadd_ps(v_2_0, u_1_36, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_2_8, u_1_44, sum_1_8);
											sum_1_16 = mm_fmadd_ps(v_2_16, u_1_52, sum_1_16);
											sum_1_24 = mm_fmadd_ps(v_2_24, u_1_60, sum_1_24);
											//sum_1_32 = mm_fmadd_ps(v_2_32, u_1_68, sum_1_32);
											sum_1_32 = _mm_add_ps(_mm_mul_ps(v_2_32, u_1_68), sum_1_32);
											sum_1_0 = mm_fmadd_ps(v_3_0, u_1_72, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_3_8, u_1_80, sum_1_8);
											sum_1_16 = mm_fmadd_ps(v_3_16, u_1_88, sum_1_16);
											sum_1_24 = mm_fmadd_ps(v_3_24, u_1_96, sum_1_24);
											//sum_1_32 = mm_fmadd_ps(v_3_32, u_1_104, sum_1_32);
											sum_1_32 = _mm_add_ps(_mm_mul_ps(v_3_32, u_1_104), sum_1_32);
											sum_1_0 = mm_fmadd_ps(v_4_0, u_1_108, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_4_8, u_1_116, sum_1_8);
											sum_1_16 = mm_fmadd_ps(v_4_16, u_1_124, sum_1_16);
											sum_1_24 = mm_fmadd_ps(v_4_24, u_1_132, sum_1_24);
											//sum_1_32 = mm_fmadd_ps(v_4_32, u_1_140, sum_1_32);
											sum_1_32 = _mm_add_ps(_mm_mul_ps(v_4_32, u_1_140), sum_1_32);


											sum_2_0 = mm_fmadd_ps(v_1_0, u_2_0, sum_2_0);
											sum_2_8 = mm_fmadd_ps(v_1_8, u_2_8, sum_2_8);
											sum_2_16 = mm_fmadd_ps(v_1_16, u_2_16, sum_2_16);
											sum_2_24 = mm_fmadd_ps(v_1_24, u_2_24, sum_2_24);
											//sum_2_32 = mm_fmadd_ps(v_1_32, u_2_32, sum_2_32);
											sum_2_32 = _mm_add_ps(_mm_mul_ps(v_1_32, u_2_32), sum_2_32);
											sum_2_0 = mm_fmadd_ps(v_2_0, u_2_36, sum_2_0);
											sum_2_8 = mm_fmadd_ps(v_2_8, u_2_44, sum_2_8);
											sum_2_16 = mm_fmadd_ps(v_2_16, u_2_52, sum_2_16);
											sum_2_24 = mm_fmadd_ps(v_2_24, u_2_60, sum_2_24);
											//sum_2_32 = mm_fmadd_ps(v_2_32, u_2_68, sum_2_32);
											sum_2_32 = _mm_add_ps(_mm_mul_ps(v_2_32, u_2_68), sum_2_32);
											sum_2_0 = mm_fmadd_ps(v_3_0, u_2_72, sum_2_0);
											sum_2_8 = mm_fmadd_ps(v_3_8, u_2_80, sum_2_8);
											sum_2_16 = mm_fmadd_ps(v_3_16, u_2_88, sum_2_16);
											sum_2_24 = mm_fmadd_ps(v_3_24, u_2_96, sum_2_24);
											//sum_2_32 = mm_fmadd_ps(v_3_32, u_2_104, sum_2_32);
											sum_2_32 = _mm_add_ps(_mm_mul_ps(v_3_32, u_2_104), sum_2_32);
											sum_2_0 = mm_fmadd_ps(v_4_0, u_2_108, sum_2_0);
											sum_2_8 = mm_fmadd_ps(v_4_8, u_2_116, sum_2_8);
											sum_2_16 = mm_fmadd_ps(v_4_16, u_2_124, sum_2_16);
											sum_2_24 = mm_fmadd_ps(v_4_24, u_2_132, sum_2_24);
											//sum_2_32 = mm_fmadd_ps(v_4_32, u_2_140, sum_2_32);
											sum_2_32 = _mm_add_ps(_mm_mul_ps(v_4_32, u_2_140), sum_2_32);

											sum_3_0 = mm_fmadd_ps(v_1_0, u_3_0, sum_3_0);
											sum_3_8 = mm_fmadd_ps(v_1_8, u_3_8, sum_3_8);
											sum_3_16 = mm_fmadd_ps(v_1_16, u_3_16, sum_3_16);
											sum_3_24 = mm_fmadd_ps(v_1_24, u_3_24, sum_3_24);
											//sum_3_32 = mm_fmadd_ps(v_1_32, u_3_32, sum_3_32);
											sum_3_32 = _mm_add_ps(_mm_mul_ps(v_1_32, u_3_32), sum_3_32);
											sum_3_0 = mm_fmadd_ps(v_2_0, u_3_36, sum_3_0);
											sum_3_8 = mm_fmadd_ps(v_2_8, u_3_44, sum_3_8);
											sum_3_16 = mm_fmadd_ps(v_2_16, u_3_52, sum_3_16);
											sum_3_24 = mm_fmadd_ps(v_2_24, u_3_60, sum_3_24);
											//sum_3_32 = mm_fmadd_ps(v_2_32, u_3_68, sum_3_32);
											sum_3_32 = _mm_add_ps(_mm_mul_ps(v_2_32, u_3_68), sum_3_32);
											sum_3_0 = mm_fmadd_ps(v_3_0, u_3_72, sum_3_0);
											sum_3_8 = mm_fmadd_ps(v_3_8, u_3_80, sum_3_8);
											sum_3_16 = mm_fmadd_ps(v_3_16, u_3_88, sum_3_16);
											sum_3_24 = mm_fmadd_ps(v_3_24, u_3_96, sum_3_24);
											//sum_3_32 = mm_fmadd_ps(v_3_32, u_3_104, sum_3_32);
											sum_3_32 = _mm_add_ps(_mm_mul_ps(v_3_32, u_3_104), sum_3_32);
											sum_3_0 = mm_fmadd_ps(v_4_0, u_3_108, sum_3_0);
											sum_3_8 = mm_fmadd_ps(v_4_8, u_3_116, sum_3_8);
											sum_3_16 = mm_fmadd_ps(v_4_16, u_3_124, sum_3_16);
											sum_3_24 = mm_fmadd_ps(v_4_24, u_3_132, sum_3_24);
											//sum_3_32 = mm_fmadd_ps(v_4_32, u_3_140, sum_3_32);
											sum_3_32 = _mm_add_ps(_mm_mul_ps(v_4_32, u_3_140), sum_3_32);

											sum_4_0 = mm_fmadd_ps(v_1_0, u_4_0, sum_4_0);
											sum_4_8 = mm_fmadd_ps(v_1_8, u_4_8, sum_4_8);
											sum_4_16 = mm_fmadd_ps(v_1_16, u_4_16, sum_4_16);
											sum_4_24 = mm_fmadd_ps(v_1_24, u_4_24, sum_4_24);
											//sum_4_32 = mm_fmadd_ps(v_1_32, u_4_32, sum_4_32);
											sum_4_32 = _mm_add_ps(_mm_mul_ps(v_1_32, u_4_32), sum_4_32);
											sum_4_0 = mm_fmadd_ps(v_2_0, u_4_36, sum_4_0);
											sum_4_8 = mm_fmadd_ps(v_2_8, u_4_44, sum_4_8);
											sum_4_16 = mm_fmadd_ps(v_2_16, u_4_52, sum_4_16);
											sum_4_24 = mm_fmadd_ps(v_2_24, u_4_60, sum_4_24);
											//sum_4_32 = mm_fmadd_ps(v_2_32, u_4_68, sum_4_32);
											sum_4_32 = _mm_add_ps(_mm_mul_ps(v_2_32, u_4_68), sum_4_32);
											sum_4_0 = mm_fmadd_ps(v_3_0, u_4_72, sum_4_0);
											sum_4_8 = mm_fmadd_ps(v_3_8, u_4_80, sum_4_8);
											sum_4_16 = mm_fmadd_ps(v_3_16, u_4_88, sum_4_16);
											sum_4_24 = mm_fmadd_ps(v_3_24, u_4_96, sum_4_24);
											//sum_4_32 = mm_fmadd_ps(v_3_32, u_4_104, sum_4_32);
											sum_4_32 = _mm_add_ps(_mm_mul_ps(v_3_32, u_4_104), sum_4_32);
											sum_4_0 = mm_fmadd_ps(v_4_0, u_4_108, sum_4_0);
											sum_4_8 = mm_fmadd_ps(v_4_8, u_4_116, sum_4_8);
											sum_4_16 = mm_fmadd_ps(v_4_16, u_4_124, sum_4_16);
											sum_4_24 = mm_fmadd_ps(v_4_24, u_4_132, sum_4_24);
											//sum_4_32 = mm_fmadd_ps(v_4_32, u_4_140, sum_4_32);
											sum_4_32 = _mm_add_ps(_mm_mul_ps(v_4_32, u_4_140), sum_4_32);

#elif SIMD_TYPE >= SIMDTYPE_SSE
											u_1_0 = mm_load_ps(U_data + U_offset_och_ich_1);
											u_1_4 = mm_load_ps(U_data + U_offset_och_ich_1 + 4);
											u_1_8 = mm_load_ps(U_data + U_offset_och_ich_1 + 8);
											u_1_12 = mm_load_ps(U_data + U_offset_och_ich_1 + 12);
											u_1_16 = mm_load_ps(U_data + U_offset_och_ich_1 + 16);
											u_1_20 = mm_load_ps(U_data + U_offset_och_ich_1 + 20);
											u_1_24 = mm_load_ps(U_data + U_offset_och_ich_1 + 24);
											u_1_28 = mm_load_ps(U_data + U_offset_och_ich_1 + 28);
											u_1_32 = mm_load_ps(U_data + U_offset_och_ich_1 + 32);
											u_1_36 = mm_load_ps(U_data + U_offset_och_ich_1 + 36);
											u_1_40 = mm_load_ps(U_data + U_offset_och_ich_1 + 40);
											u_1_44 = mm_load_ps(U_data + U_offset_och_ich_1 + 44);
											u_1_48 = mm_load_ps(U_data + U_offset_och_ich_1 + 48);
											u_1_52 = mm_load_ps(U_data + U_offset_och_ich_1 + 52);
											u_1_56 = mm_load_ps(U_data + U_offset_och_ich_1 + 56);
											u_1_60 = mm_load_ps(U_data + U_offset_och_ich_1 + 60);

											u_2_0 = mm_load_ps(U_data + U_offset_och_ich_2);
											u_2_4 = mm_load_ps(U_data + U_offset_och_ich_2 + 4);
											u_2_8 = mm_load_ps(U_data + U_offset_och_ich_2 + 8);
											u_2_12 = mm_load_ps(U_data + U_offset_och_ich_2 + 12);
											u_2_16 = mm_load_ps(U_data + U_offset_och_ich_2 + 16);
											u_2_20 = mm_load_ps(U_data + U_offset_och_ich_2 + 20);
											u_2_24 = mm_load_ps(U_data + U_offset_och_ich_2 + 24);
											u_2_28 = mm_load_ps(U_data + U_offset_och_ich_2 + 28);
											u_2_32 = mm_load_ps(U_data + U_offset_och_ich_2 + 32);
											u_2_36 = mm_load_ps(U_data + U_offset_och_ich_2 + 36);
											u_2_40 = mm_load_ps(U_data + U_offset_och_ich_2 + 40);
											u_2_44 = mm_load_ps(U_data + U_offset_och_ich_2 + 44);
											u_2_48 = mm_load_ps(U_data + U_offset_och_ich_2 + 48);
											u_2_52 = mm_load_ps(U_data + U_offset_och_ich_2 + 52);
											u_2_56 = mm_load_ps(U_data + U_offset_och_ich_2 + 56);
											u_2_60 = mm_load_ps(U_data + U_offset_och_ich_2 + 60);

											u_3_0 = mm_load_ps(U_data + U_offset_och_ich_3);
											u_3_4 = mm_load_ps(U_data + U_offset_och_ich_3 + 4);
											u_3_8 = mm_load_ps(U_data + U_offset_och_ich_3 + 8);
											u_3_12 = mm_load_ps(U_data + U_offset_och_ich_3 + 12);
											u_3_16 = mm_load_ps(U_data + U_offset_och_ich_3 + 16);
											u_3_20 = mm_load_ps(U_data + U_offset_och_ich_3 + 20);
											u_3_24 = mm_load_ps(U_data + U_offset_och_ich_3 + 24);
											u_3_28 = mm_load_ps(U_data + U_offset_och_ich_3 + 28);
											u_3_32 = mm_load_ps(U_data + U_offset_och_ich_3 + 32);
											u_3_36 = mm_load_ps(U_data + U_offset_och_ich_3 + 36);
											u_3_40 = mm_load_ps(U_data + U_offset_och_ich_3 + 40);
											u_3_44 = mm_load_ps(U_data + U_offset_och_ich_3 + 44);
											u_3_48 = mm_load_ps(U_data + U_offset_och_ich_3 + 48);
											u_3_52 = mm_load_ps(U_data + U_offset_och_ich_3 + 52);
											u_3_56 = mm_load_ps(U_data + U_offset_och_ich_3 + 56);
											u_3_60 = mm_load_ps(U_data + U_offset_och_ich_3 + 60);

											u_4_0 = mm_load_ps(U_data + U_offset_och_ich_4);
											u_4_4 = mm_load_ps(U_data + U_offset_och_ich_4 + 4);
											u_4_8 = mm_load_ps(U_data + U_offset_och_ich_4 + 8);
											u_4_12 = mm_load_ps(U_data + U_offset_och_ich_4 + 12);
											u_4_16 = mm_load_ps(U_data + U_offset_och_ich_4 + 16);
											u_4_20 = mm_load_ps(U_data + U_offset_och_ich_4 + 20);
											u_4_24 = mm_load_ps(U_data + U_offset_och_ich_4 + 24);
											u_4_28 = mm_load_ps(U_data + U_offset_och_ich_4 + 28);
											u_4_32 = mm_load_ps(U_data + U_offset_och_ich_4 + 32);
											u_4_36 = mm_load_ps(U_data + U_offset_och_ich_4 + 36);
											u_4_40 = mm_load_ps(U_data + U_offset_och_ich_4 + 40);
											u_4_44 = mm_load_ps(U_data + U_offset_och_ich_4 + 44);
											u_4_48 = mm_load_ps(U_data + U_offset_och_ich_4 + 48);
											u_4_52 = mm_load_ps(U_data + U_offset_och_ich_4 + 52);
											u_4_56 = mm_load_ps(U_data + U_offset_och_ich_4 + 56);
											u_4_60 = mm_load_ps(U_data + U_offset_och_ich_4 + 60);

											v_1_0 = mm_load_ps(V_data + V_offset_ich_row_col_1);
											v_1_4 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 4);
											v_1_8 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 8);
											v_1_12 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 12);
											v_2_0 = mm_load_ps(V_data + V_offset_ich_row_col_2);
											v_2_4 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 4);
											v_2_8 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 8);
											v_2_12 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 12);
											v_3_0 = mm_load_ps(V_data + V_offset_ich_row_col_3);
											v_3_4 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 4);
											v_3_8 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 8);
											v_3_12 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 12);
											v_4_0 = mm_load_ps(V_data + V_offset_ich_row_col_4);
											v_4_4 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 4);
											v_4_8 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 8);
											v_4_12 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 12);

											sum_1_0 = mm_fmadd_ps(v_1_0, u_1_0, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_1_4, u_1_4, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_1_8, u_1_8, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_1_12, u_1_12, sum_1_12);
											sum_1_0 = mm_fmadd_ps(v_2_0, u_1_16, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_2_4, u_1_20, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_2_8, u_1_24, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_2_12, u_1_28, sum_1_12);
											sum_1_0 = mm_fmadd_ps(v_3_0, u_1_32, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_3_4, u_1_36, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_3_8, u_1_40, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_3_12, u_1_44, sum_1_12);
											sum_1_0 = mm_fmadd_ps(v_4_0, u_1_48, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_4_4, u_1_52, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_4_8, u_1_56, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_4_12, u_1_60, sum_1_12);

											sum_2_0 = mm_fmadd_ps(v_1_0, u_2_0, sum_2_0);
											sum_2_4 = mm_fmadd_ps(v_1_4, u_2_4, sum_2_4);
											sum_2_8 = mm_fmadd_ps(v_1_8, u_2_8, sum_2_8);
											sum_2_12 = mm_fmadd_ps(v_1_12, u_2_12, sum_2_12);
											sum_2_0 = mm_fmadd_ps(v_2_0, u_2_16, sum_2_0);
											sum_2_4 = mm_fmadd_ps(v_2_4, u_2_20, sum_2_4);
											sum_2_8 = mm_fmadd_ps(v_2_8, u_2_24, sum_2_8);
											sum_2_12 = mm_fmadd_ps(v_2_12, u_2_28, sum_2_12);
											sum_2_0 = mm_fmadd_ps(v_3_0, u_2_32, sum_2_0);
											sum_2_4 = mm_fmadd_ps(v_3_4, u_2_36, sum_2_4);
											sum_2_8 = mm_fmadd_ps(v_3_8, u_2_40, sum_2_8);
											sum_2_12 = mm_fmadd_ps(v_3_12, u_2_44, sum_2_12);
											sum_2_0 = mm_fmadd_ps(v_4_0, u_2_48, sum_2_0);
											sum_2_4 = mm_fmadd_ps(v_4_4, u_2_52, sum_2_4);
											sum_2_8 = mm_fmadd_ps(v_4_8, u_2_56, sum_2_8);
											sum_2_12 = mm_fmadd_ps(v_4_12, u_2_60, sum_2_12);

											sum_3_0 = mm_fmadd_ps(v_1_0, u_3_0, sum_3_0);
											sum_3_4 = mm_fmadd_ps(v_1_4, u_3_4, sum_3_4);
											sum_3_8 = mm_fmadd_ps(v_1_8, u_3_8, sum_3_8);
											sum_3_12 = mm_fmadd_ps(v_1_12, u_3_12, sum_3_12);
											sum_3_0 = mm_fmadd_ps(v_2_0, u_3_16, sum_3_0);
											sum_3_4 = mm_fmadd_ps(v_2_4, u_3_20, sum_3_4);
											sum_3_8 = mm_fmadd_ps(v_2_8, u_3_24, sum_3_8);
											sum_3_12 = mm_fmadd_ps(v_2_12, u_3_28, sum_3_12);
											sum_3_0 = mm_fmadd_ps(v_3_0, u_3_32, sum_3_0);
											sum_3_4 = mm_fmadd_ps(v_3_4, u_3_36, sum_3_4);
											sum_3_8 = mm_fmadd_ps(v_3_8, u_3_40, sum_3_8);
											sum_3_12 = mm_fmadd_ps(v_3_12, u_3_44, sum_3_12);
											sum_3_0 = mm_fmadd_ps(v_4_0, u_3_48, sum_3_0);
											sum_3_4 = mm_fmadd_ps(v_4_4, u_3_52, sum_3_4);
											sum_3_8 = mm_fmadd_ps(v_4_8, u_3_56, sum_3_8);
											sum_3_12 = mm_fmadd_ps(v_4_12, u_3_60, sum_3_12);

											sum_4_0 = mm_fmadd_ps(v_1_0, u_4_0, sum_4_0);
											sum_4_4 = mm_fmadd_ps(v_1_4, u_4_4, sum_4_4);
											sum_4_8 = mm_fmadd_ps(v_1_8, u_4_8, sum_4_8);
											sum_4_12 = mm_fmadd_ps(v_1_12, u_4_12, sum_4_12);
											sum_4_0 = mm_fmadd_ps(v_2_0, u_4_16, sum_4_0);
											sum_4_4 = mm_fmadd_ps(v_2_4, u_4_20, sum_4_4);
											sum_4_8 = mm_fmadd_ps(v_2_8, u_4_24, sum_4_8);
											sum_4_12 = mm_fmadd_ps(v_2_12, u_4_28, sum_4_12);
											sum_4_0 = mm_fmadd_ps(v_3_0, u_4_32, sum_4_0);
											sum_4_4 = mm_fmadd_ps(v_3_4, u_4_36, sum_4_4);
											sum_4_8 = mm_fmadd_ps(v_3_8, u_4_40, sum_4_8);
											sum_4_12 = mm_fmadd_ps(v_3_12, u_4_44, sum_4_12);
											sum_4_0 = mm_fmadd_ps(v_4_0, u_4_48, sum_4_0);
											sum_4_4 = mm_fmadd_ps(v_4_4, u_4_52, sum_4_4);
											sum_4_8 = mm_fmadd_ps(v_4_8, u_4_56, sum_4_8);
											sum_4_12 = mm_fmadd_ps(v_4_12, u_4_60, sum_4_12);
#else
											for (int i = 0; i < tile_length_; i++)
											{
												mult_data1[i] += U_data[U_offset_och_ich_1 + i] * V_data[V_offset_ich_row_col_1 + i];
												mult_data1[i] += U_data[U_offset_och_ich_1 + tile_length_ + i] * V_data[V_offset_ich_row_col_2 + i];
												mult_data1[i] += U_data[U_offset_och_ich_1 + 2 * tile_length_ + i] * V_data[V_offset_ich_row_col_3 + i];
												mult_data1[i] += U_data[U_offset_och_ich_1 + 3 * tile_length_ + i] * V_data[V_offset_ich_row_col_4 + i];

												mult_data2[i] += U_data[U_offset_och_ich_2 + i] * V_data[V_offset_ich_row_col_1 + i];
												mult_data2[i] += U_data[U_offset_och_ich_2 + tile_length_ + i] * V_data[V_offset_ich_row_col_2 + i];
												mult_data2[i] += U_data[U_offset_och_ich_2 + 2 * tile_length_ + i] * V_data[V_offset_ich_row_col_3 + i];
												mult_data2[i] += U_data[U_offset_och_ich_2 + 3 * tile_length_ + i] * V_data[V_offset_ich_row_col_4 + i];

												mult_data3[i] += U_data[U_offset_och_ich_3 + i] * V_data[V_offset_ich_row_col_1 + i];
												mult_data3[i] += U_data[U_offset_och_ich_3 + tile_length_ + i] * V_data[V_offset_ich_row_col_2 + i];
												mult_data3[i] += U_data[U_offset_och_ich_3 + 2 * tile_length_ + i] * V_data[V_offset_ich_row_col_3 + i];
												mult_data3[i] += U_data[U_offset_och_ich_3 + 3 * tile_length_ + i] * V_data[V_offset_ich_row_col_4 + i];

												mult_data4[i] += U_data[U_offset_och_ich_4 + i] * V_data[V_offset_ich_row_col_1 + i];
												mult_data4[i] += U_data[U_offset_och_ich_4 + tile_length_ + i] * V_data[V_offset_ich_row_col_2 + i];
												mult_data4[i] += U_data[U_offset_och_ich_4 + 2 * tile_length_ + i] * V_data[V_offset_ich_row_col_3 + i];
												mult_data4[i] += U_data[U_offset_och_ich_4 + 3 * tile_length_ + i] * V_data[V_offset_ich_row_col_4 + i];
											}
#endif
										}

										for (; ich < input_Channel_; ich++)
										{
											int offset = ich * tile_length_;
											int U_offset_och_ich_1 = U_offset_och_1 + offset;
											int U_offset_och_ich_2 = U_offset_och_2 + offset;
											int U_offset_och_ich_3 = U_offset_och_3 + offset;
											int U_offset_och_ich_4 = U_offset_och_4 + offset;

											int V_offset_ich_row_col_1 = V_offset_row_col + ich * h_w_tile_stride;

#if SIMD_TYPE >= SIMDTYPE_AVX
											//u_1
											u_1_0 = mm_load_ps(U_data + U_offset_och_ich_1);
											u_1_8 = mm_load_ps(U_data + U_offset_och_ich_1 + 8);
											u_1_16 = mm_load_ps(U_data + U_offset_och_ich_1 + 16);
											u_1_24 = mm_load_ps(U_data + U_offset_och_ich_1 + 24);
											u_1_32 = _mm_loadu_ps(U_data + U_offset_och_ich_1 + 32);

											//u_2
											u_2_0 = mm_load_ps(U_data + U_offset_och_ich_2);
											u_2_8 = mm_load_ps(U_data + U_offset_och_ich_2 + 8);
											u_2_16 = mm_load_ps(U_data + U_offset_och_ich_2 + 16);
											u_2_24 = mm_load_ps(U_data + U_offset_och_ich_2 + 24);
											u_2_32 = _mm_loadu_ps(U_data + U_offset_och_ich_2 + 32);

											//u_3
											u_3_0 = mm_load_ps(U_data + U_offset_och_ich_3);
											u_3_8 = mm_load_ps(U_data + U_offset_och_ich_3 + 8);
											u_3_16 = mm_load_ps(U_data + U_offset_och_ich_3 + 16);
											u_3_24 = mm_load_ps(U_data + U_offset_och_ich_3 + 24);
											u_3_32 = _mm_loadu_ps(U_data + U_offset_och_ich_3 + 32);

											//u_4
											u_4_0 = mm_load_ps(U_data + U_offset_och_ich_4);
											u_4_8 = mm_load_ps(U_data + U_offset_och_ich_4 + 8);
											u_4_16 = mm_load_ps(U_data + U_offset_och_ich_4 + 16);
											u_4_24 = mm_load_ps(U_data + U_offset_och_ich_4 + 24);
											u_4_32 = _mm_loadu_ps(U_data + U_offset_och_ich_4 + 32);

											//v
											v_1_0 = mm_load_ps(V_data + V_offset_ich_row_col_1);
											v_1_8 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 8);
											v_1_16 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 16);
											v_1_24 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 24);
											v_1_32 = _mm_loadu_ps(V_data + V_offset_ich_row_col_1 + 32);

											sum_1_0 = mm_fmadd_ps(v_1_0, u_1_0, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_1_8, u_1_8, sum_1_8);
											sum_1_16 = mm_fmadd_ps(v_1_16, u_1_16, sum_1_16);
											sum_1_24 = mm_fmadd_ps(v_1_24, u_1_24, sum_1_24);
											//sum_1_32 = mm_fmadd_ps(v_1_32, u_1_32, sum_1_32);
											sum_1_32 = _mm_add_ps(_mm_mul_ps(v_1_32, u_1_32), sum_1_32);

											sum_2_0 = mm_fmadd_ps(v_1_0, u_2_0, sum_2_0);
											sum_2_8 = mm_fmadd_ps(v_1_8, u_2_8, sum_2_8);
											sum_2_16 = mm_fmadd_ps(v_1_16, u_2_16, sum_2_16);
											sum_2_24 = mm_fmadd_ps(v_1_24, u_2_24, sum_2_24);
											//sum_2_32 = mm_fmadd_ps(v_1_32, u_2_32, sum_2_32);
											sum_2_32 = _mm_add_ps(_mm_mul_ps(v_1_32, u_2_32), sum_2_32);

											sum_3_0 = mm_fmadd_ps(v_1_0, u_3_0, sum_3_0);
											sum_3_8 = mm_fmadd_ps(v_1_8, u_3_8, sum_3_8);
											sum_3_16 = mm_fmadd_ps(v_1_16, u_3_16, sum_3_16);
											sum_3_24 = mm_fmadd_ps(v_1_24, u_3_24, sum_3_24);
											//sum_3_32 = mm_fmadd_ps(v_1_32, u_3_32, sum_3_32);
											sum_3_32 = _mm_add_ps(_mm_mul_ps(v_1_32, u_3_32), sum_3_32);

											sum_4_0 = mm_fmadd_ps(v_1_0, u_4_0, sum_4_0);
											sum_4_8 = mm_fmadd_ps(v_1_8, u_4_8, sum_4_8);
											sum_4_16 = mm_fmadd_ps(v_1_16, u_4_16, sum_4_16);
											sum_4_24 = mm_fmadd_ps(v_1_24, u_4_24, sum_4_24);
											//sum_4_32 = mm_fmadd_ps(v_1_32, u_4_32, sum_4_32);
											sum_4_32 = _mm_add_ps(_mm_mul_ps(v_1_32, u_4_32), sum_4_32);
#elif SIMD_TYPE >= SIMDTYPE_SSE
											u_1_0 = mm_load_ps(U_data + U_offset_och_ich_1);
											u_1_4 = mm_load_ps(U_data + U_offset_och_ich_1 + 4);
											u_1_8 = mm_load_ps(U_data + U_offset_och_ich_1 + 8);
											u_1_12 = mm_load_ps(U_data + U_offset_och_ich_1 + 12);

											u_2_0 = mm_load_ps(U_data + U_offset_och_ich_2);
											u_2_4 = mm_load_ps(U_data + U_offset_och_ich_2 + 4);
											u_2_8 = mm_load_ps(U_data + U_offset_och_ich_2 + 8);
											u_2_12 = mm_load_ps(U_data + U_offset_och_ich_2 + 12);

											u_3_0 = mm_load_ps(U_data + U_offset_och_ich_3);
											u_3_4 = mm_load_ps(U_data + U_offset_och_ich_3 + 4);
											u_3_8 = mm_load_ps(U_data + U_offset_och_ich_3 + 8);
											u_3_12 = mm_load_ps(U_data + U_offset_och_ich_3 + 12);

											u_4_0 = mm_load_ps(U_data + U_offset_och_ich_4);
											u_4_4 = mm_load_ps(U_data + U_offset_och_ich_4 + 4);
											u_4_8 = mm_load_ps(U_data + U_offset_och_ich_4 + 8);
											u_4_12 = mm_load_ps(U_data + U_offset_och_ich_4 + 12);

											v_1_0 = mm_load_ps(V_data + V_offset_ich_row_col_1);
											v_1_4 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 4);
											v_1_8 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 8);
											v_1_12 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 12);

											sum_1_0 = mm_fmadd_ps(v_1_0, u_1_0, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_1_4, u_1_4, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_1_8, u_1_8, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_1_12, u_1_12, sum_1_12);

											sum_2_0 = mm_fmadd_ps(v_1_0, u_2_0, sum_2_0);
											sum_2_4 = mm_fmadd_ps(v_1_4, u_2_4, sum_2_4);
											sum_2_8 = mm_fmadd_ps(v_1_8, u_2_8, sum_2_8);
											sum_2_12 = mm_fmadd_ps(v_1_12, u_2_12, sum_2_12);

											sum_3_0 = mm_fmadd_ps(v_1_0, u_3_0, sum_3_0);
											sum_3_4 = mm_fmadd_ps(v_1_4, u_3_4, sum_3_4);
											sum_3_8 = mm_fmadd_ps(v_1_8, u_3_8, sum_3_8);
											sum_3_12 = mm_fmadd_ps(v_1_12, u_3_12, sum_3_12);

											sum_4_0 = mm_fmadd_ps(v_1_0, u_4_0, sum_4_0);
											sum_4_4 = mm_fmadd_ps(v_1_4, u_4_4, sum_4_4);
											sum_4_8 = mm_fmadd_ps(v_1_8, u_4_8, sum_4_8);
											sum_4_12 = mm_fmadd_ps(v_1_12, u_4_12, sum_4_12);
#else
											for (int i = 0; i < tile_length_; i++)
											{
												mult_data1[i] += U_data[U_offset_och_ich_1 + i] * V_data[V_offset_ich_row_col_1 + i];
												mult_data2[i] += U_data[U_offset_och_ich_2 + i] * V_data[V_offset_ich_row_col_1 + i];
												mult_data3[i] += U_data[U_offset_och_ich_3 + i] * V_data[V_offset_ich_row_col_1 + i];
												mult_data4[i] += U_data[U_offset_och_ich_4 + i] * V_data[V_offset_ich_row_col_1 + i];
											}
#endif
										}

#if SIMD_TYPE >= SIMDTYPE_AVX
										mm_store_ps(mult_data1, sum_1_0);
										mm_store_ps(mult_data1 + 8, sum_1_8);
										mm_store_ps(mult_data1 + 16, sum_1_16);
										mm_store_ps(mult_data1 + 24, sum_1_24);
										_mm_storeu_ps(mult_data1 + 32, sum_1_32);
										mm_store_ps(mult_data2, sum_2_0);
										mm_store_ps(mult_data2 + 8, sum_2_8);
										mm_store_ps(mult_data2 + 16, sum_2_16);
										mm_store_ps(mult_data2 + 24, sum_2_24);
										_mm_storeu_ps(mult_data2 + 32, sum_2_32);
										mm_store_ps(mult_data3, sum_3_0);
										mm_store_ps(mult_data3 + 8, sum_3_8);
										mm_store_ps(mult_data3 + 16, sum_3_16);
										mm_store_ps(mult_data3 + 24, sum_3_24);
										_mm_storeu_ps(mult_data3 + 32, sum_3_32);
										mm_store_ps(mult_data4, sum_4_0);
										mm_store_ps(mult_data4 + 8, sum_4_8);
										mm_store_ps(mult_data4 + 16, sum_4_16);
										mm_store_ps(mult_data4 + 24, sum_4_24);
										_mm_storeu_ps(mult_data4 + 32, sum_4_32);
#elif SIMD_TYPE >= SIMDTYPE_SSE
										mm_store_ps(mult_data1, sum_1_0);
										mm_store_ps(mult_data1 + 4, sum_1_4);
										mm_store_ps(mult_data1 + 8, sum_1_8);
										mm_store_ps(mult_data1 + 12, sum_1_12);
										mm_store_ps(mult_data2, sum_2_0);
										mm_store_ps(mult_data2 + 4, sum_2_4);
										mm_store_ps(mult_data2 + 8, sum_2_8);
										mm_store_ps(mult_data2 + 12, sum_2_12);
										mm_store_ps(mult_data3, sum_3_0);
										mm_store_ps(mult_data3 + 4, sum_3_4);
										mm_store_ps(mult_data3 + 8, sum_3_8);
										mm_store_ps(mult_data3 + 12, sum_3_12);
										mm_store_ps(mult_data4, sum_4_0);
										mm_store_ps(mult_data4 + 4, sum_4_4);
										mm_store_ps(mult_data4 + 8, sum_4_8);
										mm_store_ps(mult_data4 + 12, sum_4_12);
#endif

										calculate_ATmA43(mult_data1, result1);
										calculate_ATmA43(mult_data2, result2);
										calculate_ATmA43(mult_data3, result3);
										calculate_ATmA43(mult_data4, result4);

										int row_in_output_data = i / w_tile_num_ * m_;
										int col_in_output_data = i % w_tile_num_* m_;
										int top_offset_row_col = row_in_output_data * output_dim_w_ + col_in_output_data;
										int top_offset_num_och_row_col_1 = top_offset_num_och_1 + top_offset_row_col;
										int top_offset_num_och_row_col_2 = top_offset_num_och_2 + top_offset_row_col;
										int top_offset_num_och_row_col_3 = top_offset_num_och_3 + top_offset_row_col;
										int top_offset_num_och_row_col_4 = top_offset_num_och_4 + top_offset_row_col;

										for (int row = 0; row < m_; row++)
										{
											int result_offset_row = row * m_;
											for (int col = 0; col < m_; col++)
											{
												top_data[top_offset_num_och_row_col_1 + col] = result1[result_offset_row + col] + bias1;
												top_data[top_offset_num_och_row_col_2 + col] = result2[result_offset_row + col] + bias2;
												top_data[top_offset_num_och_row_col_3 + col] = result3[result_offset_row + col] + bias3;
												top_data[top_offset_num_och_row_col_4 + col] = result4[result_offset_row + col] + bias4;
											}
											top_offset_num_och_row_col_1 += output_dim_w_;
											top_offset_num_och_row_col_2 += output_dim_w_;
											top_offset_num_och_row_col_3 += output_dim_w_;
											top_offset_num_och_row_col_4 += output_dim_w_;
										}
									}
								}

								for (int och = remain_outch_start; och < output_Channel_; och++)
								{
#if SIMD_TYPE >= SIMDTYPE_AVX
									mm_type sum_1_0;
									mm_type sum_1_8;
									mm_type sum_1_16;
									mm_type sum_1_24;
									__m128 sum_1_32;

									mm_type u_1_0;
									mm_type u_1_8;
									mm_type u_1_16;
									mm_type u_1_24;
									__m128 u_1_32;
									mm_type u_1_36;
									mm_type u_1_44;
									mm_type u_1_52;
									mm_type u_1_60;
									__m128 u_1_68;
									mm_type u_1_72;
									mm_type u_1_80;
									mm_type u_1_88;
									mm_type u_1_96;
									__m128 u_1_104;
									mm_type u_1_108;
									mm_type u_1_116;
									mm_type u_1_124;
									mm_type u_1_132;
									__m128 u_1_140;

									mm_type v_1_0;
									mm_type v_1_8;
									mm_type v_1_16;
									mm_type v_1_24;
									__m128 v_1_32;
									mm_type v_2_0;
									mm_type v_2_8;
									mm_type v_2_16;
									mm_type v_2_24;
									__m128 v_2_32;
									mm_type v_3_0;
									mm_type v_3_8;
									mm_type v_3_16;
									mm_type v_3_24;
									__m128 v_3_32;
									mm_type v_4_0;
									mm_type v_4_8;
									mm_type v_4_16;
									mm_type v_4_24;
									__m128 v_4_32;
#elif SIMD_TYPE >= SIMDTYPE_SSE
									mm_type sum_1_0;
									mm_type sum_1_4;
									mm_type sum_1_8;
									mm_type sum_1_12;
									mm_type u_1_0;
									mm_type u_1_4;
									mm_type u_1_8;
									mm_type u_1_12;
									mm_type u_1_16;
									mm_type u_1_20;
									mm_type u_1_24;
									mm_type u_1_28;
									mm_type u_1_32;
									mm_type u_1_36;
									mm_type u_1_40;
									mm_type u_1_44;
									mm_type u_1_48;
									mm_type u_1_52;
									mm_type u_1_56;
									mm_type u_1_60;
									mm_type v_1_0;
									mm_type v_1_4;
									mm_type v_1_8;
									mm_type v_1_12;
									mm_type v_2_0;
									mm_type v_2_4;
									mm_type v_2_8;
									mm_type v_2_12;
									mm_type v_3_0;
									mm_type v_3_4;
									mm_type v_3_8;
									mm_type v_3_12;
									mm_type v_4_0;
									mm_type v_4_4;
									mm_type v_4_8;
									mm_type v_4_12;
#endif

									int U_offset_och_1 = och * U_offset_single_och;

									float bias1 = bias_data[och];
									float result1[16];
									int top_offset_num_och_1 = top_offset_num + och * output_spatial_dim_;

									for (int i = 0; i < total_tile_num; i++)
									{
										float mult_data1[36] = { 0 };
										int V_offset_row_col = i * tile_length_;

#if SIMD_TYPE >= SIMDTYPE_AVX
										sum_1_0 = mm_setzero_ps();
										sum_1_8 = mm_setzero_ps();
										sum_1_16 = mm_setzero_ps();
										sum_1_24 = mm_setzero_ps();
										sum_1_32 = _mm_setzero_ps();
#elif SIMD_TYPE >= SIMDTYPE_SSE
										sum_1_0 = mm_setzero_ps();
										sum_1_4 = mm_setzero_ps();
										sum_1_8 = mm_setzero_ps();
										sum_1_12 = mm_setzero_ps();
#endif // SIMD_TYPE >= SIMDTYPE_AVX
										int ich = 0;
										for (; ich + 3 < input_Channel_; ich += 4)
										{
											int U_offset_och_ich_1 = U_offset_och_1 + ich * tile_length_;

											int V_offset_ich_row_col_1 = V_offset_row_col + ich * h_w_tile_stride;
											int V_offset_ich_row_col_2 = V_offset_ich_row_col_1 + h_w_tile_stride;
											int V_offset_ich_row_col_3 = V_offset_ich_row_col_2 + h_w_tile_stride;
											int V_offset_ich_row_col_4 = V_offset_ich_row_col_3 + h_w_tile_stride;

#if SIMD_TYPE >= SIMDTYPE_AVX
											u_1_0 = mm_load_ps(U_data + U_offset_och_ich_1);
											u_1_8 = mm_load_ps(U_data + U_offset_och_ich_1 + 8);
											u_1_16 = mm_load_ps(U_data + U_offset_och_ich_1 + 16);
											u_1_24 = mm_load_ps(U_data + U_offset_och_ich_1 + 24);
											u_1_32 = _mm_loadu_ps(U_data + U_offset_och_ich_1 + 32);
											u_1_36 = mm_load_ps(U_data + U_offset_och_ich_1 + 36);
											u_1_44 = mm_load_ps(U_data + U_offset_och_ich_1 + 44);
											u_1_52 = mm_load_ps(U_data + U_offset_och_ich_1 + 52);
											u_1_60 = mm_load_ps(U_data + U_offset_och_ich_1 + 60);
											u_1_68 = _mm_loadu_ps(U_data + U_offset_och_ich_1 + 68);
											u_1_72 = mm_load_ps(U_data + U_offset_och_ich_1 + 72);
											u_1_80 = mm_load_ps(U_data + U_offset_och_ich_1 + 80);
											u_1_88 = mm_load_ps(U_data + U_offset_och_ich_1 + 88);
											u_1_96 = mm_load_ps(U_data + U_offset_och_ich_1 + 96);
											u_1_104 = _mm_loadu_ps(U_data + U_offset_och_ich_1 + 104);
											u_1_108 = mm_load_ps(U_data + U_offset_och_ich_1 + 108);
											u_1_116 = mm_load_ps(U_data + U_offset_och_ich_1 + 116);
											u_1_124 = mm_load_ps(U_data + U_offset_och_ich_1 + 124);
											u_1_132 = mm_load_ps(U_data + U_offset_och_ich_1 + 132);
											u_1_140 = _mm_loadu_ps(U_data + U_offset_och_ich_1 + 140);

											v_1_0 = mm_load_ps(V_data + V_offset_ich_row_col_1);
											v_1_8 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 8);
											v_1_16 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 16);
											v_1_24 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 24);
											v_1_32 = _mm_loadu_ps(V_data + V_offset_ich_row_col_1 + 32);
											v_2_0 = mm_load_ps(V_data + V_offset_ich_row_col_2);
											v_2_8 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 8);
											v_2_16 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 16);
											v_2_24 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 24);
											v_2_32 = _mm_loadu_ps(V_data + V_offset_ich_row_col_2 + 32);
											v_3_0 = mm_load_ps(V_data + V_offset_ich_row_col_3);
											v_3_8 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 8);
											v_3_16 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 16);
											v_3_24 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 24);
											v_3_32 = _mm_loadu_ps(V_data + V_offset_ich_row_col_3 + 32);
											v_4_0 = mm_load_ps(V_data + V_offset_ich_row_col_4);
											v_4_8 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 8);
											v_4_16 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 16);
											v_4_24 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 24);
											v_4_32 = _mm_loadu_ps(V_data + V_offset_ich_row_col_4 + 32);

											sum_1_0 = mm_fmadd_ps(v_1_0, u_1_0, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_1_8, u_1_8, sum_1_8);
											sum_1_16 = mm_fmadd_ps(v_1_16, u_1_16, sum_1_16);
											sum_1_24 = mm_fmadd_ps(v_1_24, u_1_24, sum_1_24);
											//sum_1_32 = mm_fmadd_ps(v_1_32, u_1_32, sum_1_32);
											sum_1_32 = _mm_add_ps(_mm_mul_ps(v_1_32, u_1_32), sum_1_32);

											sum_1_0 = mm_fmadd_ps(v_2_0, u_1_36, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_2_8, u_1_44, sum_1_8);
											sum_1_16 = mm_fmadd_ps(v_2_16, u_1_52, sum_1_16);
											sum_1_24 = mm_fmadd_ps(v_2_24, u_1_60, sum_1_24);
											//sum_1_32 = mm_fmadd_ps(v_2_32, u_1_68, sum_1_32);
											sum_1_32 = _mm_add_ps(_mm_mul_ps(v_2_32, u_1_68), sum_1_32);

											sum_1_0 = mm_fmadd_ps(v_3_0, u_1_72, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_3_8, u_1_80, sum_1_8);
											sum_1_16 = mm_fmadd_ps(v_3_16, u_1_88, sum_1_16);
											sum_1_24 = mm_fmadd_ps(v_3_24, u_1_96, sum_1_24);
											//sum_1_32 = mm_fmadd_ps(v_3_32, u_1_104, sum_1_32);
											sum_1_32 = _mm_add_ps(_mm_mul_ps(v_3_32, u_1_104), sum_1_32);

											sum_1_0 = mm_fmadd_ps(v_4_0, u_1_108, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_4_8, u_1_116, sum_1_8);
											sum_1_16 = mm_fmadd_ps(v_4_16, u_1_124, sum_1_16);
											sum_1_24 = mm_fmadd_ps(v_4_24, u_1_132, sum_1_24);
											//sum_1_32 = mm_fmadd_ps(v_4_32, u_1_140, sum_1_32);
											sum_1_32 = _mm_add_ps(_mm_mul_ps(v_4_32, u_1_140), sum_1_32);
#elif SIMD_TYPE >= SIMDTYPE_SSE
											u_1_0 = mm_load_ps(U_data + U_offset_och_ich_1);
											u_1_4 = mm_load_ps(U_data + U_offset_och_ich_1 + 4);
											u_1_8 = mm_load_ps(U_data + U_offset_och_ich_1 + 8);
											u_1_12 = mm_load_ps(U_data + U_offset_och_ich_1 + 12);
											u_1_16 = mm_load_ps(U_data + U_offset_och_ich_1 + 16);
											u_1_20 = mm_load_ps(U_data + U_offset_och_ich_1 + 20);
											u_1_24 = mm_load_ps(U_data + U_offset_och_ich_1 + 24);
											u_1_28 = mm_load_ps(U_data + U_offset_och_ich_1 + 28);
											u_1_32 = mm_load_ps(U_data + U_offset_och_ich_1 + 32);
											u_1_36 = mm_load_ps(U_data + U_offset_och_ich_1 + 36);
											u_1_40 = mm_load_ps(U_data + U_offset_och_ich_1 + 40);
											u_1_44 = mm_load_ps(U_data + U_offset_och_ich_1 + 44);
											u_1_48 = mm_load_ps(U_data + U_offset_och_ich_1 + 48);
											u_1_52 = mm_load_ps(U_data + U_offset_och_ich_1 + 52);
											u_1_56 = mm_load_ps(U_data + U_offset_och_ich_1 + 56);
											u_1_60 = mm_load_ps(U_data + U_offset_och_ich_1 + 60);

											v_1_0 = mm_load_ps(V_data + V_offset_ich_row_col_1);
											v_1_4 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 4);
											v_1_8 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 8);
											v_1_12 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 12);
											v_2_0 = mm_load_ps(V_data + V_offset_ich_row_col_2);
											v_2_4 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 4);
											v_2_8 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 8);
											v_2_12 = mm_load_ps(V_data + V_offset_ich_row_col_2 + 12);
											v_3_0 = mm_load_ps(V_data + V_offset_ich_row_col_3);
											v_3_4 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 4);
											v_3_8 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 8);
											v_3_12 = mm_load_ps(V_data + V_offset_ich_row_col_3 + 12);
											v_4_0 = mm_load_ps(V_data + V_offset_ich_row_col_4);
											v_4_4 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 4);
											v_4_8 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 8);
											v_4_12 = mm_load_ps(V_data + V_offset_ich_row_col_4 + 12);

											sum_1_0 = mm_fmadd_ps(v_1_0, u_1_0, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_1_4, u_1_4, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_1_8, u_1_8, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_1_12, u_1_12, sum_1_12);
											sum_1_0 = mm_fmadd_ps(v_2_0, u_1_16, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_2_4, u_1_20, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_2_8, u_1_24, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_2_12, u_1_28, sum_1_12);
											sum_1_0 = mm_fmadd_ps(v_3_0, u_1_32, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_3_4, u_1_36, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_3_8, u_1_40, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_3_12, u_1_44, sum_1_12);
											sum_1_0 = mm_fmadd_ps(v_4_0, u_1_48, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_4_4, u_1_52, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_4_8, u_1_56, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_4_12, u_1_60, sum_1_12);
#else
											for (int i = 0; i < tile_length_; i++)
											{
												mult_data1[i] += U_data[U_offset_och_ich_1 + i] * V_data[V_offset_ich_row_col_1 + i];
												mult_data1[i] += U_data[U_offset_och_ich_1 + tile_length_ + i] * V_data[V_offset_ich_row_col_2 + i];
												mult_data1[i] += U_data[U_offset_och_ich_1 + 2 * tile_length_ + i] * V_data[V_offset_ich_row_col_3 + i];
												mult_data1[i] += U_data[U_offset_och_ich_1 + 3 * tile_length_ + i] * V_data[V_offset_ich_row_col_4 + i];
											}
#endif
										}

										for (; ich < input_Channel_; ich++)
										{
											int U_offset_och_ich_1 = U_offset_och_1 + ich * tile_length_;
											int V_offset_ich_row_col_1 = V_offset_row_col + ich * h_w_tile_stride;

#if SIMD_TYPE >= SIMDTYPE_AVX
											u_1_0 = mm_load_ps(U_data + U_offset_och_ich_1);
											u_1_8 = mm_load_ps(U_data + U_offset_och_ich_1 + 8);
											u_1_16 = mm_load_ps(U_data + U_offset_och_ich_1 + 16);
											u_1_24 = mm_load_ps(U_data + U_offset_och_ich_1 + 24);
											u_1_32 = _mm_loadu_ps(U_data + U_offset_och_ich_1 + 32);


											v_1_0 = mm_load_ps(V_data + V_offset_ich_row_col_1);
											v_1_8 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 8);
											v_1_16 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 16);
											v_1_24 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 24);
											v_1_32 = _mm_loadu_ps(V_data + V_offset_ich_row_col_1 + 32);


											sum_1_0 = mm_fmadd_ps(v_1_0, u_1_0, sum_1_0);
											sum_1_8 = mm_fmadd_ps(v_1_8, u_1_8, sum_1_8);
											sum_1_16 = mm_fmadd_ps(v_1_16, u_1_16, sum_1_16);
											sum_1_24 = mm_fmadd_ps(v_1_24, u_1_24, sum_1_24);
											//sum_1_32 = mm_fmadd_ps(v_1_32, u_1_32, sum_1_32);
											sum_1_32 = _mm_add_ps(_mm_mul_ps(v_1_32, u_1_32), sum_1_32);
#elif SIMD_TYPE >= SIMDTYPE_SSE
											u_1_0 = mm_load_ps(U_data + U_offset_och_ich_1);
											u_1_4 = mm_load_ps(U_data + U_offset_och_ich_1 + 4);
											u_1_8 = mm_load_ps(U_data + U_offset_och_ich_1 + 8);
											u_1_12 = mm_load_ps(U_data + U_offset_och_ich_1 + 12);

											v_1_0 = mm_load_ps(V_data + V_offset_ich_row_col_1);
											v_1_4 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 4);
											v_1_8 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 8);
											v_1_12 = mm_load_ps(V_data + V_offset_ich_row_col_1 + 12);

											sum_1_0 = mm_fmadd_ps(v_1_0, u_1_0, sum_1_0);
											sum_1_4 = mm_fmadd_ps(v_1_4, u_1_4, sum_1_4);
											sum_1_8 = mm_fmadd_ps(v_1_8, u_1_8, sum_1_8);
											sum_1_12 = mm_fmadd_ps(v_1_12, u_1_12, sum_1_12);
#else
											for (int i = 0; i < tile_length_; i++)
											{
												mult_data1[i] += U_data[U_offset_och_ich_1 + i] * V_data[V_offset_ich_row_col_1 + i];
											}
#endif
										}


#if SIMD_TYPE >= SIMDTYPE_AVX
										mm_store_ps(mult_data1, sum_1_0);
										mm_store_ps(mult_data1 + 8, sum_1_8);
										mm_store_ps(mult_data1 + 16, sum_1_16);
										mm_store_ps(mult_data1 + 24, sum_1_24);
										_mm_storeu_ps(mult_data1 + 32, sum_1_32);
#elif SIME_TYPE >= SIMDTYPE_SSE
										mm_store_ps(Mult_data + Mult_offset_och_row_col_1, sum_1_0);
										mm_store_ps(Mult_data + Mult_offset_och_row_col_1 + 4, sum_1_4);
										mm_store_ps(Mult_data + Mult_offset_och_row_col_1 + 8, sum_1_8);
										mm_store_ps(Mult_data + Mult_offset_och_row_col_1 + 12, sum_1_12);
#endif // SIMD_TYPE >= SIMDTYPE_AVX

										calculate_ATmA43(mult_data1, result1);

										int row_in_output_data = i / w_tile_num_ * m_;
										int col_in_output_data = i % w_tile_num_* m_;
										int top_offset_row_col = row_in_output_data * output_dim_w_ + col_in_output_data;
										int top_offset_num_och_row_col_1 = top_offset_num_och_1 + top_offset_row_col;

										for (int row = 0; row < m_; row++)
										{
											int result_offset_row = row * m_;
											for (int col = 0; col < m_; col++)
											{
												top_data[top_offset_num_och_row_col_1 + col] = result1[result_offset_row + col] + bias1;
											}
											top_offset_num_och_row_col_1 += output_dim_w_;
										}
									}
								}
							}
						}

						delete V_data;
						if ((add_h != 0) || (add_w != 0))
						{
							tensor_operation_cpu::cut_border_cpu(top, top, 0, add_h, 0, add_w);
						}
					}
					else
					{
						LOG(FATAL) << "group wrong!!!";
					}
				}
				else if (order_ == NHWC)
				{
					NOT_IMPLEMENTED;
				}
				else
				{
					NOT_IMPLEMENTED;
				}
			}
		}
	}
}