#include "../../include/Excalibur/operation_reflector.hpp"
#include "../../include/Excalibur/operation_convolutiondepthwise.hpp"
#include "./operation_make_border.hpp"
#include <algorithm>
#include "../../include/Primitives/profiler.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_convolutiondepthwise<Dtype>::operation_convolutiondepthwise(const operation_param& param) : operation_convolution<Dtype>(param)
		{

		}

		template<typename Dtype>
		int operation_convolutiondepthwise<Dtype>::init_weights(FILE *fp)
		{
			int mem = operation_convolution::init_weights(fp);
			if (/*kernel_size_h_ == 3 && kernel_size_w_ == 3*/false)
			{
				//F(2,3), calculate U= GgG^T
				//float G[] = { 1.f, 0.f, 0.f, 0.5f, 0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.f, 0.f, 1.f };
				U_.reset(new memory::tensor<float>(std::vector<int>{1, output_channel_, 4, 4}, -1, memory::NCHW, nullptr));
				V_.reset(new memory::tensor<float>(std::vector<int>{1, output_channel_, 4, 4}, -1, memory::NCHW, nullptr));
				float* U_data = U_->mutable_cpu_data();
				const float* weights_data = weights_f32_[0]->cpu_data();
				for (size_t g = 0; g < output_channel_; g++)
				{
					const float* w = weights_data + 9 * g;
					float* u = U_data + 16 * g;
					u[0] = w[0];
					u[1] = (w[0] + w[1] + w[2]) / 2;
					u[2] = (w[0] - w[1] + w[2]) / 2;
					u[3] = w[2];
					u[4] = (w[0] + w[3] + w[6]) / 2;
					u[5] = (w[0] + w[1] + w[2] + w[3] + w[4] + w[5] + w[6] + w[7] + w[8]) / 4;
					u[6] = (w[0] - w[1] + w[2] + w[3] - w[4] + w[5] + w[6] - w[7] + w[8]) / 4;
					u[7] = (w[2] + w[5] + w[8]) / 2;
					u[8] = (w[0] - w[3] + w[6]) / 2;
					u[9] = (w[0] + w[1] + w[2] - w[3] - w[4] - w[5] + w[6] + w[7] + w[8]) / 4;
					u[10] = (w[0] - w[1] + w[2] - w[3] + w[4] - w[5] + w[6] - w[7] + w[8]) / 4;
					u[11] = (w[2] - w[5] + w[8]) / 2;
					u[12] = w[6];
					u[13] = (w[6] + w[7] + w[8]) / 2;
					u[14] = (w[6] - w[7] + w[8]) / 2;
					u[15] = w[8];
				}
			}
			return mem;
		}

		template<typename Dtype>
		void operation_convolutiondepthwise<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);
			CHECK_EQ(output_channel_, group_);
			num_ = bottoms[0]->num();
			input_dim_h_ = bottoms[0]->height();
			input_dim_w_ = bottoms[0]->width();
			input_channel_ = bottoms[0]->channels();
			CHECK_EQ(input_channel_, output_channel_);
			output_dim_h_ = (input_dim_h_ + pad_bottom_ + pad_top_ - kernel_size_h_) / stride_h_ + 1;
			output_dim_w_ = (input_dim_w_ + pad_left_ + pad_right_ - kernel_size_w_) / stride_w_ + 1;
			if ((kernel_size_h_ == 3 && kernel_size_w_ == 3) && (stride_h_ == 1 && stride_w_ == 1))
			{
				forward_k3s1_f32(bottoms[0], tops[0]);
			}
			else if ((kernel_size_h_ == 3 && kernel_size_w_ == 3) && (stride_h_ == 2 && stride_w_ == 2))
			{
				forward_k3s2_f32(bottoms[0], tops[0]);
			}
			else
			{
				const float* bottom_data = bottoms[0]->cpu_data();
				const float* weights_data = weights_f32_[0]->cpu_data();
				const float* bias_data = nullptr;
				float* top_data = nullptr;
				if (bias_term_)
				{
					bias_data = weights_f32_[1]->cpu_data();
				}
				switch (bottoms[0]->order())
				{
				case memory::NCHW:
					tops[0].reset(new memory::tensor<float>(std::vector<int>{num_, output_channel_, output_dim_h_, output_dim_w_},
						bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
					top_data = tops[0]->mutable_cpu_data();
					for (size_t n = 0; n < num_; n++)
					{
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
						for (int i = 0; i < tops[0]->count(); i++)
						{
							const int pw = i % output_dim_w_;
							const int ph = (i / output_dim_w_) % output_dim_h_;
							const int c = (i / output_dim_w_ / output_dim_h_) % output_channel_;
							const int n_step = i / output_dim_w_ / output_dim_h_ / output_channel_;
							int hstart = ph * stride_h_ - pad_top_;
							int wstart = pw * stride_w_ - pad_left_;
							int hend = std::min(hstart + kernel_size_h_, input_dim_h_ + pad_bottom_);
							int wend = std::min(wstart + kernel_size_w_, input_dim_w_ + pad_right_);
							hstart = std::max(hstart, 0);
							wstart = std::max(wstart, 0);
							hend = std::min(hend, input_dim_h_);
							wend = std::min(wend, input_dim_w_);
							float aveval = 0;
							const float* bottom_slice =
								bottom_data + n * bottoms[0]->count(1, 4) + (n_step * output_channel_ + c) * input_dim_h_ * input_dim_w_;
							const float* weight_slice =
								weights_data + c * kernel_size_h_ * kernel_size_w_;
							int khstart = hend < kernel_size_h_ ? kernel_size_h_ - hend : 0;
							int kwstart = wend < kernel_size_w_ ? kernel_size_w_ - wend : 0;
							for (int h = hstart; h < hend; ++h) 
							{
								for (int w = wstart; w < wend; ++w) 
								{
									aveval += bottom_slice[h * input_dim_h_ + w] * weight_slice[(khstart + h - hstart) * kernel_size_w_ + (kwstart + w - wstart)];
								}
							}
							if (bias_term_) 
							{
								aveval += bias_data[c];
							}
							top_data[n * bottoms[0]->count(1, 4) + i] = aveval;
						}
					}
					
					break;
				case memory::NHWC:
					tops[0].reset(new memory::tensor<float>(std::vector<int>{num_, output_dim_h_, output_dim_w_, output_channel_},
						bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
					NOT_IMPLEMENTED;
					break;
				default:
					NOT_IMPLEMENTED;
					break;
				}
			}

		}

		template<typename Dtype>
		void operation_convolutiondepthwise<Dtype>::forward_gpu_f32(
#ifdef USE_CUDA
			cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
			const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			NOT_IMPLEMENTED;
		}

		template<typename Dtype>
		void operation_convolutiondepthwise<Dtype>::forward_winograd_f32(std::shared_ptr <memory::tensor<float>>& bottom, 
			std::shared_ptr < memory::tensor<float>>& top)
		{
			if (bottom->order() != memory::NCHW)
			{
				bottom->convert_order();
			}
			top.reset(new memory::tensor<float>(std::vector<int>{num_, output_channel_, output_dim_h_, output_dim_w_},
				bottom->device(), bottom->order(), bottom->allocator()));
			std::shared_ptr<memory::tensor<float>> border_bottom;
			int wino_pad_h = (input_dim_h_ + pad_bottom_ + pad_top_) % 2;
			int wino_pad_w = (input_dim_w_ + pad_left_ + pad_right_) % 2;
			make_border<float>(bottom, border_bottom, pad_top_, pad_bottom_ + wino_pad_h, pad_left_, pad_right_ + wino_pad_w, border_constant, pad_value_);
			const int tile_height = (border_bottom->height() - 4) / 2 + 1;
			const int tile_width = (border_bottom->width() - 4) / 2 + 1;
			const int b_width = border_bottom->width();
			std::shared_ptr<memory::tensor<float>> border_top;
			border_top.reset(new memory::tensor<float>(std::vector<int>{num_, output_channel_, tile_height * 2, tile_width * 2},
				top->device(), memory::NCHW, top->allocator()));
			const int t_width = border_top->width();
			const float* bottom_data = border_bottom->cpu_data();
			//const float* weights_data = weights_f32_[0]->cpu_data();
			const float* bias_data = nullptr;
			if (bias_term_)
			{
				bias_data = weights_f32_[1]->cpu_data();
			}
			const float* U = U_->cpu_data();
			float* V = V_->mutable_cpu_data();
			for (size_t n = 0; n < num_; n++)
			{
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (int g = 0; g < group_; g++)
				{
					const float* bottom_data_slice = bottom_data + g * border_bottom->count(2, 4);
					float* top_data_slice = border_top->mutable_cpu_data() + g * border_top->count(2, 4);
					const float* u = U + 16 * g;
					float* v = V + 16 * g;
					float d[16];
					for (size_t tile_h = 0; tile_h < tile_height; tile_h++)
					{
						for (size_t tile_w = 0; tile_w < tile_width; tile_w++)
						{
							//get d
							for (size_t i = 0; i < 16; i++)
							{
								d[i] = bottom_data_slice[(tile_h * 2 + i / 4) * b_width + tile_w * 2 + i % 4];
							}
							//F(2,3), caclulate B^TdB
							v[0] = d[0];
							v[1] = d[1] - d[2] - d[3];
							v[2] = -d[0] + d[1] + d[2];
							v[3] = -d[3];
							v[4] = d[4] - d[8] + d[12];
							v[5] = d[5] - d[6] + d[7] - d[9] + d[10] - d[11] + d[13] - d[14] + d[15];
							v[6] = -d[4] + d[5] + d[6] + d[8] - d[9] - d[10] - d[12] + d[13] + d[14];
							v[7] = -d[7] + d[11] - d[15];
							v[8] = -d[0] + d[4] + d[8];
							v[9] = -d[1] + d[2] - d[3] + d[5] - d[6] + d[7] + d[9] - d[10] + d[11];
							v[10] = d[0] - d[1] - d[2] - d[4] + d[5] + d[6] - d[8] + d[9] + d[10];
							v[11] = d[3] - d[7] - d[11];
							v[12] = -d[12];
							v[13] = -d[13] + d[14] - d[15];
							v[14] = d[12] - d[13] - d[14];
							v[15] = d[15];
							//calculate M = U.*V
							for (size_t i = 0; i < 16 / mm_align_size; i++)
							{
								__m256 vx = _mm256_load_ps(v + i * mm_align_size);
								__m256 ux = _mm256_load_ps(u + i * mm_align_size);
								vx = _mm256_mul_ps(vx, ux);
								_mm256_store_ps(v + i * mm_align_size, vx);
							}
							//calculate A^TMA
							top_data_slice[(tile_h * 2) * t_width + tile_w * 2] = v[0] + v[1] + v[2] + v[4] + v[5] + v[6] + v[8] + v[9] + v[10] + bias_term_ ? bias_data[g] : 0.0f;
							top_data_slice[(tile_h * 2) * t_width + tile_w * 2 + 1] = v[1] - v[2] + v[3] + v[5] - v[6] + v[7] + v[9] - v[10] + v[11] + bias_term_ ? bias_data[g] : 0.0f;
							top_data_slice[(tile_h * 2 + 1) * t_width + tile_w * 2] = v[4] + v[5] + v[6] - v[8] - v[9] - v[10] + v[12] + v[13] + v[14] + bias_term_ ? bias_data[g] : 0.0f;
							top_data_slice[(tile_h * 2 + 1) * t_width + tile_w * 2 + 1] = v[5] - v[6] + v[7] - v[9] + v[10] - v[11] + v[13] - v[14] + v[15] + bias_term_ ? bias_data[g] : 0.0f;
						}
					}
				}
			}
			//cut pad border
			for (size_t h = 0; h < output_dim_h_; h++)
			{
				memcpy(top->mutable_cpu_data() + h * output_dim_w_, border_top->cpu_data() + h * (output_dim_w_ + wino_pad_w), output_dim_w_ * sizeof(float));
			}
		}


		template<typename Dtype>
		void operation_convolutiondepthwise<Dtype>::forward_k3s1_f32(const std::shared_ptr < memory::tensor<float>>& bottom,
			std::shared_ptr < memory::tensor<float>>& top)
		{
			/*profiler *p = profiler::get();
			p->scope_start("make_border");*/
			std::shared_ptr<memory::tensor<float>> bottom_bordered;
			// time cost 2ms in make_border!
			make_border<float>(bottom, bottom_bordered, pad_top_, pad_bottom_, pad_left_, pad_right_, border_constant, pad_value_);
			if (bottom_bordered->order() != memory::NCHW)
			{
				bottom_bordered->convert_order();
			}
			/*p->scope_end();
			p->scope_start("reset");*/
			top.reset(new memory::tensor<float>(std::vector<int>{1, output_channel_, output_dim_h_, output_dim_w_},
				bottom->device(), bottom->order(), bottom->allocator()));
			float* top_data = top->mutable_cpu_data();
			const int top_cstep = top->count(2, 4);
			const int bottom_cstep = bottom_bordered->count(2, 4);
			auto bottom_data = bottom_bordered->cpu_data();
			const int real_width = bottom_bordered->width();
			const float* weights_data = weights_f32_[0]->cpu_data();
			const float* bias_data = nullptr;
			if (bias_term_)
			{
				bias_data = weights_f32_[1]->cpu_data();
			}
			/*p->scope_end();
			p->scope_start("exec");*/
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
			for (int g = 0; g < group_; g++)
			{
				float *out = top_data + g * top_cstep;
				const float bias_data0 = bias_term_ ? bias_data[g] : 0.f;
				const float* weights_data0 = weights_data + g * 9;
				float* outptr = out;
				float* outptr2 = outptr + output_dim_w_;

				const float* img0 = bottom_data + g * bottom_cstep;

				const float* r0 = img0;
				const float* r1 = img0 + real_width;
				const float* r2 = img0 + real_width * 2;
				const float* r3 = img0 + real_width * 3;

				const float* k0 = weights_data0;
				const float* k1 = weights_data0 + 3;
				const float* k2 = weights_data0 + 6;
#if SIMD_TYPE >= SIMDTYPE_SSE
				__m128 k0_data = _mm_loadu_ps(k0);
				__m128 k1_data = _mm_loadu_ps(k1);
				__m128 k2_data = _mm_loadu_ps(k2);
				int i = 0;

				for (; i + 1 < output_dim_h_; i += 2)
				{

					int remain = output_dim_w_;

					for (; remain > 0; remain--)
					{
						__m128 r0_data = _mm_loadu_ps(r0);
						__m128 r1_data = _mm_loadu_ps(r1);
						__m128 r2_data = _mm_loadu_ps(r2);
						__m128 r3_data = _mm_loadu_ps(r3);

						*outptr = mul_add_3x3_simd(r0_data, r1_data, r2_data, k0_data, k1_data, k2_data, bias_data0);
						*outptr2 = mul_add_3x3_simd(r1_data, r2_data, r3_data, k0_data, k1_data, k2_data, bias_data0);

						r0++;
						r1++;
						r2++;
						r3++;
						outptr++;
						outptr2++;
					}

					r0 += 2 + real_width;
					r1 += 2 + real_width;
					r2 += 2 + real_width;
					r3 += 2 + real_width;

					outptr += output_dim_w_;
					outptr2 += output_dim_w_;
				}

				for (; i < output_dim_h_; i++)
				{
					int remain = output_dim_w_;

					for (; remain > 0; remain--)
					{
						__m128 r0_data = _mm_loadu_ps(r0);
						__m128 r1_data = _mm_loadu_ps(r1);
						__m128 r2_data = _mm_loadu_ps(r2);

						*outptr = mul_add_3x3_simd(r0_data, r1_data, r2_data, k0_data, k1_data, k2_data, bias_data0);
						r0++;
						r1++;
						r2++;
						outptr++;
					}

					r0 += 2;
					r1 += 2;
					r2 += 2;
				}
#else
				int i = 0;

				for (; i + 1 < output_dim_h_; i += 2)
				{

					int remain = output_dim_w_;

					for (; remain > 0; remain--)
					{
						*outptr = mul_add_3x3_native(r0, r1, r2, k0, k1, k2, bias_data0);
						*outptr2 = mul_add_3x3_native(r1, r2, r3, k0, k1, k2, bias_data0);

						r0++;
						r1++;
						r2++;
						r3++;
						outptr++;
						outptr2++;
					}

					r0 += 2 + input_dim_w_;
					r1 += 2 + input_dim_w_;
					r2 += 2 + input_dim_w_;
					r3 += 2 + input_dim_w_;

					outptr += output_dim_w_;
					outptr2 += output_dim_w_;
				}

				for (; i < output_dim_h_; i++)
				{
					int remain = output_dim_w_;

					for (; remain > 0; remain--)
					{
						*outptr = mul_add_3x3_native(r0, r1, r2, k0, k1, k2, bias_data0);

						r0++;
						r1++;
						r2++;
						outptr++;
					}

					r0 += 2;
					r1 += 2;
					r2 += 2;
				}
#endif
			}

			//p->scope_end();
		}

		template<typename Dtype>
		void operation_convolutiondepthwise<Dtype>::forward_k3s2_f32(const std::shared_ptr < memory::tensor<float>>& bottom,
			std::shared_ptr < memory::tensor<float>>& top)
		{
			profiler *p = profiler::get();
			p->scope_start("make_border");
			std::shared_ptr<memory::tensor<float>> bottom_bordered;
			make_border<float>(bottom, bottom_bordered, pad_top_, pad_bottom_, pad_left_, pad_right_, border_constant, pad_value_);
			if (bottom_bordered->order() != memory::NCHW)
			{
				bottom_bordered->convert_order();
			}
			p->scope_end();
			p->scope_start("reset");
			top.reset(new memory::tensor<float>(std::vector<int>{1, output_channel_, output_dim_h_, output_dim_w_},
				bottom->device(), bottom->order(), bottom->allocator()));
			float* top_data = top->mutable_cpu_data();
			const int top_cstep = top->count(2, 4);
			const int bottom_cstep = bottom_bordered->count(2, 4);
			auto bottom_data = bottom_bordered->cpu_data();
			const float* weights_data = weights_f32_[0]->cpu_data();
			const float* bias_data = nullptr;
			if (bias_term_)
			{
				bias_data = weights_f32_[1]->cpu_data();
			}
			const int tailstep = bottom_bordered->width() - 2 * output_dim_w_ + bottom_bordered->width();
			p->scope_end();
			p->scope_start("exec");
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
			for (int g = 0; g < group_; g++)
			{
				float *out = top_data + g * top_cstep;

				const float bias0 = bias_term_ ? bias_data[g] : 0.f;

				const float* kernel0 = weights_data + g * 9;

				float* outptr = out;

				const float* img0 = bottom_data + g * bottom_cstep;

				const float* r0 = img0;
				const float* r1 = img0 + bottom_bordered->width();
				const float* r2 = img0 + bottom_bordered->width() * 2;

				const float* k0 = kernel0;
				const float* k1 = kernel0 + 3;
				const float* k2 = kernel0 + 6;

#if SIMD_TYPE >= SIMDTYPE_SSE
				__m128 k0_data = _mm_loadu_ps(k0);
				__m128 k1_data = _mm_loadu_ps(k1);
				__m128 k2_data = _mm_loadu_ps(k2);

				int i = 0;

				for (; i < output_dim_h_; i++)
				{
					int remain = output_dim_w_;

					for (; remain > 0; remain--)
					{
						__m128 r0_data = _mm_loadu_ps(r0);
						__m128 r1_data = _mm_loadu_ps(r1);
						__m128 r2_data = _mm_loadu_ps(r2);

						*outptr = mul_add_3x3_simd(r0_data, r1_data, r2_data, k0_data, k1_data, k2_data, bias0);

						r0 += 2;
						r1 += 2;
						r2 += 2;
						outptr++;
					}

					r0 += tailstep;
					r1 += tailstep;
					r2 += tailstep;
				}

#else
				int i = 0;

				for (; i < outh; i++)
				{
					int remain = outw;

					for (; remain > 0; remain--)
					{
						*outptr = mul_add_3x3_native(r0, r1, r2, k0, k1, k2, bias0);

						r0 += 2;
						r1 += 2;
						r2 += 2;
						outptr++;
					}

					r0 += tailstep;
					r1 += tailstep;
					r2 += tailstep;
				}
#endif

			}

			p->scope_end();
		}

		INSTANCE_CLASS(operation_convolutiondepthwise);
		REGISTE(operation_convolutiondepthwise);
	}
}