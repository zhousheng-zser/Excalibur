#include "../../../include/Excalibur/arm/operation_innerproduct_arm.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"
#include <random>

#include "arm/gemm_symm_int8.h"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_innerproduct_arm<Dtype>::operation_innerproduct_arm(const operation_param & param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					num_output_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "1")
				{
					bias_term_ = (bool)atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "2")
				{
					weight_data_size_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "8")
				{
					int8_scale_term_ = (bool)atoi(split_string(attrs[i], "=")[1].c_str());
					this->params_.set_int8_quantization(int8_scale_term_);
				}
				else if (split_string(attrs[i], "=")[0] == "9")
				{
					activation_type_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "10")
				{
					NOT_IMPLEMENTED;
				}
				else if (split_string(attrs[i], "=")[0] == "-23330")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported InnerProduct Attribution " << split_string(attrs[i], "=")[0];
				}
			}
		}

		template<typename Dtype>
		int operation_innerproduct_arm<Dtype>::init_weights()
		{
			std::default_random_engine e;
			std::normal_distribution<float> n(0, 0.3);
			std::uniform_int_distribution<int> u(-128, 127);
			int mem = 0;
			if (!this->params_.int8_quantization_)
			{
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(weight_data_size_, this->params_.device_, memory::NCHW, nullptr)));
				for (size_t i = 0; i < weight_data_size_; i++)
				{
					this->weights_f32_[0]->mutable_cpu_data()[i] = n(e);
				}
				mem += weight_data_size_ * sizeof(float);
				if (this->bias_term_)
				{
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(num_output_, this->params_.device_, memory::NCHW, nullptr)));
					for (size_t i = 0; i < num_output_; i++)
					{
						this->weights_f32_[1]->mutable_cpu_data()[i] = n(e);
					}
					mem += num_output_ * sizeof(float);
				}
			}
			else
			{
				size_t align_data_size = (weight_data_size_ + 4 - 1) & -4;
				this->weights_i8_.push_back(std::shared_ptr<memory::tensor<signed char>>(new memory::tensor<signed char>(align_data_size, this->params_.device_, memory::NCHW, nullptr)));
				for (size_t i = 0; i < align_data_size; i++)
				{
					this->weights_i8_[0]->mutable_cpu_data()[i] = u(e);
				}
				mem += align_data_size;
				if (this->bias_term_)
				{
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(1, this->params_.device_, memory::NCHW, nullptr)));
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(num_output_, this->params_.device_, memory::NCHW, nullptr)));
					for (size_t i = 0; i < num_output_; i++)
					{
						this->weights_f32_[1]->mutable_cpu_data()[i] = n(e);
					}
					mem += num_output_ * sizeof(float);
				}
				this->weights_scaletable_i8_.resize(1);
				this->weights_scaletable_i8_[0] = n(e);
				this->featmap_scaletable_i8_.resize(1);
				this->featmap_scaletable_i8_[0] = n(e);
				mem += 2 * sizeof(float);

				if (std::fabs(this->weights_scaletable_i8_[0]) <= 1e-6)
					scale_in = 0.f;
				else
					scale_in = 1.f / (4 * this->weights_scaletable_i8_[0] * this->featmap_scaletable_i8_[0]);

#ifdef __aarch64__
				weight_i8_reordered_.reset(new memory::tensor<int8_t>(this->weights_i8_[0]->count(), this->params_.device_, memory::NCHW, nullptr));
				reorder_a(const_cast<int8_t*>(this->weights_i8_[0]->cpu_data()), weight_i8_reordered_->mutable_cpu_data(), num_output_, weight_data_size_ / num_output_, weight_data_size_ / num_output_);
#endif

			}
			return mem;
		}

		template<typename Dtype>
		int operation_innerproduct_arm<Dtype>::init_weights(FILE *fp)
		{
			int quantize_tag;
			fread(&quantize_tag, 1, sizeof(int), fp);
			int mem = 0;
			if (quantize_tag == 0)
			{
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(weight_data_size_, this->params_.device_, memory::NCHW, nullptr)));
				fread(this->weights_f32_[0]->mutable_cpu_data(), 1, weight_data_size_ * sizeof(float), fp);
				mem += weight_data_size_ * sizeof(float);
				if (this->bias_term_)
				{
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(num_output_, this->params_.device_, memory::NCHW, nullptr)));
					fread(this->weights_f32_[1]->mutable_cpu_data(), 1, num_output_ * sizeof(float), fp);
					mem += num_output_ * sizeof(float);
				}
				return mem;
			}
			else if (quantize_tag == 871224)
			{
				size_t align_data_size = (weight_data_size_ + 4 - 1) & -4;
				this->weights_i8_.push_back(std::shared_ptr<memory::tensor<signed char>>(new memory::tensor<signed char>(align_data_size, this->params_.device_, memory::NCHW, nullptr)));
				fread(this->weights_i8_[0]->mutable_cpu_data(), 1, align_data_size, fp);
				mem += align_data_size;
				if (this->bias_term_)
				{
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(1, this->params_.device_, memory::NCHW, nullptr)));
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(num_output_, this->params_.device_, memory::NCHW, nullptr)));
					fread(this->weights_f32_[1]->mutable_cpu_data(), 1, num_output_ * sizeof(float), fp);
					mem += num_output_ * sizeof(float);
				}
				this->weights_scaletable_i8_.resize(1);
				fread(this->weights_scaletable_i8_.data(), 1, 1 * sizeof(float), fp);
				this->featmap_scaletable_i8_.resize(1);
				fread(this->featmap_scaletable_i8_.data(), 1, 1 * sizeof(float), fp);
				mem += 2 * sizeof(float);

				if (std::fabs(this->weights_scaletable_i8_[0]) <= 1e-6)
					scale_in = 0.f;
				else
					scale_in = 1.f / (4 * this->weights_scaletable_i8_[0] * this->featmap_scaletable_i8_[0]);

#ifdef __aarch64__
				weight_i8_reordered_.reset(new memory::tensor<int8_t>(this->weights_i8_[0]->count(), this->params_.device_, memory::NCHW, nullptr));
				reorder_a(const_cast<int8_t*>(this->weights_i8_[0]->cpu_data()), weight_i8_reordered_->mutable_cpu_data(), num_output_, weight_data_size_ / num_output_, weight_data_size_ / num_output_);
#endif

				return mem;
			}
			else
			{
				NOT_IMPLEMENTED;
				return 0;
			}
		}

		template<typename Dtype>
		void operation_innerproduct_arm<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms, std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);

			memory::orderType order = bottoms[0]->order();
			if (order == memory::NHWC)
				bottoms[0]->convert_order();

			int num = bottoms[0]->num();
			int w = bottoms[0]->width();
			int h = bottoms[0]->height();
			int channels = bottoms[0]->channels();
			int size = w * h;

			tops[0].reset(new memory::tensor<float>(std::vector<int>{num, num_output_, 1, 1}, -1, memory::NCHW));

			const float* weight_data_ptr = this->weights_f32_[0]->cpu_data();
			const float *bias_data = nullptr;
			if (bias_term_)
			{
				bias_data = this->weights_f32_[1]->cpu_data();
			}

			for (int num_i = 0; num_i < num; num_i++)
			{
				const float *bottom_data = bottoms[0]->cpu_data() + num_i * channels * size;
				float *top_data = tops[0]->mutable_cpu_data() + num_i * num_output_;

				int nn_num_output = num_output_ >> 2;
				int remain_num_output_start = nn_num_output << 2;
#pragma omp parallel for num_threads(2) 
				for (int pp = 0; pp < nn_num_output; pp++)
				{
					int p = pp * 4;

					float sum0 = 0.f;
					float sum1 = 0.f;
					float sum2 = 0.f;
					float sum3 = 0.f;

					if (bias_term_)
					{
						sum0 = bias_data[p];
						sum1 = bias_data[p + 1];
						sum2 = bias_data[p + 2];
						sum3 = bias_data[p + 3];
					}

					const float* w0 = weight_data_ptr + size * channels * p;
					const float* w1 = weight_data_ptr + size * channels * (p + 1);
					const float* w2 = weight_data_ptr + size * channels * (p + 2);
					const float* w3 = weight_data_ptr + size * channels * (p + 3);

#if __ARM_NEON
					float32x4_t _sum0 = vdupq_n_f32(0.f);
					float32x4_t _sum1 = vdupq_n_f32(0.f);
					float32x4_t _sum2 = vdupq_n_f32(0.f);
					float32x4_t _sum3 = vdupq_n_f32(0.f);
#endif // __ARM_NEON

					// channels
					for (int q = 0; q < channels; q++)
					{
						const float* m = bottom_data + (q)* size;

#if __ARM_NEON
						int nn = size >> 2;
						int remain = size & 3;
#else
						int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
						for (; nn > 0; nn--)
						{
							float32x4_t _m = vld1q_f32(m);

							float32x4_t _w0 = vld1q_f32(w0);
							_sum0 = vmlaq_f32(_sum0, _m, _w0);

							float32x4_t _w1 = vld1q_f32(w1);
							_sum1 = vmlaq_f32(_sum1, _m, _w1);

							float32x4_t _w2 = vld1q_f32(w2);
							_sum2 = vmlaq_f32(_sum2, _m, _w2);

							float32x4_t _w3 = vld1q_f32(w3);
							_sum3 = vmlaq_f32(_sum3, _m, _w3);

							m += 4;
							w0 += 4;
							w1 += 4;
							w2 += 4;
							w3 += 4;
						}
#endif // __ARM_NEON
						for (; remain > 0; remain--)
						{
							sum0 += *m * *w0;
							sum1 += *m * *w1;
							sum2 += *m * *w2;
							sum3 += *m * *w3;

							m++;
							w0++;
							w1++;
							w2++;
							w3++;
						}

					}

#if __ARM_NEON
					float32x2_t _sum0ss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
					float32x2_t _sum1ss = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
					float32x2_t _sum2ss = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
					float32x2_t _sum3ss = vadd_f32(vget_low_f32(_sum3), vget_high_f32(_sum3));

					float32x2_t _sum01ss = vpadd_f32(_sum0ss, _sum1ss);
					float32x2_t _sum23ss = vpadd_f32(_sum2ss, _sum3ss);

					sum0 += vget_lane_f32(_sum01ss, 0);
					sum1 += vget_lane_f32(_sum01ss, 1);
					sum2 += vget_lane_f32(_sum23ss, 0);
					sum3 += vget_lane_f32(_sum23ss, 1);

#endif // __ARM_NEON

					top_data[p] = sum0;
					top_data[p + 1] = sum1;
					top_data[p + 2] = sum2;
					top_data[p + 3] = sum3;
				}

				// num_output
#pragma omp parallel for num_threads(2) 
				for (int p = remain_num_output_start; p < num_output_; p++)
				{
					float sum = 0.f;

					if (bias_term_)
						sum = bias_data[p];

					const float* w = weight_data_ptr + size * channels * p;

#if __ARM_NEON
					float32x4_t _sum = vdupq_n_f32(0.f);
					float32x4_t _sum2 = vdupq_n_f32(0.f);
#endif // __ARM_NEON

					// channels
					for (int q = 0; q < channels; q++)
					{
						const float* m = bottom_data + (q)* size;

#if __ARM_NEON
						int nn = size >> 3;
						int remain = size & 7;
#else
						int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
						if (nn > 0)
						{
							asm volatile(
								"0:                                   \n"
								"prfm       pldl1keep, [%1, #256]     \n"
								"ld1        {v0.4s, v1.4s}, [%1], #32 \n"
								"prfm       pldl1keep, [%2, #256]     \n"
								"ld1        {v2.4s, v3.4s}, [%2], #32 \n"
								"fmla       %3.4s, v0.4s, v2.4s       \n"
								"subs       %w0, %w0, #1              \n"
								"fmla       %4.4s, v1.4s, v3.4s       \n"
								"bne        0b                        \n"
								: "=r"(nn),     // %0
								"=r"(m),      // %1
								"=r"(w),      // %2
								"=w"(_sum),   // %3
								"=w"(_sum2)   // %4
								: "0"(nn),
								"1"(m),
								"2"(w),
								"3"(_sum),
								"4"(_sum2)
								: "cc", "memory", "v0", "v1", "v2", "v3"
								);
						}
#else
						if (nn > 0)
						{
							asm volatile(
								"0:                             \n"
								"pld        [%1, #256]          \n"
								"vld1.f32   {d0-d3}, [%1 :128]! \n"
								"pld        [%2, #256]          \n"
								"vld1.f32   {d4-d7}, [%2]!      \n"
								"vmla.f32   %q3, q0, q2         \n"
								"subs       %0, #1              \n"
								"vmla.f32   %q4, q1, q3         \n"
								"bne        0b                  \n"
								: "=r"(nn),     // %0
								"=r"(m),      // %1
								"=r"(w),      // %2
								"=w"(_sum),   // %3
								"=w"(_sum2)   // %4
								: "0"(nn),
								"1"(m),
								"2"(w),
								"3"(_sum),
								"4"(_sum2)
								: "cc", "memory", "q0", "q1", "q2", "q3"
								);
						}
#endif // __aarch64__
#endif // __ARM_NEON
						for (; remain > 0; remain--)
						{
							sum += *m * *w;

							m++;
							w++;
						}
					}

#if __ARM_NEON
					_sum = vaddq_f32(_sum, _sum2);
#if __aarch64__
					sum += vaddvq_f32(_sum);
#else
					float32x2_t _sumss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
					_sumss = vpadd_f32(_sumss, _sumss);
					sum += vget_lane_f32(_sumss, 0);
#endif // __aarch64__
#endif // __ARM_NEON

					top_data[p] = sum;
				}
			}
		}

		template<typename Dtype>
		void operation_innerproduct_arm<Dtype>::forward_cpu_i8(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms, std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), 1);
			CHECK_EQ(tops.size(), 1);

			int num = bottoms[0]->num();
			int channels = bottoms[0]->channels();
			int size = bottoms[0]->count(1,4);

			std::shared_ptr<memory::tensor<int8_t>> bottom(new memory::tensor<int8_t>(bottoms[0]->data_shape(), this->params_.device_, bottoms[0]->order()));

			quantize_float32_to_int8(bottoms[0], bottom);

			memory::orderType order = bottom->order();
			if (order == memory::NHWC)
				bottom->convert_order();

			std::shared_ptr<memory::tensor<int>> top(new memory::tensor<int>(std::vector<int>{num, num_output_, 1, 1}, this->params_.device_, memory::NCHW));
#ifdef __aarch64__
			const int8_t* weight_i8_reordered_data = weight_i8_reordered_->cpu_data();

			for (size_t n = 0; n < num; n++)
			{
				const int8_t* bottom_data = bottom->cpu_data() + n * size;
				int* top_data = top->mutable_cpu_data() + n * num_output_;
				
				int8kernel(top_data, bottom_data, weight_i8_reordered_data, 1, size, num_output_, num_output_, 0, 0);
			}
#else
			for (size_t n = 0; n < num; n++)
			{
				const int8_t* bottom_data = bottom->cpu_data() + n * size;
				int* top_data = top->mutable_cpu_data() + n * num_output_;
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int p = 0; p < num_output_; p++)
				{
					const int8_t* weight_i8_data = this->weights_i8_[0]->cpu_data() + p * size;
					for (int i = 0; i < size; i++)
					{
						top_data[p] += bottom_data[i] * weight_i8_data[i];
					}
				}
			}
#endif
			tops[0].reset(new memory::tensor<float>(top->data_shape(), this->params_.device_, top->order()));
			dequantize_int32_to_float32(top, tops[0]);
		}

		template<typename Dtype>
		void operation_innerproduct_arm<Dtype>::quantize_float32_to_int8(const std::shared_ptr<memory::tensor<float>>& src,
			std::shared_ptr<memory::tensor<signed char>>& dst)
		{
			int total_size = src->count();

			dst.reset(new memory::tensor<signed char>(src->data_shape(), this->params_.device_, src->order()));
			const float* bottom = src->cpu_data();
			signed char* bottom_int8_ = dst->mutable_cpu_data();

#ifdef _OPENMP 
#pragma omp parallel for num_threads(2) 
#endif
			for (int i = 0; i < total_size; i++)
			{
				bottom_int8_[i] = float32_to_int8(bottom[i] * this->featmap_scaletable_i8_[0]);
			}
		}

		template<typename Dtype>
		void operation_innerproduct_arm<Dtype>::dequantize_int32_to_float32(std::shared_ptr<memory::tensor<int>>& src,
			std::shared_ptr<memory::tensor<float>>& dst)
		{
			int num = src->num();
			const float* bias_data = nullptr;
			if(this->bias_term_)
				bias_data = this->weights_f32_[1]->cpu_data();

			for (size_t n = 0; n < num; n++)
			{
				const int* top_int32_data = src->cpu_data() + n * num_output_;
				float* top_f32_data = dst->mutable_cpu_data() + n * num_output_;
				if (this->bias_term_)
				{
					for (size_t i = 0; i < num_output_; i++)
					{
						top_f32_data[i] = top_int32_data[i] * scale_in + bias_data[i];
					}
				}
				else
				{
					for (size_t i = 0; i < num_output_; i++)
					{
						top_f32_data[i] = top_int32_data[i] * scale_in;
					}
				}
			}
		}

		INSTANCE_CLASS(operation_innerproduct_arm);
		REGISTE(operation_innerproduct_arm);
	}
}