#include "../../../include/Excalibur/arm/operation_innerproduct_arm.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"
#include <random>


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
				if (bias_term_)
				{
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(num_output_, this->params_.device_, memory::NCHW, nullptr)));
					fread(this->weights_f32_[1]->mutable_cpu_data(), 1, num_output_ * sizeof(float), fp);
					mem += num_output_ * sizeof(float);
				}
				return mem;
			}
			else if (quantize_tag == 871224)
			{
				this->weights_i8_.push_back(std::shared_ptr<memory::tensor<signed char>>(new memory::tensor<signed char>(weight_data_size_, this->params_.device_, memory::NCHW, nullptr)));
				fread(this->weights_i8_[0]->mutable_cpu_data(), 1, weight_data_size_ * sizeof(signed char), fp);
				mem += weight_data_size_ * sizeof(signed char);
				// fake float32 data, just for code consistency
				this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(1, this->params_.device_, memory::NCHW, nullptr)));
				if (weight_data_size_ % 4 != 0)
				{
					fread(this->weights_f32_[0]->mutable_cpu_data(), 1, (4 - weight_data_size_ % 4) * sizeof(signed char), fp);
					mem += 1 * sizeof(signed char);
				}
				if (bias_term_)
				{
					this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(num_output_, this->params_.device_, memory::NCHW, nullptr)));
					fread(this->weights_f32_[1]->mutable_cpu_data(), 1, num_output_ * sizeof(float), fp);
					mem += num_output_ * sizeof(signed char);
				}
				this->weights_scaletable_i8_.resize(1);
				fread(this->weights_scaletable_i8_.data(), 1, 1 * sizeof(float), fp);
				this->featmap_scaletable_i8_.resize(1);
				fread(this->featmap_scaletable_i8_.data(), 1, 1 * sizeof(float), fp);
				mem += 2 * sizeof(signed char);
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
				if (bias_term_)
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
				if (bias_term_)
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
				mem += (1 + 1) * sizeof(float);
			}
			return mem;
		}

		INSTANCE_CLASS(operation_innerproduct_arm);
		REGISTE(operation_innerproduct_arm);
	}
}