#include "../../../include/Excalibur/arm/operation_prelu_arm.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"
#include <random>

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_prelu_arm<Dtype>::operation_prelu_arm(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					num_slope_ = atoi(split_string(attrs[i], "=")[1].c_str());
					CHECK_GE(num_slope_, 1);
					if (num_slope_ == 1)
					{
						share_channel_ = true;
					}
				}
				else if (split_string(attrs[i], "=")[0] == "-23330")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported PReLU Attribution " << split_string(attrs[i], "=")[0];
				}
			}
			this->params_.inplace_ = true;
		}

		template<typename Dtype>
		int operation_prelu_arm<Dtype>::init_weights(FILE *fp)
		{
			this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(num_slope_, this->params_.device_, memory::NCHW, nullptr)));
			fread(this->weights_f32_[0]->mutable_cpu_data(), 1, num_slope_ * sizeof(float), fp);
			return num_slope_ * sizeof(float);
		}

		template<typename Dtype>
		int operation_prelu_arm<Dtype>::init_weights()
		{
			std::default_random_engine e;
			std::normal_distribution<float> n(0, 0.3);
			this->weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(num_slope_, this->params_.device_, memory::NCHW, nullptr)));
			for (size_t i = 0; i < num_slope_; i++)
			{
				this->weights_f32_[0]->mutable_cpu_data()[i] = std::abs(n(e));
			}
			return num_slope_ * sizeof(float);
		}

		template<typename Dtype>
		void operation_prelu_arm<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), tops.size());
			tops = bottoms;
			for (size_t i = 0; i < bottoms.size(); i++)
			{
				memory::orderType order = bottoms[i]->order();

				if (order == memory::NHWC)
					bottoms[i]->convert_order();

				int num = bottoms[i]->num();
				int w = bottoms[i]->width();
				int h = bottoms[i]->height();
				int channels = bottoms[i]->channels();
				int size = w * h;

				const float *slope_data = this->weights_f32_[0]->cpu_data();

				for (int num_i = 0; num_i < num; num_i++)
				{
					float *bottom_data = bottoms[i]->mutable_cpu_data() + num_i * channels * size;

#pragma omp parallel for num_threads(2) 
					for (int q = 0; q < channels; q++)
					{
						float* ptr = bottom_data + (q)* size;
						float slope = slope_data[0];
						if (!share_channel_)
						{
							slope = slope_data[q];
						}
#if __ARM_NEON
						int nn = size >> 2;
						int remain = size - (nn << 2);
#else
						int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__

						float32x4_t _zero = vdupq_n_f32(0.f);
						float32x4_t _slope = vdupq_n_f32(slope);
						for (; nn > 0; nn--)
						{
							float32x4_t _p = vld1q_f32(ptr);
							uint32x4_t _lemask = vcleq_f32(_p, _zero);
							float32x4_t _ps = vmulq_f32(_p, _slope);
							_p = vbslq_f32(_lemask, _ps, _p);
							vst1q_f32(ptr, _p);

							ptr += 4;
						}
#else
						if (nn > 0)
						{
							asm volatile(
								"veor       q1, q0, q0          \n"
								"vdup.f32   q2, %4              \n"
								"0:                             \n"
								"pld        [%1, #128]          \n"
								"vld1.f32   {d0-d1}, [%1]  \n"
								"vcle.f32   q3, q0, q1          \n"
								"vmul.f32   q4, q0, q2          \n"
								"vbit.32    q0, q4, q3          \n"
								"subs       %0, #1              \n"
								"vst1.f32   {d0-d1}, [%1]! \n"
								"bne        0b                  \n"
								: "=r"(nn),     // %0
								"=r"(ptr)     // %1
								: "0"(nn),
								"1"(ptr),
								"r"(slope)    // %4
								: "cc", "memory", "q0", "q1", "q2", "q3", "q4"
								);
						}
#endif // __aarch64__
#endif // __ARM_NEON
						for (; remain > 0; remain--)
						{
							if (*ptr < 0)
								*ptr *= slope;

							ptr++;
						}
					}
				}
			}
		}

		INSTANCE_CLASS(operation_prelu_arm);
		REGISTE(operation_prelu_arm);
	}
}