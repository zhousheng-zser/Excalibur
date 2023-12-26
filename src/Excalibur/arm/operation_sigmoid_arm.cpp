#include "../../../include/Excalibur/arm/operation_sigmoid_arm.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"
#include <cmath>
#if __ARM_NEON
#include "arm/neon_mathfun.hpp"
#endif // __ARM_NEON

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_sigmoid_arm<Dtype>::operation_sigmoid_arm(const operation_param& param) : operation<Dtype>(param)
		{
			this->params_.inplace_ = true;
		}

		template<typename Dtype>
		void operation_sigmoid_arm<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), tops.size());

			for (size_t i = 0; i < bottoms.size(); i++)
			{
				tops[i].reset(new memory::tensor<float>(bottoms[i]->data_shape(), bottoms[i]->device(), bottoms[i]->order(), bottoms[i]->allocator()));

				int num = bottoms[i]->num();
				int w = bottoms[i]->width();
				int h = bottoms[i]->height();
				int channels = bottoms[i]->channels();
				int size = w * h;

				for (int num_i = 0; num_i < num; num_i++)
				{
					const float *bottom_base = bottoms[i]->cpu_data() + num_i * channels * size;
					float* top_base = tops[i]->mutable_cpu_data() + num_i * channels * size;
#pragma omp parallel for num_threads(2) 
					for (int q = 0; q < channels; q++)
					{
						const float* bottom_data = bottom_base + (q)* size;
						float* top_data = top_base + (q)* size;

#if __ARM_NEON
						int nn = size >> 2;
						int remain = size - (nn << 2);
#else
						int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
						float32x4_t _one = vdupq_n_f32(1.f);
						for (; nn > 0; nn--)
						{
							float32x4_t _p = vld1q_f32(bottom_data);
							_p = vnegq_f32(_p);
							_p = exp_ps(_p);
							_p = vaddq_f32(_p, _one);
							float32x4_t _outp = vrecpeq_f32(_p);
							_outp = vmulq_f32(vrecpsq_f32(_p, _outp), _outp);
							//             _outp = vmulq_f32(vrecpsq_f32(_p, _outp), _outp);
							vst1q_f32(top_data, _outp);

							bottom_data += 4;
							top_data += 4;
						}
#endif // __ARM_NEON
						for (; remain > 0; remain--)
						{
							*top_data = 1.f / (1.f + std::exp(-*bottom_data));

							bottom_data++;
							top_data++;
						}
					}
				}
			}
		}

		INSTANCE_CLASS(operation_sigmoid_arm);
		REGISTE(operation_sigmoid_arm);
	}
}