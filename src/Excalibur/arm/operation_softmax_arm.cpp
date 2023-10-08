#include "../../../include/Excalibur/arm/operation_softmax_arm.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"
#include <cstring>
#include <cmath>
#if __ARM_NEON
#include "arm/neon_mathfun.hpp"
#endif // __ARM_NEON

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_softmax_arm<Dtype>::operation_softmax_arm(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					axis_ = atoi(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "1")
				{
					//do nothing
				}
				else if (split_string(attrs[i], "=")[0] == "-23330")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported Convolution Attribution " << split_string(attrs[i], "=")[0];
				}
			}
		}

		template<typename Dtype>
		void operation_softmax_arm<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), tops.size());

			for (size_t i = 0; i < bottoms.size(); i++)
			{
				memory::orderType order = bottoms[i]->order();
				if (order == memory::NHWC)
					bottoms[i]->convert_order();

				int step;
				int num = bottoms[i]->num();

				if (bottoms[i]->data_shape().size() <= 2)
				{
					step = 1;
					scale_.reset(new memory::tensor<float>(std::vector<int>{1, 1}, -1, memory::NCHW, nullptr));
				}
				else
				{
					step = bottoms[i]->width() * bottoms[i]->height();
					scale_.reset(new memory::tensor<float>(std::vector<int>{1, 1, bottoms[i]->height(), bottoms[i]->width()}, -1, memory::NCHW, nullptr));//单通道，NHWC和NCHW无差别
				}

				tops[i].reset(new memory::tensor<float>(bottoms[i]->data_shape(), bottoms[i]->device(), memory::NCHW, bottoms[i]->allocator()));

				int channels = bottoms[i]->channels();
				int dim = bottoms[i]->count(1, bottoms[i]->data_shape().size());

				sum_multiplier_.reset(new memory::tensor<float>(std::vector<int>{1, step}, -1, memory::NCHW, nullptr));
				float *sum = sum_multiplier_->mutable_cpu_data();
				float* scale_data = scale_->mutable_cpu_data();

				for (int num_i = 0; num_i < num; num_i++)
				{
					const float* bottom_data = bottoms[i]->cpu_data() + num_i * dim;
					float * top_data = tops[i]->mutable_cpu_data() + num_i * dim;

					std::memset(sum, 0, step * sizeof(float));
					std::memcpy(top_data, bottom_data, dim * sizeof(float));
					std::memcpy(scale_data, bottom_data, step * sizeof(float));

					for (int j = 0; j < channels; j++) {
						for (int k = 0; k < step; k++) {
							scale_data[k] = std::max(scale_data[k], bottom_data[j * step + k]);
						}
					}

#pragma omp parallel for
					for (int q = 0; q < channels; q++)
					{
						float* ptr = top_data + (q)* step;
						float* maxptr = scale_data;

#if __ARM_NEON
						int nn = step >> 2;
						int remain = step - (nn << 2);
#else
						int remain = step;
#endif // __ARM_NEON

#if __ARM_NEON
						for (; nn > 0; nn--)
						{
							float32x4_t _p = vld1q_f32(ptr);
							float32x4_t _max = vld1q_f32(maxptr);

							_p = exp_ps(vsubq_f32(_p, _max));

							vst1q_f32(ptr, _p);

							ptr += 4;
							maxptr += 4;
						}
#endif // __ARM_NEON

						for (; remain > 0; remain--)
						{
							*ptr = exp(*ptr - *maxptr);

							ptr++;
							maxptr++;
						}
					}

					for (int q = 0; q < channels; q++)
					{
						float* ptr = top_data + (q)* step;
						float* sumptr = sum;

#if __ARM_NEON
						int nn = step >> 2;
						int remain = step - (nn << 2);
#else
						int remain = step;
#endif // __ARM_NEON

#if __ARM_NEON
						for (; nn > 0; nn--)
						{
							float32x4_t _p = vld1q_f32(ptr);
							float32x4_t _sum = vld1q_f32(sumptr);
							_sum = vaddq_f32(_sum, _p);
							vst1q_f32(sumptr, _sum);

							ptr += 4;
							sumptr += 4;
						}
#endif // __ARM_NEON

						for (; remain > 0; remain--)
						{
							*sumptr += *ptr;

							ptr++;
							sumptr++;
						}
					}

#pragma omp parallel for
					for (int q = 0; q < channels; q++)
					{
						float* ptr = top_data + (q)* step;
						float* sumptr = sum;

#if __ARM_NEON
						int nn = step >> 2;
						int remain = step - (nn << 2);
#else
						int remain = step;
#endif // __ARM_NEON

#if __ARM_NEON
						for (; nn > 0; nn--)
						{
							float32x4_t _p = vld1q_f32(ptr);
							float32x4_t _sum = vld1q_f32(sumptr);
#if __aarch64__
							_p = vdivq_f32(_p, _sum);
#else
							_p = div_ps(_p, _sum);
#endif // __aarch64__
							vst1q_f32(ptr, _p);

							ptr += 4;
							sumptr += 4;
						}
#endif // __ARM_NEON

						for (; remain > 0; remain--)
						{
							*ptr /= *sumptr;

							ptr++;
							sumptr++;
						}
					}
				}
			}
		}

		INSTANCE_CLASS(operation_softmax_arm);
		REGISTE(operation_softmax_arm);
	}
}