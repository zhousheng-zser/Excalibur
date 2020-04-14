#include "arm/softmax_arm.hpp"
#include <cfloat>
#include <cmath>
#include <cstring>
#include <algorithm>

#if __ARM_NEON
#include <arm_neon.h>
#include "arm/neon_mathfun.hpp"
#endif // __ARM_NEON

using namespace glasssix::memory;

void glasssix::excalibur::softmax_arm::Forward_cpu(const std::shared_ptr<tensor<float>>& bottom, std::shared_ptr<tensor<float>>& top)
{
	// value = exp( value - global max value )
	// sum all value
	// value = value / sum

	order_ = bottom->order();
	if (order_ == NCHW)
	{
		outer_num_ = bottom->num();

		if (bottom->data_shape().size() <= 2)
		{
			inner_num_ = 1;
			scale_.reset(new tensor<float>(std::vector<int>{1, 1}, -1, NCHW));
		}
		else
		{
			inner_num_ = bottom->width() * bottom->height();
			scale_.reset(new tensor<float>(std::vector<int>{1, 1, bottom->height(), bottom->width()}, -1, NCHW));//单通道，NHWC和NCHW无差别
		}

		top.reset(new tensor<float>(bottom->data_shape()/*std::vector<int>{outer_num_, bottom->channels()}*/, -1, NCHW));
		
		int channels = bottom->channels();
		int dim = bottom->count(1, bottom->data_shape().size());

		sum_multiplier_.reset(new tensor<float>(std::vector<int>{1, inner_num_}, -1, NCHW));
		float *sum = sum_multiplier_->mutable_cpu_data();
		float* scale_data = scale_->mutable_cpu_data();

		for (int num_i = 0; num_i < outer_num_; num_i++)
		{
			const float* bottom_data = bottom->cpu_data() + num_i * dim;
			float * top_data = top->mutable_cpu_data() + num_i * dim;

			std::memset(sum, 0, inner_num_ * sizeof(float));
			std::memcpy(top_data, bottom_data, dim * sizeof(float));
			std::memcpy(scale_data, bottom_data, inner_num_ * sizeof(float));

			for (int j = 0; j < channels; j++) {
				for (int k = 0; k < inner_num_; k++) {
					scale_data[k] = std::max(scale_data[k], bottom_data[j * inner_num_ + k]);
				}
			}

#pragma omp parallel for
			for (int q = 0; q<channels; q++)
			{
				float* ptr = top_data + (q) * inner_num_;
				float* maxptr = scale_data;

#if __ARM_NEON
				int nn = inner_num_ >> 2;
				int remain = inner_num_ - (nn << 2);
#else
				int remain = inner_num_;
#endif // __ARM_NEON

#if __ARM_NEON
				for (; nn>0; nn--)
				{
					float32x4_t _p = vld1q_f32(ptr);
					float32x4_t _max = vld1q_f32(maxptr);

					_p = exp_ps(vsubq_f32(_p, _max));

					vst1q_f32(ptr, _p);

					ptr += 4;
					maxptr += 4;
				}
#endif // __ARM_NEON

				for (; remain>0; remain--)
				{
					*ptr = exp(*ptr - *maxptr);

					ptr++;
					maxptr++;
				}
			}

			for (int q = 0; q<channels; q++)
			{
				float* ptr = top_data + (q) * inner_num_;
				float* sumptr = sum;

#if __ARM_NEON
				int nn = inner_num_ >> 2;
				int remain = inner_num_ - (nn << 2);
#else
				int remain = inner_num_;
#endif // __ARM_NEON

#if __ARM_NEON
				for (; nn>0; nn--)
				{
					float32x4_t _p = vld1q_f32(ptr);
					float32x4_t _sum = vld1q_f32(sumptr);
					_sum = vaddq_f32(_sum, _p);
					vst1q_f32(sumptr, _sum);

					ptr += 4;
					sumptr += 4;
				}
#endif // __ARM_NEON

				for (; remain>0; remain--)
				{
					*sumptr += *ptr;

					ptr++;
					sumptr++;
				}
			}

#pragma omp parallel for
			for (int q = 0; q<channels; q++)
			{
				float* ptr = top_data + (q) * inner_num_;
				float* sumptr = sum;

#if __ARM_NEON
				int nn = inner_num_ >> 2;
				int remain = inner_num_ - (nn << 2);
#else
				int remain = inner_num_;
#endif // __ARM_NEON

#if __ARM_NEON
				for (; nn>0; nn--)
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

				for (; remain>0; remain--)
				{
					*ptr /= *sumptr;

					ptr++;
					sumptr++;
				}
			}
		}
	}
	else
	{
		softmax::Forward_cpu(bottom, top);
	}
}
