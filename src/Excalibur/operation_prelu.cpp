#include "../../include/Excalibur/operation_prelu.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"
#include <random>

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_prelu<Dtype>::operation_prelu(const operation_param& param) : operation<Dtype>(param)
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
			params_.inplace_ = true;
		}

		template<typename Dtype>
		int operation_prelu<Dtype>::init_weights(FILE *fp)
		{
			weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(num_slope_, params_.device_, memory::NCHW, nullptr)));
			fread(weights_f32_[0]->mutable_cpu_data(), 1, num_slope_ * sizeof(float), fp);
			return num_slope_ * sizeof(float);
		}

		template<typename Dtype>
		int operation_prelu<Dtype>::init_weights()
		{
			std::default_random_engine e;
			std::normal_distribution<float> n(0, 0.3);
			weights_f32_.push_back(std::shared_ptr<memory::tensor<float>>(new memory::tensor<float>(num_slope_, params_.device_, memory::NCHW, nullptr)));
			for (size_t i = 0; i < num_slope_; i++)
			{
				weights_f32_[0]->mutable_cpu_data()[i] = abs(n(e));
			}
			return num_slope_ * sizeof(float);
		}

		template<typename Dtype>
		void operation_prelu<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), tops.size());
			for (size_t i = 0; i < bottoms.size(); i++)
			{
				tops[i].reset(new memory::tensor<float>(bottoms[i]->data_shape(), bottoms[i]->device(), bottoms[i]->order(), bottoms[i]->allocator()));
				float* top_data = tops[i]->mutable_cpu_data();
				const float* bottom_data = bottoms[i]->cpu_data();
				const float* slope_data = weights_f32_[0]->cpu_data();
				if (bottoms[i]->order() == memory::NCHW)
				{
					for (size_t n = 0; n < bottoms[i]->num(); n++)
					{
						int ch = bottoms[i]->channels();
						for (size_t c = 0; c < ch; c++)
						{
							int step = bottoms[i]->count(2, 4);
							int offset = n * ch * step + c * step;
							int slope_id = share_channel_ ? c : 0;
#if (SIMD_X86_INSTR_SET >= SIMD_X86_SSE_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_SSE4_2_VERSION) //SSE
							int simd_times = (count - count % 4) / 4;
							__m128 slope_vec = _mm_broadcast_ss(slope_data + slope_id);
							__m128 zero_vec = _mm_setzero_ps();
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
							for (int j = 0; j < simd_times; j++)
							{
								__m256 d = _mm_load_ps(bottom_data + 4 * j);
								d = _mm_add_ps(_mm_max_ps(zero_vec, d), _mm_mul_ps(slope_vec, _mm_min_ps(zero_vec, d)));
								_mm_store_ps(top_data + 4 * j, d);
							}
							for (int j = 4 * simd_times; j < step; j++)
							{
								top_data[offset + j] =
									bottom_data[offset + j] >= 0.0f ? bottom_data[offset + j] : slope_data[slope_id] * bottom_data[offset + j];
							}
#elif (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX
							int simd_times = (step - step % 8) / 8;
							__m256 slope_vec = _mm256_broadcast_ss(slope_data + slope_id);
							__m256 zero_vec = _mm256_setzero_ps();
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
							for (int j = 0; j < simd_times; j++)
							{
								__m256 d = _mm256_load_ps(bottom_data + 8 * j);
								d = _mm256_add_ps(_mm256_max_ps(zero_vec, d), _mm256_mul_ps(slope_vec, _mm256_min_ps(zero_vec, d)));
								_mm256_store_ps(top_data + 8 * j, d);
							}
							for (int j = 8 * simd_times; j < step; j++)
							{
								top_data[offset + j] =
									bottom_data[offset + j] >= 0.0f ? bottom_data[offset + j] : slope_data[slope_id] * bottom_data[offset + j];
							}
#else // Native code
#ifdef _OPENMP
#pragma omp parallel for num_threads(2) 
#endif
							for (int j = 0; j < step; j++)
							{
								top_data[offset + j] =
									bottom_data[offset + j] >= 0.0f ? bottom_data[offset + j] : slope_data[slope_id] * bottom_data[offset + j];
							}
#endif
						}
					}
				}
				else if (bottoms[i]->order() == memory::NHWC)
				{
					for (size_t n = 0; n < bottoms[i]->num(); n++)
					{
						NOT_IMPLEMENTED;
					}
				}
				else
				{
					LOG(FATAL) << "Un-supported Order Type.";
				}
			}
		}


		template<typename Dtype>
		void operation_prelu<Dtype>::forward_gpu_f32(
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

		INSTANCE_CLASS(operation_prelu);
		REGISTE(operation_prelu);
	}
}