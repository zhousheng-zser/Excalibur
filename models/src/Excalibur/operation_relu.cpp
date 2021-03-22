#include "../../include/Excalibur/operation_relu.hpp"
#include "../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		operation_relu<Dtype>::operation_relu(const operation_param& param) : operation<Dtype>(param)
		{
			auto attrs = split_string(param.specific_params_, " ");
			for (size_t i = 0; i < attrs.size(); i++)
			{
				if (split_string(attrs[i], "=")[0] == "0")
				{
					slope_ = atof(split_string(attrs[i], "=")[1].c_str());
				}
				else if (split_string(attrs[i], "=")[0] == "-23330")
				{
					//do nothing
				}
				else
				{
					LOG(FATAL) << "Un-supported ReLU Attribution " << split_string(attrs[i], "=")[0];
				}
			}
			this->params_.inplace_ = true;
		}

		template<typename Dtype>
		void operation_relu<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
			std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
		{
			CHECK_EQ(bottoms.size(), tops.size());
			for (size_t i = 0; i < bottoms.size(); i++)
			{
				tops[i].reset(new memory::tensor<float>(bottoms[i]->data_shape(), bottoms[i]->device(), bottoms[i]->order(), bottoms[i]->allocator()));
				float* top_data = tops[i]->mutable_cpu_data();
				const float* bottom_data = bottoms[i]->cpu_data();
				const int count = bottoms[i]->count();
#if (SIMD_X86_INSTR_SET >= SIMD_X86_SSE_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_SSE4_2_VERSION) //SSE
				int simd_times = (count - count % 4) / 4;
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int j = 0; j < simd_times; j++)
				{
					__m128 d = _mm_load_ps(bottom_data + 4 * j);
					d = _mm_max_ps(_mm_setzero_ps(), d);
					_mm_store_ps(top_data + 4 * j, d);
				}
				for (int j = 4 * simd_times; j < bottoms[i]->count(); j++)
				{
					top_data[j] = bottom_data[j] >= 0.0f ? bottom_data[j] : 0.0f;
				}
#elif (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) //AVX
				int simd_times = (count - count % 8) / 8;
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int j = 0; j < simd_times; j++)
				{
					__m256 d = _mm256_load_ps(bottom_data + 8 * j);
					d = _mm256_max_ps(_mm256_setzero_ps(), d);
					_mm256_store_ps(top_data + 8 * j, d);
				}
				for (int j = 8 * simd_times; j < bottoms[i]->count(); j++)
				{
					top_data[j] = bottom_data[j] >= 0.0f ? bottom_data[j] : 0.0f;
				}
#else
#ifdef _OPENMP
#pragma omp parallel for num_threads(2)
#endif
				for (int j = 0; j < bottoms[i]->count(); j++)
				{
					top_data[j] = bottom_data[j] >= 0.0f ? bottom_data[j] : 0.0f;
				}
#endif
			}
		}

#ifndef USE_CUDA
		STUB_GPU(operation_relu);
#endif

		INSTANCE_CLASS(operation_relu);
		REGISTE(operation_relu);
	}
}