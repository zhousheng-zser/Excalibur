#include "Excalibur/operation_hardswish.hpp"
#include "Excalibur/math_functions.hpp"
#include "Excalibur/operation_reflector.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        operation_hardswish<Dtype>::operation_hardswish(const operation_param &param) : operation<Dtype>(param)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                switch (std::stoi(kvs[0]))
                {
                case 0:
                    this->alpha_ = std::stof(kvs[1]);
                    break;
                case 1:
                    this->beta_ = std::stof(kvs[1]);
                    break;
                default:
                    LOG(FATAL) << "Un-supported HardSwish Attribution " << kvs[0];
                    break;
                }
            }
        }

        template <class Dtype>
        void operation_hardswish<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);
            tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            float *top_data = tops[0]->mutable_cpu_data();
            const float *bottom_data = bottoms[0]->cpu_data();
            int size = bottoms[0]->count();
            int remain = size;
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) // AVX
            __m256 zero = _mm256_set1_ps(0.f);
            __m256 one = _mm256_set1_ps(1.f);
            for (int i = 0; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(bottom_data);
                __m256 _ans = _mm256_set1_ps(beta_);
                _ans = _mm256_fmadd_ps(_p, _mm256_set1_ps(alpha_), _ans);
                _ans = _mm256_max_ps(_ans, zero);
                _ans = _mm256_min_ps(_ans, one);
                _ans = _mm256_mul_ps(_ans, _p);
                _mm256_store_ps(top_data, _ans);
                top_data += 8;
                bottom_data += 8;
                remain -= 8;
            }
#endif // AVX
            
            for (int i = 0; i < remain; ++i)
            {
                top_data[i] = bottom_data[i] * std::min(std::max(0.f, alpha_ * bottom_data[i] + beta_), 1.f);
            }
        }

#ifndef USE_CUDA
        STUB_GPU(operation_hardswish);
#endif

        INSTANCE_CLASS(operation_hardswish);
        REGISTE(operation_hardswish);
    }
}