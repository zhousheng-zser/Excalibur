#include "Excalibur/operation_leakyrelu.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Excalibur/math_functions.hpp"
#include <algorithm>
#include <cfloat>

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        operation_leakyrelu<Dtype>::operation_leakyrelu(const operation_param &param) : operation<Dtype>(param)
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
                default:
                    LOG(FATAL) << "Un-supported Leakyrelu Attribution " << kvs[0];
                    break;
                }
            }
        }

        template <class Dtype>
        void operation_leakyrelu<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            const float *bottom_data = bottoms[0]->cpu_data();
            float *top_data = tops[0]->mutable_cpu_data();
            int size = bottoms[0]->count();
            int remain = size;
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) // AVX
            for (int i = 0; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_load_ps(bottom_data);
                __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), _p);
                __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), _p);
                _p = _mm256_add_ps(pos, _mm256_mul_ps(neg, _mm256_set1_ps(alpha_)));
                _mm256_store_ps(top_data, _p);
                top_data += 8;
                bottom_data += 8;
            }
            remain = size % 8;
#endif
            for (int i = 0; i < remain; ++i)
            {
                top_data[i] = bottom_data[i] < 0 ? this->alpha_ * bottom_data[i] : bottom_data[i];
            }
        }

#ifndef USE_CUDA
        STUB_GPU(operation_leakyrelu);
#endif

        INSTANCE_CLASS(operation_leakyrelu);
        REGISTE(operation_leakyrelu);
    }
}