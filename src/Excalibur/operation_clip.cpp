#include "Excalibur/operation.hpp"
#include "Excalibur/operation_clip.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Primitives/pool_allocator.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        operation_clip<Dtype>::operation_clip(const operation_param &param) : operation<Dtype>(param)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                switch (std::stoi(kvs[0]))
                {
                case 0:
                    this->min_ = std::stof(kvs[1]);
                    break;
                case 1:
                    this->max_ = std::stof(kvs[1]);
                    break;
                default:
                    LOG(FATAL) << "Un-supported Concat Attribution " << kvs[1];
                    break;
                }
            }
        }

        template <class Dtype>
        void operation_clip<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            const float *bptr = bottoms[0]->cpu_data();
            float *tptr = tops[0]->mutable_cpu_data();
            int count = bottoms[0]->count();
            int remain = count;
#if (SIMD_X86_INSTR_SET >= SIMD_X86_AVX_VERSION) && (SIMD_X86_INSTR_SET <= SIMD_X86_AVX2_VERSION) // AVX
            __m256 min = _mm256_set1_ps(min_);
            __m256 max = _mm256_set1_ps(max_);
            for (int i = 0; i + 7 < count; i += 8)
            {
                __m256 x = _mm256_loadu_ps(bptr);
                _mm256_store_ps(tptr, _mm256_max_ps(min, _mm256_min_ps(max, x)));
                bptr += 8;
                tptr += 8;
            }
            remain = count % 8;
#endif
            for (int i = 0; i < remain; ++i)
            {
                tptr[i] = std::min(max_, std::max(min_, bptr[i]));
            }
        }

        INSTANCE_CLASS(operation_clip);
        REGISTE(operation_clip);
    }
}