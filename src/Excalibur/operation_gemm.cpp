#include "Excalibur/operation_gemm.hpp"
#include "Excalibur/math_functions.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Excalibur/math_functions.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template<class Dtype>
        operation_gemm<Dtype>::operation_gemm(const operation_param &param): operation<Dtype>(param)
        {

        }

        template<class Dtype>
        void operation_gemm<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 2);
            CHECK_EQ(tops.size(), 1);

            int num = bottoms[0]->num();

            int M = bottoms[0]->height();
            int K = bottoms[0]->width();
            int N = bottoms[1]->width();

            tops[0].reset(new memory::tensor<float>(std::vector<int>{num, 1, M, N}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));

            const float *bottom_A_data = bottoms[0]->cpu_data();
            const float *bottom_B_data = bottoms[1]->cpu_data();
            float *top_data = tops[0]->mutable_cpu_data();

            math_functions::cpu_sgemm(CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, bottom_A_data, bottom_B_data, 0.0f, top_data);
        }

        INSTANCE_CLASS(operation_gemm);
        REGISTE(operation_gemm);
    }
}