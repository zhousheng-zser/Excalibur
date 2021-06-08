#include "Excalibur/operation_exp.hpp"
#include "Excalibur/math_functions.hpp"
#include "Excalibur/operation_reflector.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template<class Dtype>
        operation_exp<Dtype>::operation_exp(const operation_param &param): operation<Dtype>(param)
        {

        }

        template<class Dtype>
        void operation_exp<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            const float *bottom_data = bottoms[0]->cpu_data();
            float *top_data = tops[0]->mutable_cpu_data();

            for(int i = 0; i < bottoms[0]->count(); ++i)
            {
                top_data[i] = std::exp(bottom_data[i]);
            }
        }

        INSTANCE_CLASS(operation_exp);
        REGISTE(operation_exp);
    }
}