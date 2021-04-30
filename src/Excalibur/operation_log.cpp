#include "Excalibur/operation_log.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Excalibur/math_functions.hpp"
#include <algorithm>
#include <cfloat>

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        operation_log<Dtype>::operation_log(const operation_param &param) : operation<Dtype>(param)
        {
        }

        template <class Dtype>
        void operation_log<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            const float *bottom_data = bottoms[0]->cpu_data();
            float *top_data = tops[0]->mutable_cpu_data();
            for (int i = 0; i < bottoms[0]->count(); ++i)
            {
                top_data[i] = std::log(bottom_data[i]);
            }
        }

        INSTANCE_CLASS(operation_log);
        REGISTE(operation_log);
    }
}