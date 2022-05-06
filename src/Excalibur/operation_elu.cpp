#include <cmath>

#include "Excalibur/operation.hpp"
#include "Excalibur/operation_elu.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Primitives/pool_allocator.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template<class Dtype>
        operation_elu<Dtype>::operation_elu(const operation_param &param): operation<Dtype>(param)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for(int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                switch (std::stoi(kvs[0]))
                {
                case 0:
                    this->alpha_ = std::stof(kvs[1]);
                    break;
                default:
                    break;
                }
            }
        }

        template<class Dtype>
        void operation_elu<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            float *top_data = tops[0]->mutable_cpu_data();
            const float *bottom_data = bottoms[0]->cpu_data();

            for(int i = 0; i < bottoms[0]->count(); ++i)
            {
                top_data[i] = bottom_data[i] < 0 ? (std::exp(bottom_data[i]) - 1) * this->alpha_ : bottom_data[i];
            }
        }


#ifndef USE_CUDA
        STUB_GPU(operation_relu);
#endif

        INSTANCE_CLASS(operation_elu);
        REGISTE(operation_elu);
    }
}