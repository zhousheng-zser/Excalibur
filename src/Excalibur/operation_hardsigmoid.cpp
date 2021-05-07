#include "Excalibur/operation_hardsigmoid.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Excalibur/math_functions.hpp"
#include <algorithm>
#include <cfloat>

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        operation_hardsigmoid<Dtype>::operation_hardsigmoid(const operation_param &param) : operation<Dtype>(param)
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
                    LOG(FATAL) << "Un-supported Hardsigmoid Attribution " << kvs[0];
                    break;
                }
            }
        }

        template <class Dtype>
        void operation_hardsigmoid<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(bottoms.size(), 1);

            tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            const float *bottom_data = bottoms[0]->cpu_data();
            float *top_data = tops[0]->mutable_cpu_data();

            for (int i = 0; i < bottoms[0]->count(); ++i)
            {
                top_data[i] = std::max(0.f, std::min(1.f, this->alpha_ * bottom_data[i] + this->beta_));
            }
        }

        INSTANCE_CLASS(operation_hardsigmoid);
        REGISTE(operation_hardsigmoid);
    }
}