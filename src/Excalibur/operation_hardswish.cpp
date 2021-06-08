#include "Excalibur/operation_hardswish.hpp"
#include "Excalibur/math_functions.hpp"
#include "Excalibur/operation_reflector.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        operation_hardswish<Dtype>::operation_hardswish(const operation_param &param) : offset_(3.f), scale_(6.f), threshold_(6.f), operation<Dtype>(param)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                switch (std::stoi(kvs[0]))
                {
                case 0:
                    this->threshold_ = std::stof(kvs[1]);
                    break;
                case 1:
                    this->scale_ = std::stof(kvs[1]);
                    break;
                case 2:
                    this->offset_ = std::stof(kvs[1]);
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
            int size = bottoms[0]->count();
            float *top_data = tops[0]->mutable_cpu_data();
            const float *bottom_data = bottoms[0]->cpu_data();
            for (int i = 0; i < size; ++i)
            {
                top_data[i] = bottom_data[i] * (std::min(std::max(0.f, bottom_data[i] + offset_), threshold_)) / scale_;
            }
        }

#ifndef USE_CUDA
        STUB_GPU(operation_hardswish);
#endif

        INSTANCE_CLASS(operation_hardswish);
        REGISTE(operation_hardswish);
    }
}