#include "Excalibur/operation_expand.hpp"
#include "Excalibur/math_functions.hpp"
#include "Excalibur/operation_reflector.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        operation_expand<Dtype>::operation_expand(const operation_param &param) : operation<Dtype>(param)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                if (std::stoi(kvs[0]) == 0)
                {
                    this->w_ = std::stof(kvs[1]);
                }
                else if (std::stoi(kvs[0]) == 1)
                {
                    this->h_ = std::stof(kvs[1]);
                }
                else if (std::stoi(kvs[0]) == 2)
                {
                    this->c_ = std::stof(kvs[1]);
                }
                else
                {
                    LOG(FATAL) << "Un-supported Expand Attribution " << kvs[1];
                }
            }
        }

        template <class Dtype>
        void operation_expand<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            // int bottom_w = bottoms[0]->width();
            // int bottom_h = bottoms[0]->height();
            // int bottom_c = bottoms[0]->channels();
            int num = bottoms[0]->num();
            // int bottom_cstep = bottom_w * bottom_h;
            int bottom_count = bottoms[0]->count();
            int top_count = num * c_ * h_ * w_;

            tops[0].reset(new memory::tensor<float>(std::vector<int>{num, c_, h_, w_}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));

            if (bottoms[0]->order() == memory::NCHW)
            {
                for (int n = 0; n < num; ++n)
                {
                    const float *bottom_data = bottoms[0]->cpu_data() + bottoms[0]->offset(n);
                    float *top_data = tops[0]->mutable_cpu_data() + tops[0]->offset(n);

                    if (bottom_count == 1) // input 1
                    {
                        for (int i = 0; i < top_count; ++i)
                        {
                            top_data[i] = *bottom_data;
                        }
                    }
                }
            }
            else
            {
                NOT_IMPLEMENTED;
            }
        }

        INSTANCE_CLASS(operation_expand);
        REGISTE(operation_expand);
    }
}