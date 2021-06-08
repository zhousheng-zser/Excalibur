#include "Excalibur/operation_scatternd.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Excalibur/math_functions.hpp"
#include <algorithm>
#include <cfloat>

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        operation_scatternd<Dtype>::operation_scatternd(const operation_param &param) : operation<Dtype>(param)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                switch (std::stoi(kvs[0]))
                {
                case 0:
                    for (std::string v : split_string(kvs[1], ","))
                    {
                        this->indices_.push_back(std::stoi(v));
                    }
                    break;
                default:
                    LOG(FATAL) << "Un-supported Scatternd Type " << kvs[1];
                    break;
                }
            }
        }

        template <class Dtype>
        void operation_scatternd<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 2);
            CHECK_EQ(tops.size(), 1);
            CHECK_EQ(bottoms[1]->data_shape().size(), indices_.size());

            int width = bottoms[0]->width();
            int height = bottoms[0]->height();
            int channels = bottoms[0]->channels();
            int num = bottoms[0]->num();
            const float *updates_init_ptr = bottoms[1]->cpu_data();
            tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            tops[0] = bottoms[0];

            for (int n = 0; n < num; ++n)
            {
                float *output_init_ptr = tops[0]->mutable_cpu_data() + tops[0]->offset(n);
                if (bottoms[0]->order() == memory::NCHW)
                {
                    for (int idx = 0; idx < indices_.size(); ++idx)
                    {
                        CHECK_GE(indices_[idx], 0);
                        CHECK_LT(indices_[idx], channels);

                        const float *upptr = updates_init_ptr + idx * width * height;
                        float *outptr = output_init_ptr + indices_[idx] * width * height;
                        memcpy(outptr, upptr, height * width);
                    }
                }
                else
                {
                    NOT_IMPLEMENTED;
                }
            }
        }

        INSTANCE_CLASS(operation_scatternd);
        REGISTE(operation_scatternd);
    }
}