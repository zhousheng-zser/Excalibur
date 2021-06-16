#include "Excalibur/operation.hpp"
#include "Excalibur/operation_gather.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Primitives/pool_allocator.hpp"
#include "Excalibur/math_functions.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        operation_gather<Dtype>::operation_gather(const operation_param &param) : operation<Dtype>(param), axis_(0)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                switch (std::stoi(kvs[0]))
                {
                case 0:
                    //axis
                    axis_ = std::stoi(kvs[1]);
                    break;
                case 1:
                    for (std::string &i : split_string(kvs[1], ","))
                    {
                        indexs_.push_back(std::stoi(i));
                    }
                    break;
                default:
                    LOG(FATAL) << "Un-supported gather Attribution " << kvs[1];
                    break;
                }
            }
        }

        template <class Dtype>
        void operation_gather<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);
            int num = bottoms[0]->num();
            int channels = bottoms[0]->channels();
            int width = bottoms[0]->width();
            int height = bottoms[0]->height();
            const float *bottom_data = bottoms[0]->cpu_data();

            if (channels > 1)
            {
                if (axis_ == 0)
                {
                    NOT_IMPLEMENTED;
                }
                else if (axis_ == 1)
                {
                    NOT_IMPLEMENTED;
                }
                else if (axis_ == 2)
                {
                    tops[0].reset(new memory::tensor<float>(std::vector<int>{num, (int)indexs_.size(), channels, width}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
                }
            }
            else
            {
                if (axis_ == 0)
                {
                    NOT_IMPLEMENTED;
                }
                else if (axis_ == 1)
                {
                    NOT_IMPLEMENTED;
                }
                else if (axis_ == 2)
                {
                    tops[0].reset(new memory::tensor<float>(std::vector<int>{num, 1, height, (int)indexs_.size()}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
                }
            }
            float *top_data = tops[0]->mutable_cpu_data();

            if (bottoms[0]->order() == memory::NCHW)
            {
                for (int n = 0; n < num; ++n)
                {
                    if (channels > 1)
                    {
                        if (axis_ == 0)
                        {
                            NOT_IMPLEMENTED;
                        }
                        else if (axis_ == 1)
                        {
                            NOT_IMPLEMENTED;
                        }
                        else if (axis_ == 2)
                        {
                            for (int i = 0; i < indexs_.size(); ++i)
                            {
                                for (int c = 0; c < channels; ++c)
                                {
                                    std::copy(bottom_data + bottoms[0]->offset(n, c, indexs_[0]), bottom_data + bottoms[0]->offset(n, c, indexs_[0]) + width, top_data);
                                    top_data += width;
                                }
                            }
                        }
                    }
                    else
                    {
                        if (axis_ == 0)
                        {
                            NOT_IMPLEMENTED;
                        }
                        else if (axis_ == 1)
                        {
                            NOT_IMPLEMENTED;
                        }
                        else if (axis_ == 2)
                        {
                            for (int i = 0; i < indexs_.size(); ++i)
                            {
                                for (int h = 0; h < height; ++h)
                                {
                                    *(top_data++) = *(bottom_data + bottoms[0]->offset(n, 0, h, indexs_[i]));
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                NOT_IMPLEMENTED;
            }
        }

        INSTANCE_CLASS(operation_gather);
        REGISTE(operation_gather);
    }
}