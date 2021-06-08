#include "Excalibur/operation_slice.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Excalibur/math_functions.hpp"
#include <algorithm>
#include <cfloat>

namespace glasssix
{
    namespace excalibur
    {
        static inline int cal_target_size(int a, int b)
        {
            return a % b ? a / b + 1 : a / b;
        }

        template <class Dtype>
        operation_slice<Dtype>::operation_slice(const operation_param &param) : starts_(3, 0), ends_(3, INT_MAX), steps_(3, 1), operation<Dtype>(param)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                switch (std::stoi(kvs[0]))
                {
                case 0:
                    this->starts_.clear();
                    for (std::string v : split_string(kvs[1], ","))
                    {
                        this->starts_.push_back(std::stoi(v));
                    }
                    break;
                case 1:
                    this->ends_.clear();
                    for (std::string v : split_string(kvs[1], ","))
                    {
                        int val = std::stoi(v);
                        this->ends_.push_back(val == -1 ? INT_MAX : val);
                    }
                    break;
                case 2:
                    this->steps_.clear();
                    for (std::string v : split_string(kvs[1], ","))
                    {
                        this->steps_.push_back(std::stoi(v));
                    }
                    break;
                default:
                    LOG(FATAL) << "Un-supported Slice Attribution " << kvs[0];
                    break;
                }
            }
        }

        template <class Dtype>
        void operation_slice<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            if (!starts_[0] && !starts_[1] && !starts_[2] && ends_[0] == INT_MAX && ends_[1] == INT_MAX && ends_[2] == INT_MAX && steps_[0] == 1 && steps_[1] == 1 && steps_[2] == 1)
            {
                tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
                tops[0] = bottoms[0];
                return;
            }

            int src_w = bottoms[0]->width();
            int src_h = bottoms[0]->height();
            int src_c = bottoms[0]->channels();
            int num = bottoms[0]->num();

            int _w = ends_[0] == INT_MAX ? src_w - starts_[0] : ends_[0] - starts_[0];
            int _h = ends_[1] == INT_MAX ? src_h - starts_[1] : ends_[1] - starts_[1];
            int _c = ends_[2] == INT_MAX ? src_c - starts_[2] : ends_[2] - starts_[2];

            int dst_w = cal_target_size(_w, steps_[0]);
            int dst_h = cal_target_size(_h, steps_[1]);
            int dst_c = cal_target_size(_c, steps_[2]);

            tops[0].reset(new memory::tensor<float>(std::vector<int>{num, dst_c, dst_h, dst_w}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));

            for (int n = 0; n < num; ++n)
            {
                float *top_data = tops[0]->mutable_cpu_data() + tops[0]->offset(n);
                if (bottoms[0]->order() == memory::NCHW)
                {
                    for (int ch = 0; ch < dst_c; ++ch)
                    {
                        for (int h = 0; h < dst_h; ++h)
                        {
                            for (int w = 0; w < dst_w; ++w)
                            {
                                const float *bottom_data = bottoms[0]->cpu_data() + bottoms[0]->offset(n, starts_[2] + ch * steps_[2], starts_[1] + h * steps_[1], starts_[0] + w * steps_[0]);
                                *(top_data++) = *bottom_data;
                            }
                        }
                    }
                }
                else
                {
                    NOT_IMPLEMENTED;
                }
            }
        }

        INSTANCE_CLASS(operation_slice);
        REGISTE(operation_slice);
    }
}