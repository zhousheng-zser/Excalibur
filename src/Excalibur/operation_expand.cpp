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
        }

        template <class Dtype>
        void operation_expand<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 2);
            CHECK_EQ(tops.size(), 1);

            int bottom_w = bottoms[0]->width();
            int bottom_h = bottoms[0]->height();
            int bottom_c = bottoms[0]->channels();
            int num = bottoms[0]->num();
            int bottom_cstep = bottom_w * bottom_h;
            int bottom_count = bottoms[0]->count();

            const float *axis_data = bottoms[1]->cpu_data();
            int expand_c = axis_data[0];
            int expand_h = axis_data[1];
            int expand_w = axis_data[2];

            int top_w = bottom_w * expand_w;
            int top_h = bottom_h * expand_h;
            int top_c = bottom_c * expand_c;

            tops[0].reset(new memory::tensor<float>(std::vector<int>{num, top_c, top_h, top_w}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            if (bottoms[0]->order() == memory::NCHW)
            {
                for (int n = 0; n < num; ++n)
                {
                    const float *bottom_data = bottoms[0]->cpu_data() + bottoms[0]->offset(n);
                    float *top_data = tops[0]->mutable_cpu_data() + tops[0]->offset(n);

                    for (int k = 0; k < expand_c * bottom_c; ++k)
                    {
                        if(k == bottom_c) bottom_data -= bottom_count;
                        for (int j = 0; j < expand_h * bottom_h; ++j)
                        {
                            if (j == bottom_h) bottom_data -= bottom_cstep;
                            for (int i = 0; i < expand_w; ++i)
                            {
                                memcpy(top_data, bottom_data, bottom_w);
                                top_data += bottom_w;
                            }
                            bottom_data += bottom_w;
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