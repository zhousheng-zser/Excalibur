#include "Excalibur/operation_pad.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Excalibur/operation_make_border.hpp"
#include "Excalibur/math_functions.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        operation_pad<Dtype>::operation_pad(const operation_param &param) : operation<Dtype>(param)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            { 
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                switch (std::stoi(kvs[0]))
                {
                case 0:
                    top_ = std::stof(kvs[1]);
                    break;
                case 1:
                    bottom_ = std::stoi(kvs[1]);
                    break;
                case 2:
                    left_ = std::stoi(kvs[1]);
                    break;
                case 3:
                    right_ = std::stoi(kvs[1]);
                    break;
                case 4:
                    mode_ = std::stoi(kvs[1]);
                    break;
                case 5:
                    constant_value_ = std::stof(kvs[1]);
                    break;
                case 7:
                    front_ = std::stoi(kvs[1]);
                    break;
                case 8:
                    behind_ = std::stoi(kvs[1]);
                    break;
                default:
                    LOG(FATAL) << "Un-supported Leakyrelu Attribution " << kvs[0];
                    break;
                }
            }
        }

        template <class Dtype>
        void operation_pad<Dtype>::copy_make_border_image(const std::shared_ptr<memory::tensor<float>> &bottom, std::shared_ptr<memory::tensor<float>> &top)
        {
            int w = bottom->width();
            int h = bottom->height();
            int c = bottom->channels();

            if (bottom->order() == memory::NCHW)
            {
                if (mode_ == CONSTANT)
                {
                    float* top_data = top->mutable_cpu_data();
                    for (int n = 0; n < top->num(); ++n)
                    {
                        math_functions::cpu_set(front_ * w * h, constant_value_, top_data);
                        top_data += front_ * w * h;

                        math_functions::excalibur_copy(w * h * c, bottom->cpu_data(), top_data, this->params_.device_);
                        top_data += c * w * h;

                        math_functions::cpu_set(behind_ * w * h, constant_value_, top_data);
                        top_data += behind_ * w * h;
                    }

                    make_border(top, top, top_, bottom_, left_, right_, border_constant, constant_value_);
                }
                //else if (type_ == REFLECT)
                //{
                //    for (int n = 0; n < bottom->num(); ++n)
                //    {
                //        for (int ch = 0; ch < src_c; ++ch)
                //        {
                //            // next channel
                //            const float *bottom_data = bottom->cpu_data() + bottom->offset(n, ch);
                //            float *top_data = top->mutable_cpu_data() + top->offset(n, ch);

                //            int y = 0;
                //            // fill top
                //            bottom_data += top_num * src_w;
                //            for (; y < top_num; ++y)
                //            {
                //                int x = 0;
                //                for (; x < left_num; ++x)
                //                {
                //                    top_data[x] = bottom_data[left_num - x];
                //                }
                //                if (src_w < 12)
                //                {
                //                    top_data[x] = bottom_data[x - left_num];
                //                }
                //                else
                //                {
                //                    memcpy(top_data + left_num, bottom_data, sizeof(float) * src_w);
                //                    x += src_w;
                //                }
                //                for (; x < dst_w; ++x)
                //                {
                //                    top_data[x] = bottom_data[src_w - (x - left_num - src_w) - 2];
                //                }

                //                top_data += dst_w;
                //                bottom_data -= src_w;
                //            }

                //            // fill center
                //            for (; y < top_num + src_h; ++y)
                //            {
                //                int x = 0;
                //                for (; x < left_num; ++x)
                //                {
                //                    top_data[x] = bottom_data[left_num - x];
                //                }
                //                if (src_w < 12)
                //                {
                //                    for (; x < left_num + src_w; ++x)
                //                    {
                //                        top_data[x] = bottom_data[x - left_num];
                //                    }
                //                }
                //                else
                //                {
                //                    memcpy(top_data + left_num, bottom_data, sizeof(float) * src_w);
                //                    x += src_w;
                //                }
                //                for (; x < dst_w; ++x)
                //                {
                //                    top_data[x] = bottom_data[src_w - (x - left_num - src_w) - 2];
                //                }

                //                top_data += dst_w;
                //                bottom_data += src_w;
                //            }

                //            // fill bottom
                //            bottom_data -= 2 * src_w;
                //            for (; y < dst_h; ++y)
                //            {
                //                int x = 0;
                //                for (; x < left_num; ++x)
                //                {
                //                    top_data[x] = bottom_data[left_num - x];
                //                }
                //                if (src_w < 12)
                //                {
                //                    for (; x < left_num + src_w; ++x)
                //                    {
                //                        top_data[x] = bottom_data[x - left_num];
                //                    }
                //                }
                //                else
                //                {
                //                    memcpy(top_data + left_num, bottom_data, sizeof(float) * src_w);
                //                    x += src_w;
                //                }
                //                for (; x < dst_w; ++x)
                //                {
                //                    top_data[x] = bottom_data[src_w - (x - left_num - src_w) - 2];
                //                }

                //                top_data += dst_w;
                //                bottom_data -= src_w;
                //            }
                //        }
                //    }
                //}
                //else if (type_ == EDGE)
                //{
                //    for (int n = 0; n < bottom->num(); ++n)
                //    {
                //        for (int ch = 0; ch < src_c; ++ch)
                //        {
                //            // next channel
                //            const float *bottom_data = bottom->cpu_data() + bottom->offset(n, ch);
                //            float *top_data = top->mutable_cpu_data() + top->offset(n, ch);

                //            int y = 0;
                //            //fill top
                //            for (; y < top_num; ++y)
                //            {
                //                int x = 0;
                //                for (; x < left_num; ++x)
                //                {
                //                    top_data[x] = bottom_data[0];
                //                }
                //                if (src_w < 12)
                //                {
                //                    for (; x < left_num + src_w; ++x)
                //                    {
                //                        top_data[x] = bottom_data[x - left_num];
                //                    }
                //                }
                //                else
                //                {
                //                    memcpy(top_data + left_num, bottom_data, sizeof(float) * src_w);
                //                    x += src_w;
                //                }
                //                for (; x < dst_w; ++x)
                //                {
                //                    top_data[x] = bottom_data[src_w - 1];
                //                }

                //                top_data += dst_w;
                //            }

                //            //fill center
                //            for (; y < top_num + src_h; ++y)
                //            {
                //                int x = 0;
                //                for (; x < left_num; ++x)
                //                {
                //                    top_data[x] = bottom_data[0];
                //                }
                //                if (src_w < 12)
                //                {
                //                    for (; x < left_num + src_w; ++x)
                //                    {
                //                        top_data[x] = bottom_data[x - left_num];
                //                    }
                //                }
                //                else
                //                {
                //                    memcpy(top_data + left_num, bottom_data, sizeof(float) * src_w);
                //                    x += src_w;
                //                }
                //                for (; x < dst_w; ++x)
                //                {
                //                    top_data[x] = bottom_data[src_w - 1];
                //                }

                //                top_data += dst_w;
                //                bottom_data += src_w;
                //            }

                //            //fill bottom
                //            bottom_data -= src_w;
                //            for (; y < dst_h; ++y)
                //            {
                //                int x = 0;
                //                for (; x < left_num; ++x)
                //                {
                //                    top_data[x] = bottom_data[0];
                //                }
                //                if (src_w < 12)
                //                {
                //                    for (; x < left_num + src_w; ++x)
                //                    {
                //                        top_data[x] = bottom_data[x - left_num];
                //                    }
                //                }
                //                else
                //                {
                //                    memcpy(top_data + left_num, bottom_data, sizeof(float) * src_w);
                //                    x += src_w;
                //                }
                //                for (; x < dst_w; ++x)
                //                {
                //                    top_data[x] = bottom_data[src_w - 1];
                //                }

                //                top_data += dst_w;
                //            }
                //        }
                //    }
                //}
                else
                {
                    NOT_IMPLEMENTED;
                }
            }
            else
            {
                NOT_IMPLEMENTED;
            }
        }

        template <class Dtype>
        void operation_pad<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            //check pad num
            if (!(top_ || bottom_ || left_ || right_ || front_ || behind_))
            {
                tops[0] = bottoms[0];
                return;
            }
            tops[0].reset(new memory::tensor<float>(std::vector<int>{bottoms[0]->num(), bottoms[0]->channels() + front_ + behind_, bottoms[0]->height(), bottoms[0]->width()}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            // pad
            this->copy_make_border_image(bottoms[0], tops[0]);
        }

#ifndef USE_CUDA
        STUB_GPU(operation_pad);
#endif

        INSTANCE_CLASS(operation_pad);
        REGISTE(operation_pad);
    }
}