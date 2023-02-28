#include "Excalibur/operation_reduction.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Excalibur/math_functions.hpp"
#include <algorithm>
#include <cfloat>

namespace glasssix
{
    namespace excalibur
    {
        template <typename T>
        struct reduction_op_sumsq
        {
            T operator()(const T &x, const T &y) const
            {
                return x + y * y;
            }
        };

        template <typename T>
        struct reduction_op_add
        {
            T operator()(const T &x, const T &y) const
            {
                return x + y;
            }
        };

        template <typename T>
        struct post_process_sqrt
        {
            T operator()(const T &x) const
            {
                return static_cast<T>(sqrt(x));
            }
        };

        template <typename Op, typename Op2>
        static void reduction_op_keepdims(const std::shared_ptr<memory::tensor<float>> &bottom, std::shared_ptr<memory::tensor<float>> &top, float v0, bool reduce_w, bool reduce_h, bool reduce_c)
        {
            Op op;
            int width = bottom->width();
            int height = bottom->height();
            int channels = bottom->channels();
            int dims = (width == 1 ? 0 : 1) + (height == 1 ? 0 : 1) + (channels == 1 ? 0 : 1);
            if (dims == 0) dims = 1;
            if (dims == 1)
            {
                top.reset(new memory::tensor<float>(1, -1, memory::NCHW, nullptr));
                const float *ptr = bottom->cpu_data();

                float sum = v0;
                for (int i = 0; i < width; i++)
                {
                    sum += op(v0, ptr[i]);
                }
                top->mutable_cpu_data()[0] = sum;
            }
        }

        template <typename MathOp>
        static void reduction_post_process(std::shared_ptr<memory::tensor<float>> &top)
        {
            MathOp mathop;

            int width = top->width();
            int height = top->height();
            int channels = top->channels();
            int dims = (width == 1 ? 0 : 1) + (height == 1 ? 0 : 1) + (channels == 1 ? 0 : 1);
            if (dims == 0) dims = 1;
            float *top_data = top->mutable_cpu_data();
            if (dims == 1)
            {
                for (int i = 0; i < width; i++)
                {
                    top_data[i] = mathop(top_data[i]);
                }
            }
        }

        template <class Dtype>
        operation_reduction<Dtype>::operation_reduction(const operation_param &param) : operation<Dtype>(param)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                if (std::stoi(kvs[0]) == 0)
                {
                    this->operation_ = std::stof(kvs[1]);
                }
                else if (std::stoi(kvs[0]) == 1)
                {
                    this->axes_ = std::stof(kvs[1]);
                }
                else if (std::stoi(kvs[0]) == 2)
                {
                    this->keepdims_ = std::stof(kvs[1]);
                }
                else if (std::stoi(kvs[0]) == -23330)
                {
                    //do nothing
                }
                else
                {
                    LOG(FATAL) << "Un-supported Reduction Attribution " << kvs[1];
                }
            }
        }

        template <typename Op, typename Op2, typename Op3>
        static void reduction(const std::shared_ptr<memory::tensor<float>> &bottom, std::shared_ptr<memory::tensor<float>> &top, float v0, bool reduce_w, bool reduce_h, bool reduce_c, bool post_process, int keepdims)
        {
            if (keepdims)
            {
                reduction_op_keepdims<Op, Op2>(bottom, top, v0, reduce_w, reduce_h, reduce_c);
            }
            else
            {
                // reduction_op<Op, Op2>(a, b, v0, reduce_w, reduce_h, reduce_c, opt);
            }

            if (post_process)
            {
                reduction_post_process<Op3>(top);
            }
        }

        static void reduction_mean(const std::shared_ptr<memory::tensor<float>> &bottom, std::shared_ptr<memory::tensor<float>> &top)
        {
            /*
            * support 3dims only e.g 1*3*40*40 -> 1*1*40*40
            */
            auto shape = bottom->data_shape();
            CHECK_EQ(shape[0], 1);
            CHECK_GT(shape[1], 1);
            CHECK_GT(shape[2], 1);
            CHECK_GT(shape[3], 1);
            int channels_num = shape[1];
            int steps = shape[2] * shape[3];

            shape[1] = 1; //reset outTensor channels
            top.reset(new memory::tensor<float>(shape, bottom->device(), bottom->order(), bottom->allocator()));
            float* bottom_data = bottom->mutable_cpu_data();
            float* top_data = top->mutable_cpu_data();
            const int top_count = top->count();
            for (size_t i = 0; i < steps; ++i) {
                float sum = 0;
                for (size_t c = 0; c < channels_num; ++c) {
                    sum += bottom_data[c * steps + i];
                }
                top_data[i] = sum / channels_num;
            }
        }

        template <class Dtype>
        void operation_reduction<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            bool reduce_w = false;
            bool reduce_h = false;
            bool reduce_c = false;

            int width = bottoms[0]->width();
            int height = bottoms[0]->height();
            int channels = bottoms[0]->channels();
            int dims = (width == 1 ? 0 : 1) + (height == 1 ? 0 : 1) + (channels == 1 ? 0 : 1);

            if (dims == 1)
            {
                reduce_w = true;
            }

            if (operation_ == ReductionOp_L2)
                reduction<reduction_op_sumsq<float>, reduction_op_add<float>, post_process_sqrt<float>>(bottoms[0], tops[0], 0.f, reduce_w, reduce_h, reduce_c, true, keepdims_);
            if (operation_ == ReductionOp_MEAN)
                reduction_mean(bottoms[0], tops[0]); //support 3dims only
            else
                NOT_IMPLEMENTED;
        }

        INSTANCE_CLASS(operation_reduction);
        REGISTE(operation_reduction);
    }
}