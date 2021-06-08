#include "Excalibur/operation_binaryop.hpp"
#include "Excalibur/math_functions.hpp"
#include "Excalibur/operation_reflector.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <typename Op>
        static void binary_op(const std::shared_ptr<memory::tensor<float>> &bottom1, const std::shared_ptr<memory::tensor<float>> &bottom2, std::shared_ptr<memory::tensor<float>> &top)
        {
            Op op;
            int bottom_w1 = bottom1->width();
            int bottom_h1 = bottom1->height();
            int bottom_c1 = bottom1->channels();
            int size = bottom_w1 * bottom_h1;

            int bottom_w2 = bottom2->width();
            int bottom_h2 = bottom2->height();
            int bottom_c2 = bottom2->channels();

            top.reset(new memory::tensor<float>(bottom1->data_shape(), bottom1->device(), bottom1->order(), bottom1->allocator()));
            for (int q = 0; q < bottom_c1; ++q)
            {
                const float *b1 = bottom1->cpu_data() + bottom1->offset(0, q);
                const float *b2 = bottom2->cpu_data() + bottom2->offset(0, q);
                float *outptr = top->mutable_cpu_data() + top->offset(0, q);

                for (int i = 0; i < size; ++i)
                {
                    outptr[i] = op(b1[i], b2[i]);
                }
            }
        }

        template <typename Op>
        static void binary_op_scalar(const std::shared_ptr<memory::tensor<float>> &bottom, std::shared_ptr<memory::tensor<float>> &top, float coef)
        {
            Op op;
            int bottom_w = bottom->width();
            int bottom_h = bottom->height();
            int bottom_c = bottom->channels();
            int size = bottom_w * bottom_h;

            top.reset(new memory::tensor<float>(bottom->data_shape(), bottom->device(), bottom->order(), bottom->allocator()));
            for (int q = 0; q < bottom_c; ++q)
            {
                const float *b = bottom->cpu_data() + bottom->offset(0, q);
                float *outptr = top->mutable_cpu_data() + top->offset(0, q);

                for (int i = 0; i < size; ++i)
                {
                    outptr[i] = op(b[i], coef);
                }
            }
        }

        struct binary_op_add
        {
            float operator()(const float &x, const float &y) const
            {
                return x + y;
            }
        };

        struct binary_op_sub
        {
            float operator()(const float &x, const float &y) const
            {
                return x - y;
            }
        };

        struct binary_op_mul
        {
            float operator()(const float &x, const float &y) const
            {
                return x * y;
            }
        };

        struct binary_op_div
        {
            float operator()(const float &x, const float &y) const
            {
                return x / y;
            }
        };

        struct binary_op_max
        {
            float operator()(const float &x, const float &y) const
            {
                return std::max(x, y);
            }
        };

        struct binary_op_min
        {
            float operator()(const float &x, const float &y) const
            {
                return std::min(x, y);
            }
        };

        struct binary_op_pow
        {
            float operator()(const float &x, const float &y) const
            {
                return (float)pow(x, y);
            }
        };

        struct binary_op_rsub
        {
            float operator()(const float &x, const float &y) const
            {
                return y - x;
            }
        };

        struct binary_op_rdiv
        {
            float operator()(const float &x, const float &y) const
            {
                return y / x;
            }
        };

        template <class Dtype>
        operation_binaryop<Dtype>::operation_binaryop(const operation_param &param) : operation<Dtype>(param)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                switch (std::stoi(kvs[0]))
                {
                case 0:
                    this->op_type_ = (OperationType)std::stoi(kvs[1]);
                    break;
                case 1:
                    this->with_scalar_ = std::stoi(kvs[1]);
                    break;
                case 2:
                    this->coef_ = std::stof(kvs[1]);
                    break;
                default:
                    LOG(FATAL) << "Un-supported BinaryOp Attribution " << kvs[0];
                    break;
                }
            }
        }

        template <class Dtype>
        void operation_binaryop<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_GE(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            if (bottoms[0]->order() == memory::NCHW)
            {
                if (with_scalar_)
                {
                    switch (op_type_)
                    {
                    case Operation_ADD:
                        binary_op_scalar<binary_op_add>(bottoms[0], tops[0], coef_);
                        break;
                    case Operation_SUB:
                        binary_op_scalar<binary_op_sub>(bottoms[0], tops[0], coef_);
                        break;
                    case Operation_MUL:
                        binary_op_scalar<binary_op_mul>(bottoms[0], tops[0], coef_);
                        break;
                    case Operation_DIV:
                        binary_op_scalar<binary_op_div>(bottoms[0], tops[0], coef_);
                        break;
                    default:
                        NOT_IMPLEMENTED;
                        break;
                    }
                }
                else
                {
                    switch (op_type_)
                    {
                    case Operation_ADD:
                        binary_op<binary_op_add>(bottoms[0], bottoms[1], tops[0]);
                        break;
                    case Operation_SUB:
                        binary_op<binary_op_sub>(bottoms[0], bottoms[1], tops[0]);
                        break;
                    case Operation_MUL:
                        binary_op<binary_op_mul>(bottoms[0], bottoms[1], tops[0]);
                        break;
                    case Operation_DIV:
                        binary_op<binary_op_div>(bottoms[0], bottoms[1], tops[0]);
                        break;
                    default:
                        NOT_IMPLEMENTED;
                        break;
                    }
                }
            }
            else
            {
                NOT_IMPLEMENTED;
            }
        }

#ifndef USE_CUDA
        STUB_GPU(operation_binaryop);
#endif

        INSTANCE_CLASS(operation_binaryop);
        REGISTE(operation_binaryop);
    }
}