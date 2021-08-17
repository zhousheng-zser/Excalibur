#include "Excalibur/operation.hpp"
#include "Excalibur/operation_unaryop.hpp"
#include "Excalibur/operation_reflector.hpp"
#include "Primitives/pool_allocator.hpp"

namespace glasssix
{
    namespace excalibur
    {
        struct unary_op_abs
        {
            float operator()(const float &x) const
            {
                return (float)fabs(x);
            }
        };

        struct unary_op_neg
        {
            float operator()(const float &x) const
            {
                return -x;
            }
        };

        struct unary_op_floor
        {
            float operator()(const float &x) const
            {
                return (float)floor(x);
            }
        };

        struct unary_op_ceil
        {
            float operator()(const float &x) const
            {
                return (float)ceil(x);
            }
        };

        struct unary_op_square
        {
            float operator()(const float &x) const
            {
                return x * x;
            }
        };

        struct unary_op_sqrt
        {
            float operator()(const float &x) const
            {
                return (float)sqrt(x);
            }
        };

        struct unary_op_rsqrt
        {
            float operator()(const float &x) const
            {
                return (float)(1.f / sqrt(x));
            }
        };

        struct unary_op_exp
        {
            float operator()(const float &x) const
            {
                return (float)exp(x);
            }
        };

        struct unary_op_log
        {
            float operator()(const float &x) const
            {
                return (float)log(x);
            }
        };

        struct unary_op_sin
        {
            float operator()(const float &x) const
            {
                return (float)sin(x);
            }
        };

        struct unary_op_cos
        {
            float operator()(const float &x) const
            {
                return (float)cos(x);
            }
        };

        struct unary_op_tan
        {
            float operator()(const float &x) const
            {
                return (float)tan(x);
            }
        };

        struct unary_op_asin
        {
            float operator()(const float &x) const
            {
                return (float)asin(x);
            }
        };

        struct unary_op_acos
        {
            float operator()(const float &x) const
            {
                return (float)acos(x);
            }
        };

        struct unary_op_atan
        {
            float operator()(const float &x) const
            {
                return (float)atan(x);
            }
        };

        struct unary_op_reciprocal
        {
            float operator()(const float &x) const
            {
                return 1.f / x;
            }
        };

        struct unary_op_tanh
        {
            float operator()(const float &x) const
            {
                return (float)tanh(x);
            }
        };

        template <class Dtype>
        operation_unaryop<Dtype>::operation_unaryop(const operation_param &param) : operation<Dtype>(param)
        {
            std::vector<std::string> attrs = split_string(param.specific_params_, " ");
            for (int i = 0; i < attrs.size(); ++i)
            {
                std::vector<std::string> kvs = split_string(attrs[i], "=");
                if (std::stoi(kvs[0]) == 0)
                {
                    this->op_type_ = std::stof(kvs[1]);
                }
                else
                {
                    LOG(FATAL) << "Un-supported Unaryop Attribution " << kvs[1];
                }
            }
        }

        template <typename Op>
        static void unary_op_inplace(const std::shared_ptr<memory::tensor<float>> &bottom, std::shared_ptr<memory::tensor<float>> &top)
        {
            Op op;

            int size = bottom->count();
            const float *bottom_data = bottom->cpu_data();
            float *top_data = top->mutable_cpu_data();

            for (int i = 0; i < size; i++)
            {
                top_data[i] = op(bottom_data[i]);
            }
        }

        template <class Dtype>
        void operation_unaryop<Dtype>::forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            if (op_type_ == Operation_ABS)
                return unary_op_inplace<unary_op_abs>(bottoms[0], tops[0]);

            if (op_type_ == Operation_NEG)
                return unary_op_inplace<unary_op_neg>(bottoms[0], tops[0]);

            if (op_type_ == Operation_FLOOR)
                return unary_op_inplace<unary_op_floor>(bottoms[0], tops[0]);

            if (op_type_ == Operation_CEIL)
                return unary_op_inplace<unary_op_ceil>(bottoms[0], tops[0]);

            if (op_type_ == Operation_SQUARE)
                return unary_op_inplace<unary_op_square>(bottoms[0], tops[0]);

            if (op_type_ == Operation_SQRT)
                return unary_op_inplace<unary_op_sqrt>(bottoms[0], tops[0]);

            if (op_type_ == Operation_RSQRT)
                return unary_op_inplace<unary_op_rsqrt>(bottoms[0], tops[0]);

            if (op_type_ == Operation_EXP)
                return unary_op_inplace<unary_op_exp>(bottoms[0], tops[0]);

            if (op_type_ == Operation_LOG)
                return unary_op_inplace<unary_op_log>(bottoms[0], tops[0]);

            if (op_type_ == Operation_SIN)
                return unary_op_inplace<unary_op_sin>(bottoms[0], tops[0]);

            if (op_type_ == Operation_COS)
                return unary_op_inplace<unary_op_cos>(bottoms[0], tops[0]);

            if (op_type_ == Operation_TAN)
                return unary_op_inplace<unary_op_tan>(bottoms[0], tops[0]);

            if (op_type_ == Operation_ASIN)
                return unary_op_inplace<unary_op_asin>(bottoms[0], tops[0]);

            if (op_type_ == Operation_ACOS)
                return unary_op_inplace<unary_op_acos>(bottoms[0], tops[0]);

            if (op_type_ == Operation_ATAN)
                return unary_op_inplace<unary_op_atan>(bottoms[0], tops[0]);

            if (op_type_ == Operation_RECIPROCAL)
                return unary_op_inplace<unary_op_reciprocal>(bottoms[0], tops[0]);

            if (op_type_ == Operation_TANH)
                return unary_op_inplace<unary_op_tanh>(bottoms[0], tops[0]);
        }

        INSTANCE_CLASS(operation_unaryop);
        REGISTE(operation_unaryop);
    }
}