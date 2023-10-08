#include "Excalibur/operation_binaryop.hpp"
#include "Excalibur/math_functions.hpp"
#include "Excalibur/operation_reflector.hpp"

namespace glasssix
{
    namespace excalibur
    {
        static void bottom_assign(
            std::shared_ptr<memory::tensor<float>>& bottom_max,
            std::shared_ptr<memory::tensor<float>>& bottom_min,
            const std::shared_ptr<memory::tensor<float>>& bottom1,
            const std::shared_ptr<memory::tensor<float>>& bottom2)
        {
            bool secondBottomMax = false;
            auto b1_shape = bottom1->data_shape();
            auto b2_shape = bottom2->data_shape();
            CHECK_EQ(b1_shape.size(), 4);
            CHECK_EQ(b2_shape.size(), 4);
            if (b1_shape != b2_shape) {
                int dims_1 = (bottom1->width() == 1 ? 0 : 1) + (bottom1->height() == 1 ? 0 : 1) + (bottom1->channels() == 1 ? 0 : 1);
                int dims_2 = (bottom2->width() == 1 ? 0 : 1) + (bottom2->height() == 1 ? 0 : 1) + (bottom2->channels() == 1 ? 0 : 1);
                if (dims_2 > dims_1) {
                    secondBottomMax = true;
                }
                else if (dims_2 == dims_1) {
                    for (int i = 1; i < 4; i++) {
                        if (b2_shape[i] > b1_shape[i]) {
                            secondBottomMax = true;
                            break;
                        }
                    }
                }
            }
            if (secondBottomMax) {
                bottom_max = bottom2;
                bottom_min = bottom1;
            }
            else {
                bottom_max = bottom1;
                bottom_min = bottom2;
            }
        }

        template <typename Op>
        static void binary_op(const std::shared_ptr<memory::tensor<float>>& bottom1, const std::shared_ptr<memory::tensor<float>>& bottom2, std::shared_ptr<memory::tensor<float>>& top)
        {
            Op op;
            std::shared_ptr<memory::tensor<float>> bottom_max;
            std::shared_ptr<memory::tensor<float>> bottom_min;
            bottom_assign(bottom_max, bottom_min, bottom1, bottom2);

            int NUM = bottom_max->num();
            int bmax_W = bottom_max->width();
            int bmax_H = bottom_max->height();
            int bmax_C = bottom_max->channels();
            std::vector<int> bmax_shape = bottom_max->data_shape();
            int bmax_dims = (bmax_W == 1 ? 0 : 1) + (bmax_H == 1 ? 0 : 1) + (bmax_C == 1 ? 0 : 1);
            bmax_dims = bmax_dims == 0 ? 1 : bmax_dims;
            const float* bmax_data = bottom_max->cpu_data();
            int max_CHWstp = bmax_C * bmax_H * bmax_W;
            int maxHWsize = bmax_H * bmax_W;

            int minNUM = bottom_min->num();
            int bmin_W = bottom_min->width();
            int bmin_H = bottom_min->height();
            int bmin_C = bottom_min->channels();
            std::vector<int> bmin_shape = bottom_min->data_shape();
            int bmin_dims = (bmin_W == 1 ? 0 : 1) + (bmin_H == 1 ? 0 : 1) + (bmin_C == 1 ? 0 : 1);
            bmin_dims = bmin_dims == 0 ? 1 : bmin_dims;
            const float* bmin_data = bottom_min->cpu_data();
            int min_CHWstp = bmin_C * bmin_H * bmin_W;
            int minHWsize = bmin_H * bmin_W;

            if (bmax_shape == bmin_shape)
            {
                top.reset(new memory::tensor<float>(bottom_max->data_shape(), bottom_max->device(), bottom_max->order(), bottom_max->allocator()));
                float* top_data = top->mutable_cpu_data();
                CHECK_EQ(bottom_max->count(), bottom_min->count());
                for (int i = 0; i < bottom_max->count(); ++i) {
                    top_data[i] = op(bmax_data[i], bmin_data[i]);
                }
                return;
            }
			else if (min_CHWstp == 1) {
				top.reset(new memory::tensor<float>(bottom_max->data_shape(), bottom_max->device(), bottom_max->order(), bottom_max->allocator()));
				float* top_data = top->mutable_cpu_data();
				for (int num = 0; num < NUM; ++num) {
					const float bmin_Value = bmin_data[(num % minNUM) * min_CHWstp];
                    const float* maxptr = bmax_data + num * max_CHWstp;
                    float* outptr = top_data + num * max_CHWstp;

					for (int i = 0; i < max_CHWstp; ++i) {
						outptr[i] = op(maxptr[i], bmin_Value);
					}
				}
				return;
			}
            else if (bmax_dims == 3)
            {
                if (bmin_dims == 3 && bmax_C != bmin_C && bmax_C > 3)
                {
                    // e.g for: [ 1 * 6 * 2 * 160 * 160 ] op [ 1 * 1 * 2 * 160 * 160 ]
                    // in excalibur to be [ 1 * 12 * 160 * 160 ] op [ 1 * 2 * 160 * 160 ] }
                    top.reset(new memory::tensor<float>(bottom_max->data_shape(), bottom_max->device(), bottom_max->order(), bottom_max->allocator()));
                    float* top_data = top->mutable_cpu_data();

                    CHECK_EQ(minHWsize, maxHWsize);
                    const int HWsize = maxHWsize;

                    const double m = bmin_C;
                    const double nd = double(bmax_C) / m;
                    if (ceil(nd) == floor(nd))
                    {
                        const int n = nd;
                        if (n >= m) {
                            /*
                            * m*n OP m
                            * e.g.: [ BatchNum * 4 * 24 * 160 * 160] OP [ BatchNum * 4 * 1 * 160 * 160]
                            * this 24=n 4=m
                            * [24 * B] OP [1 * B] loop 4 times //B:160 * 160
                            */
                            const int step = n * HWsize;
                            for (int num = 0; num < NUM; ++num) { // BatchNum
                                float* top_data_num = top_data + num * max_CHWstp;
                                const float* bmax_data_num = bmax_data + num * max_CHWstp;
                                const float* bmin_data_num = bmin_data + (num % minNUM) * min_CHWstp;

                                for (int i = 0; i < m; ++i) {
                                    float* outptr = top_data_num + i * step;
                                    const float* maxptr = bmax_data_num + i * step;
                                    const float* minptr = bmin_data_num + i * HWsize;
                                    for (int j = 0; j < step; ++j) {
                                        outptr[j] = op(maxptr[j], minptr[j % HWsize]);
                                    }
                                }
                            }
                            return;
                        }
                        else {
                            NOT_IMPLEMENTED;
                        }
                    }
                }
                else if (bmin_dims == 2) {
                    CHECK_EQ(minHWsize, maxHWsize);
                    const int HWsize = maxHWsize;
                    top.reset(new memory::tensor<float>(bottom_max->data_shape(), bottom_max->device(), bottom_max->order(), bottom_max->allocator()));
                    float* top_data = top->mutable_cpu_data();
                    for (int num = 0; num < NUM; ++num) { // BatchNum
                        float* top_data_num = top_data + num * max_CHWstp;
                        const float* bmax_data_num = bmax_data + num * max_CHWstp;
                        const float* bmin_data_num = bmin_data + (num % minNUM) * min_CHWstp;

                        for (size_t c = 0; c < bmax_C; c++)
                        {
                            const float* ChnlPtrBmax = bmax_data_num + c * HWsize; // maxBottom assign channel
                            float* ChnlPtrTop = top_data_num + c * HWsize;
                            for (size_t i = 0; i < HWsize; i++)
                            {
                                ChnlPtrTop[i] = op(bmin_data_num[i], ChnlPtrBmax[i]);
                            }
                        }
                    }

                    return;
                }
                else if (bmin_dims == 1 && bmin_C > 1) //dim3(N,C,H,W) OP (N,C,1,1)
                {
                    top.reset(new memory::tensor<float>(bottom_max->data_shape(), bottom_max->device(), bottom_max->order(), bottom_max->allocator()));
                    float* top_data = top->mutable_cpu_data();
                    for (int num = 0; num < NUM; num++) {
                        float* top_data_num = top_data + num * max_CHWstp;
                        const float* bmax_data_num = bmax_data + num * max_CHWstp;
                        const float* bmin_data_num = bmin_data + (num % minNUM) * min_CHWstp;

                        for (int q = 0; q < bmax_C; q++)
                        {
                            const float* maxptr = bmax_data_num + q * maxHWsize;
                            float* outptr = top_data_num + q * maxHWsize;
                            for (int i = 0; i < maxHWsize; i++)
                            {
                                outptr[i] = op(bmin_data_num[q], maxptr[i]);
                            }
                        }
                    }
                    return;
                }
            }
            else if (bmax_dims == 2)
            {
                if (bmin_dims == 1)
                {
                    top.reset(new memory::tensor<float>(bottom_max->data_shape(), bottom_max->device(), bottom_max->order(), bottom_max->allocator()));
                    float* top_data = top->mutable_cpu_data();

                    for (int num = 0; num < NUM; num++) {
                        float* outptr = top_data + num * max_CHWstp;
                        const float* bmax_data_num = bmax_data + num * max_CHWstp;
                        const float* bmin_data_num = bmin_data + (num % minNUM) * min_CHWstp;
                        for (int h = 0; h < bmax_H; ++h)
                        {
                            for (int w = 0; w < bmax_W; ++w)
                            {
                                outptr[h * bmax_W + w] = op(bmax_data_num[w], bmin_data_num[w]);
                            }
                            bmax_data_num += bmax_W;
                        }
                    }
                    return;
                }
                else if (bmin_dims == 2 && bmax_shape != bmin_shape)
                {
                    //e.g for: [1 * 1 * 4 * 4] op [1 * 1 * 2 * 4]
                    //top.reset(new memory::tensor<float>(bottom_max->data_shape(), bottom_max->device(), bottom_max->order(), bottom_max->allocator()));
                    //float* outptr = top->mutable_cpu_data();
                    NOT_IMPLEMENTED;
                }
            }
            else if (bmax_dims == 1)
            {
                if (bmin_dims == 1)
                {
                    CHECK_EQ(min_CHWstp, max_CHWstp);
                    const int CHWstp = min_CHWstp;
                    if (CHWstp > 1)
                    {
                        top.reset(new memory::tensor<float>(bottom_max->data_shape(), bottom_max->device(), bottom_max->order(), bottom_max->allocator()));
                        float* top_data = top->mutable_cpu_data();

                        for (int num = 0; num < NUM; num++) {
                            float* outptr = top_data + num * CHWstp;
                            const float* maxptr = bmax_data + num * CHWstp;
                            const float* minptr = bmin_data + (num % minNUM) * CHWstp;

                            for (int i = 0; i < CHWstp; ++i)
                            {
                                outptr[i] = op(maxptr[i], minptr[i]);
                            }
                        }
                        return;
                    }
                }
                else if (bmin_dims == 3)
                {
                    if (bmax_W == 1 && bmax_H == 1 && bmax_C == bmin_C)
                    {
                        // special type 3
                        top.reset(new memory::tensor<float>(bottom_min->data_shape(), bottom_max->device(), bottom_max->order(), bottom_max->allocator()));
                        float* top_data = top->mutable_cpu_data();

                        for (int num = 0; num < NUM; num++) {
                            float* top_data_num = top_data + num * max_CHWstp;
                            const float* bmax_data_num = bmax_data + num * max_CHWstp;
                            const float* bmin_data_num = bmin_data + (num % minNUM) * min_CHWstp;

                            for (int q = 0; q < bmax_C; q++)
                            {
                                const float* minptr = bmin_data_num + q * minHWsize;
                                float* outptr = top_data_num + q * minHWsize;
                                for (int i = 0; i < minHWsize; i++)
                                {
                                    outptr[i] = op(bmax_data_num[q], minptr[i]);
                                }
                            }
                        }
                        return;
                    }
                }
            }

            NOT_IMPLEMENTED;
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

        INSTANCE_CLASS(operation_binaryop);
        REGISTE(operation_binaryop);
    }
}