#include "../../../include/Excalibur/operation_unaryop.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
    namespace excalibur
    {
#ifdef USE_CUDA
        struct unary_op_abs
        {
            __host__ __device__ float operator()(const float& x) const
            {
                return fabs(x);
            }
        };

        struct unary_op_neg
        {
            __host__ __device__ float operator()(const float& x) const
            {
                return -x;
            }
        };

        struct unary_op_floor
        {
            __host__ __device__ float operator()(const float& x) const
            {
                return floorf(x);
            }
        };

        struct unary_op_ceil
        {
            __host__ __device__ float operator()(const float& x) const
            {
                return ceilf(x);
            }
        };

        struct unary_op_square
        {
            __host__ __device__ float operator()(const float& x) const
            {
                return x * x;
            }
        };

        struct unary_op_sqrt
        {
            __host__ __device__ float operator()(const float& x) const
            {
                return sqrt(x);
            }
        };

        struct unary_op_rsqrt
        {
            __host__ __device__ float operator()(const float& x) const
            {
                return (1.f / sqrt(x));
            }
        };

        struct unary_op_exp
        {
            __host__ __device__ float operator()(const float& x) const
            {
                return exp(x);
            }
        };

        struct unary_op_log
        {
            __host__ __device__ float operator()(const float& x) const
            {
                return log(x);
            }
        };

        struct unary_op_sin
        {
            __host__ __device__ float operator()(const float& x) const
            {
                return sin(x);
            }
        };

        struct unary_op_cos
        {
            __host__ __device__ float operator()(const float& x) const
            {
                return cos(x);
            }
        };

        struct unary_op_tan
        {
            __host__ __device__ float operator()(const float& x) const
            {
                return tan(x);
            }
        };

        struct unary_op_asin
        {
            __host__ __device__ float operator()(const float& x) const
            {
                return asin(x);
            }
        };

        struct unary_op_acos
        {
            __host__ __device__ float operator()(const float& x) const
            {
                return acos(x);
            }
        };

        struct unary_op_atan
        {
            __host__ __device__ float operator()(const float& x) const
            {
                return atan(x);
            }
        };

        struct unary_op_reciprocal
        {
            __host__ __device__ float operator()(const float& x) const
            {
                return 1.f / x;
            }
        };

        struct unary_op_tanh
        {
            __host__ __device__ float operator()(const float& x) const
            {
                return tanh(x);
            }
        };

        template <typename Op>
        __global__ void unary_op_kernel(int N, const float* bottom_data, float* top_data)
        {
            Op op;
            CUDA_KERNEL_LOOP(i, N)
            {
                top_data[i] = op(bottom_data[i]);
            }
        }

#define unary_op_gpu(op,count,bottom_data,top_data) unary_op_kernel<op><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bottom_data,top_data)

        template <class Dtype>
        void operation_unaryop<Dtype>::forward_gpu_f32(
            cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
            cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
            const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
            std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            const float* bottom_data = bottoms[0]->gpu_data();
            int count = bottoms[0]->count();
            float* top_data = tops[0]->mutable_gpu_data();
            if (op_type_ == Operation_ABS)
                unary_op_gpu(unary_op_abs, count, bottom_data, top_data);

            if (op_type_ == Operation_NEG)
                unary_op_gpu(unary_op_neg, count, bottom_data, top_data);

            if (op_type_ == Operation_FLOOR)
                unary_op_gpu(unary_op_floor, count, bottom_data, top_data);

            if (op_type_ == Operation_CEIL)
                unary_op_gpu(unary_op_ceil, count, bottom_data, top_data);

            if (op_type_ == Operation_SQUARE)
                unary_op_gpu(unary_op_square, count, bottom_data, top_data);

            if (op_type_ == Operation_SQRT)
                unary_op_gpu(unary_op_sqrt, count, bottom_data, top_data);

            if (op_type_ == Operation_RSQRT)
                unary_op_gpu(unary_op_rsqrt, count, bottom_data, top_data);

            if (op_type_ == Operation_EXP)
                unary_op_gpu(unary_op_exp, count, bottom_data, top_data);

            if (op_type_ == Operation_LOG)
                unary_op_gpu(unary_op_log, count, bottom_data, top_data);

            if (op_type_ == Operation_SIN)
                unary_op_gpu(unary_op_sin, count, bottom_data, top_data);

            if (op_type_ == Operation_COS)
                unary_op_gpu(unary_op_cos, count, bottom_data, top_data);

            if (op_type_ == Operation_TAN)
                unary_op_gpu(unary_op_tan, count, bottom_data, top_data);

            if (op_type_ == Operation_ASIN)
                unary_op_gpu(unary_op_asin, count, bottom_data, top_data);

            if (op_type_ == Operation_ACOS)
                unary_op_gpu(unary_op_acos, count, bottom_data, top_data);

            if (op_type_ == Operation_ATAN)
                unary_op_gpu(unary_op_atan, count, bottom_data, top_data);

            if (op_type_ == Operation_RECIPROCAL)
                unary_op_gpu(unary_op_reciprocal, count, bottom_data, top_data);

            if (op_type_ == Operation_TANH)
                unary_op_gpu(unary_op_tanh, count, bottom_data, top_data);
        }

#ifdef USE_CUDNN
        INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_unaryop);
#else
        INSTANTIATE_OPERATION_CUDA_FWDF32(operation_unaryop);
#endif

#endif
    }
}