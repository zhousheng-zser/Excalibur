#include "../../../include/Excalibur/operation_binaryop.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

#include <cfloat>

namespace glasssix
{
    namespace excalibur
    {
#ifdef USE_CUDA
        struct binary_op_add
        {
            __host__ __device__ float operator()(const float &x, const float &y) const
            {
                return x + y;
            }
        };

        struct binary_op_sub
        {
            __host__ __device__ float operator()(const float &x, const float &y) const
            {
                return x - y;
            }
        };

        struct binary_op_mul
        {
            __host__ __device__ float operator()(const float &x, const float &y) const
            {
                return x * y;
            }
        };

        struct binary_op_div
        {
            __host__ __device__ float operator()(const float &x, const float &y) const
            {
                return x / y;
            }
        };

        struct binary_op_max
        {
            __host__ __device__ float operator()(const float &x, const float &y) const
            {
                return std::max(x, y);
            }
        };

        struct binary_op_min
        {
            __host__ __device__ float operator()(const float &x, const float &y) const
            {
                return std::min(x, y);
            }
        };

        struct binary_op_pow
        {
            __host__ __device__ float operator()(const float &x, const float &y) const
            {
                return (float)pow(x, y);
            }
        };

        struct binary_op_rsub
        {
            __host__ __device__ float operator()(const float &x, const float &y) const
            {
                return y - x;
            }
        };

        struct binary_op_rdiv
        {
            __host__ __device__ float operator()(const float &x, const float &y) const
            {
                return y / x;
            }
        };

        template <typename Op>
        __global__ void binaryop_kernel_with_scalar(int N, const float *bottom_data, float *top_data, float coef)
        {
            Op op;
            CUDA_KERNEL_LOOP(i, N)
            {
                top_data[i] = op(bottom_data[i], coef);
            }
        }

        template <typename Op>
        __global__ void binaryop_kernel_equal_dim(int N, const float *b1, const float *b2, float *top_data)
        {
            Op op;
            CUDA_KERNEL_LOOP(i, N)
            {
                top_data[i] = op(b1[i], b2[i]);
            }
        }

        template <typename Op>
        __global__ void binaryop_kernel_mod_num(int N, int mod_num, const float *b1, const float *b2, float *top_data)
        {
            Op op;
            CUDA_KERNEL_LOOP(i, N)
            {
                top_data[i] = op(b1[i], b2[i % mod_num]);
            }
        }

        template <typename Op>
        __global__ void binaryop_kernel_div_num(int N, int div_num, const float *b1, const float *b2, float *top_data)
        {
            Op op;
            CUDA_KERNEL_LOOP(i, N)
            {
                top_data[i] = op(b1[i], b2[i / div_num]);
            }
        }

        template <typename Op>
        static void binary_op(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            Op op;
            int bottom_w1 = bottoms[0]->width();
            int bottom_h1 = bottoms[0]->height();
            int bottom_c1 = bottoms[0]->channels();
            int b1_dims = (bottom_w1 == 1 ? 0 : 1) + (bottom_h1 == 1 ? 0 : 1) + (bottom_c1 == 1 ? 0 : 1);
            b1_dims = b1_dims == 0 ? 1 : b1_dims;
            const float *b1 = bottoms[0]->gpu_data();
            int c_step1 = bottom_w1 * bottom_h1;

            int bottom_w2 = bottoms[1]->width();
            int bottom_h2 = bottoms[1]->height();
            int bottom_c2 = bottoms[1]->channels();
            int b2_dims = (bottom_w2 == 1 ? 0 : 1) + (bottom_h2 == 1 ? 0 : 1) + (bottom_c2 == 1 ? 0 : 1);
            b2_dims = b2_dims == 0 ? 1 : b2_dims;
            const float *b2 = bottoms[1]->gpu_data();
            int c_step2 = bottom_w2 * bottom_h2;

            tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            int count = bottoms[0]->count();
            if (b1_dims == 3)
            {
                if (b2_dims == 3)
                {
                    binaryop_kernel_equal_dim<Op><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, b1, b2, tops[0]->mutable_gpu_data());
                    return;
                }
                else if (b2_dims == 1 && bottom_c2 > 1)
                {
                    binaryop_kernel_div_num<Op><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, c_step1, b1, b2, tops[0]->mutable_gpu_data());
                    return;
                }
            }
            else if (b1_dims == 2)
            {
                if (b2_dims == 1)
                {
                    binaryop_kernel_mod_num<Op><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bottom_w1, b1, b2, tops[0]->mutable_gpu_data());
                    return;
                }
            }
            else if (b1_dims == 1)
            {
                if (b2_dims == 1)
                {
                    if (bottom_w2 * bottom_h2 * bottom_c2 > 1)
                    {
                        binaryop_kernel_equal_dim<Op><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, b1, b2, tops[0]->mutable_gpu_data());
                        return;
                    }
                    else if (bottom_w2 * bottom_h2 * bottom_c2 == 1)
                    {
                        binaryop_kernel_mod_num<Op><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, 1, b1, b2, tops[0]->mutable_gpu_data());
                        return;
                    }
                }
                else if (b2_dims == 3)
                {
                    if (bottom_w1 == 1 && bottom_h1 == 1 && bottom_c1 == bottom_c2)
                    {
                        tops[0].reset(new memory::tensor<float>(bottoms[1]->data_shape(), bottoms[1]->device(), bottoms[1]->order(), bottoms[1]->allocator()));
                        count = bottoms[1]->count();
                        binaryop_kernel_div_num<Op><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, c_step2, b2, b1, tops[0]->mutable_gpu_data());
                        return;
                    }
                }
            }
            NOT_IMPLEMENTED;
        }

        template <class Dtype>
        void operation_binaryop<Dtype>::forward_gpu_f32(
#ifdef USE_CUDA
            cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
            cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
            const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
            std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_GE(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);
            int numA = bottoms[0]->num();
            const float *bottomA_data = bottoms[0]->gpu_data();
            std::vector<int> bottomA_shape = bottoms[0]->data_shape();

            if (with_scalar_)
            {
                tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
                float *top_data = tops[0]->mutable_gpu_data();
                if (op_type_ == Operation_ADD)
                {
                    binaryop_kernel_with_scalar<binary_op_add><<<CUDA_GET_BLOCKS(numA), CUDA_NUM_THREADS>>>(numA, bottomA_data, top_data, coef_);
                }
                else if (op_type_ == Operation_SUB)
                {
                    binaryop_kernel_with_scalar<binary_op_sub><<<CUDA_GET_BLOCKS(numA), CUDA_NUM_THREADS>>>(numA, bottomA_data, top_data, coef_);
                }
                else if (op_type_ == Operation_MUL)
                {
                    binaryop_kernel_with_scalar<binary_op_mul><<<CUDA_GET_BLOCKS(numA), CUDA_NUM_THREADS>>>(numA, bottomA_data, top_data, coef_);
                }
                else if (op_type_ == Operation_DIV)
                {
                    binaryop_kernel_with_scalar<binary_op_div><<<CUDA_GET_BLOCKS(numA), CUDA_NUM_THREADS>>>(numA, bottomA_data, top_data, coef_);
                }
            }
            else
            {
                if (op_type_ == Operation_ADD)
                {
                    binary_op<binary_op_add>(bottoms, tops);
                }
                else if (op_type_ == Operation_SUB)
                {
                    binary_op<binary_op_sub>(bottoms, tops);
                }
                else if (op_type_ == Operation_MUL)
                {
                    binary_op<binary_op_mul>(bottoms, tops);
                }
                else if (op_type_ == Operation_DIV)
                {
                    binary_op<binary_op_div>(bottoms, tops);
                }
            }
        }

#ifdef USE_CUDNN
        INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_binaryop);
#else
        INSTANTIATE_OPERATION_CUDA_FWDF32(operation_binaryop);
#endif

#endif
    }
}