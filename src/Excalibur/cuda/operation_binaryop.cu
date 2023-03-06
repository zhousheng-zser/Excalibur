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
            CHECK_EQ(b1_shape[0], 1); // excalibur not support 4dims
            CHECK_EQ(b2_shape[0], 1); // could refactor it when support 4dims, so must check it now
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
        static void binary_op(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms, std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
        {
            Op op;
            std::shared_ptr<memory::tensor<float>> bottom_max;
            std::shared_ptr<memory::tensor<float>> bottom_min;
            bottom_assign(bottom_max, bottom_min, bottoms[0], bottoms[1]);

            int bmax_W = bottom_max->width();
            int bmax_H = bottom_max->height();
            int bmax_C = bottom_max->channels();
            std::vector<int> bmax_shape = bottom_max->data_shape();
            int bmax_dims = (bmax_W == 1 ? 0 : 1) + (bmax_H == 1 ? 0 : 1) + (bmax_C == 1 ? 0 : 1);
            bmax_dims = bmax_dims == 0 ? 1 : bmax_dims;
            const float* bmax_data = bottom_max->gpu_data();
            int c_step_max = bmax_W * bmax_H;

            int bmin_W = bottom_min->width();
            int bmin_H = bottom_min->height();
            int bmin_C = bottom_min->channels();
            std::vector<int> bmin_shape = bottom_min->data_shape();
            int bmin_dims = (bmin_W == 1 ? 0 : 1) + (bmin_H == 1 ? 0 : 1) + (bmin_C == 1 ? 0 : 1);
            bmin_dims = bmin_dims == 0 ? 1 : bmin_dims;
            const float* bmin_data = bottom_min->gpu_data();
            int c_step_min = bmin_W * bmin_H;

            tops[0].reset(new memory::tensor<float>(bottom_max->data_shape(), bottom_max->device(), bottom_max->order(), bottom_max->allocator()));
            int count = bottom_max->count();
            int count_min = bottom_min->count();
            std::vector<int> const_ones(4, 1);
            if (bmin_shape == const_ones)
            {
                binaryop_kernel_mod_num<Op> << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (count, 1, bmax_data, bmin_data, tops[0]->mutable_gpu_data());
                return;
            }
            else if (bmax_shape == bmin_shape)
            {
                binaryop_kernel_equal_dim<Op><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bmax_data, bmin_data, tops[0]->mutable_gpu_data());
                return;
            }
            else if (bmax_dims == 3)
            {
                if (bmin_dims == 3 && bmax_C != bmin_C && bmax_C > 3)
                {

                    tops[0].reset(new memory::tensor<float>(bottom_max->data_shape(), bottom_max->device(), bottom_max->order(), bottom_max->allocator()));
                    // M*batch(HW) OP m*batch(HW) -> m*n*batch (or n*m*batch) OP m*batch, generally m < n
                    // here bmin_C == m
                    const double m = static_cast<double>(bmin_C);
                    const double nd = static_cast<double>(bmax_C) / m;
                    if (ceil(nd) == floor(nd))
                    {
                        const int n = nd;
                        if (n >= m) {
                            /*
                            * m*n OP m
                            * e.g.: [4 * 24 * 160 * 160] OP [4 * 1 * 160 * 160]
                            * this 24=n 4=m
                            * [24 * B] * [1 * B] loop 4 times
                            */
                            for (int i = 0; i < m; ++i) {
                                float* top_ptr = tops[0]->mutable_gpu_data() + i * n * c_step_min;
                                const float* max_ptr = bmax_data + i * n * c_step_min;
                                const float* min_ptr = bmin_data + i * c_step_min;
                                binaryop_kernel_mod_num<Op><<<CUDA_GET_BLOCKS(n * c_step_min), CUDA_NUM_THREADS>>>(n * c_step_min, c_step_min, max_ptr, min_ptr, top_ptr);
                            }
                            return;
                        }
                        else {
                            // still not meet, maybe u can try:
                            //binaryop_kernel_mod_num<Op> << <CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS >> > (count, count_min, bmax_data, bmin_data, tops[0]->mutable_gpu_data());
                            //return;
                            NOT_IMPLEMENTED;
                        }
                    }
                }
                else if (bmin_dims == 2) {
                    //e.g for: [1 * 2 * 4 * 4] op [1 * 1 * 4 * 4]
                    CHECK_EQ(c_step_min, c_step_max);
                    tops[0].reset(new memory::tensor<float>(bottom_max->data_shape(), bottom_max->device(), bottom_max->order(), bottom_max->allocator()));
                    binaryop_kernel_mod_num<Op><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, c_step_max, bmax_data, bmin_data, tops[0]->mutable_gpu_data());
                    return;
                }
                else if (bmin_dims == 1 && bmin_C > 1)
                {
                    binaryop_kernel_div_num<Op><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, c_step_max, bmax_data, bmin_data, tops[0]->mutable_gpu_data());
                    return;
                }
            }
            else if (bmax_dims == 2)
            {
                if (bmin_dims == 1)
                {
                    binaryop_kernel_mod_num<Op><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bmax_W, bmax_data, bmin_data, tops[0]->mutable_gpu_data());
                    return;
                }
                else if (bmin_dims == 2 && bmax_shape != bmin_shape)
                {
                    //e.g for: [1 * 1 * 4 * 4] op [1 * 1 * 2 * 4]
                    NOT_IMPLEMENTED;
                }
            }
            else if (bmax_dims == 1)
            {
                if (bmin_dims == 1)
                {
                    if (bmin_W * bmin_H * bmin_C > 1)
                    {
                        binaryop_kernel_equal_dim<Op><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bmax_data, bmin_data, tops[0]->mutable_gpu_data());
                        return;
                    }
                }
                else if (bmin_dims == 3)
                {
                    if (bmax_W == 1 && bmax_H == 1 && bmax_C == bmin_C)
                    {
                        tops[0].reset(new memory::tensor<float>(bottom_min->data_shape(), bottom_min->device(), bottom_min->order(), bottom_min->allocator()));
                        count = bottom_min->count();
                        binaryop_kernel_div_num<Op><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, c_step_min, bmin_data, bmax_data, tops[0]->mutable_gpu_data());
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