#include "../../../include/Excalibur/operation_binaryop.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
    namespace excalibur
    {
#ifdef USE_CUDA

        __global__ void binaryop_forward(int N, int optype, const float *bottom_data1, const float *bottom_data2, float *top_data)
        {
            if (optype == 0) // add
            {
                CUDA_KERNEL_LOOP(i, N)
                {
                    top_data[i] = bottom_data1[i] + bottom_data2[i];
                }
            }
            else if (optype == 1) // sub
            {
                CUDA_KERNEL_LOOP(i, N)
                {
                    top_data[i] = bottom_data1[i] - bottom_data2[i];
                }
            }
            else if (optype == 2) // mul
            {
                CUDA_KERNEL_LOOP(i, N)
                {
                    top_data[i] = bottom_data1[i] * bottom_data2[i];
                }
            }
            else if (optype == 3) // div
            {
                CUDA_KERNEL_LOOP(i, N)
                {
                    top_data[i] = bottom_data1[i] / bottom_data2[i];
                }
            }
            else
            {
                return;
            }
        }

        __global__ void binaryop_forward_with_scalar(int N, int optype, const float *bottom_data, float *top_data, float coef_)
        {
            if (optype == 0) // add
            {
                CUDA_KERNEL_LOOP(i, N)
                {
                    top_data[i] = bottom_data[i] + coef_;
                }
            }
            else if (optype == 1) // sub
            {
                CUDA_KERNEL_LOOP(i, N)
                {
                    top_data[i] = bottom_data[i] - coef_;
                }
            }
            else if (optype == 2) // mul
            {
                CUDA_KERNEL_LOOP(i, N)
                {
                    top_data[i] = bottom_data[i] * coef_;
                }
            }
            else if (optype == 3) // div
            {
                CUDA_KERNEL_LOOP(i, N)
                {
                    top_data[i] = bottom_data[i] / coef_;
                }
            }
            else
            {
                return;
            }
        }

        template <class Dtype>
        void operation_binaryop<Dtype>::forward_gpu_f32(
            cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
            cudnnHandle_t cudnn_handle,
#endif
            const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
            std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_GE(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            int count = bottoms[0]->count();
            tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            float *top_data = tops[0]->mutable_gpu_data();

            if (bottoms[0]->order() == memory::NCHW)
            {
                if (with_scalar_)
                {
                    binaryop_forward_with_scalar<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, op_type_, bottoms[0]->gpu_data(), top_data, coef_);
                }
                else
                {
                    binaryop_forward<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, op_type_, bottoms[0]->gpu_data(), bottoms[1]->gpu_data(), top_data);
                }
            }
            else
            {
                NOT_IMPLEMENTED;
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