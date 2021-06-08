#include "../../../include/Excalibur/operation_leakyrelu.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
    namespace excalibur
    {
#ifdef USE_CUDA

        __global__ void leakyrelu_forward(int N, const float *bottom_data, float *top_data, float alpha_)
        {
            CUDA_KERNEL_LOOP(i, N)
            {
                top_data[i] = bottom_data[i] < 0 ? alpha_ * bottom_data[i] : bottom_data[i];
            }
        }

        template <class Dtype>
        void operation_leakyrelu<Dtype>::forward_gpu_f32(
#ifdef USE_CUDA
            cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
            cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
            const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
            std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
            {
                CHECK_EQ(bottoms.size(), 1);
                CHECK_EQ(tops.size(), 1);

                tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
                int count = bottoms[0]->count();
                const float *bottom_data = bottoms[0]->gpu_data();
                float *top_data = tops[0]->mutable_gpu_data();

                leakyrelu_forward<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bottom_data, top_data, alpha_);
            }

#ifdef USE_CUDNN
    INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_leakyrelu);
#else
    INSTANTIATE_OPERATION_CUDA_FWDF32(operation_leakyrelu);
#endif

#endif
    }
}