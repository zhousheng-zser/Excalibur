#include "../../../include/Excalibur/operation_gemm.hpp"
#include "../../../include/Excalibur/math_functions.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

#ifdef USE_CUDA
#include <cuda_fp16.hpp>
#ifdef USE_CUDNN
#include <cudnn.h>
#endif // USE_CUDNN
#endif // USE_CUDA

namespace glasssix
{
    namespace excalibur
    {
#ifdef USE_CUDA
        template <typename Dtype>
        void operation_gemm<Dtype>::forward_gpu_f32(
            cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
            cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
            const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
            std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
        {
            CHECK_EQ(bottoms.size(), 2);
            CHECK_EQ(tops.size(), 1);

            int num = bottoms[0]->num();

            int M = bottoms[0]->height();
            int K = bottoms[0]->width();
            int N = bottoms[1]->width();

            tops[0].reset(new memory::tensor<float>(std::vector<int>{num, 1, M, N}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));

            const float* bottom_A_data = bottoms[0]->gpu_data();
            const float* bottom_B_data = bottoms[1]->gpu_data();
            float* top_data = tops[0]->mutable_gpu_data();

            math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, bottom_A_data, bottom_B_data, 0.0f, top_data);
        }

        template <typename Dtype>
        void operation_gemm<Dtype>::forward_gpu_f16(
            cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
            cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
            const std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& bottoms,
            std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& tops)
        {
            CHECK_EQ(bottoms.size(), 2);
            CHECK_EQ(tops.size(), 1);

            int num = bottoms[0]->num();

            int M = bottoms[0]->height();
            int K = bottoms[0]->width();
            int N = bottoms[1]->width();

            tops[0].reset(new memory::tensor<unsigned short>(std::vector<int>{num, 1, M, N}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));

            const unsigned short* bottom_A_data = bottoms[0]->gpu_data();
            const unsigned short* bottom_B_data = bottoms[1]->gpu_data();
            unsigned short* top_data = tops[0]->mutable_gpu_data();

            //math_functions::gpu_gemmEx(cublas_handle, CblasNoTrans, CblasNoTrans, this->output_channel_,
            //    this->output_spatial_dim_, this->kernel_dim_, float32_to_float16(1.0f),
            //    weights, col_buff, float32_to_float16(0.0f), output);
            math_functions::gpu_gemmEx(cublas_handle_, CblasNoTrans, CblasNoTrans, M, N, K, float32_to_float16(1.0f), bottom_A_data, bottom_B_data, float32_to_float16(0.0f), top_data);
        }

#ifdef USE_CUDNN
        INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_gemm);
        INSTANTIATE_OPERATION_CUDNN_FWDF16(operation_gemm);
#else
        INSTANTIATE_OPERATION_CUDA_FWDF32(operation_gemm);
        INSTANTIATE_OPERATION_CUDA_FWDF16(operation_gemm);
#endif

#endif //!USE_CUDA
    }
}