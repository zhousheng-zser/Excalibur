#include "../../../include/Excalibur/operation_sigmoid.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
    namespace excalibur
    {
#ifdef USE_CUDA
        __global__ void SigmoidCUDAKernel(const int N, const float *X, float *Y)
        {
            CUDA_KERNEL_LOOP(i, N)
            {
#if __CUDA_ARCH__ >= 350
                Y[i] = 1.0f / (1.0f + expf(-__ldg(X + i)));
#else
                Y[i] = 1.0f / (1.0f + expf(-X[i]));
#endif
            }
        }

        template <typename Dtype>
        void operation_sigmoid<Dtype>::forward_gpu_f32(
            cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
            cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
            const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
            std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), this->params_.device_, bottoms[0]->order(), bottoms[0]->allocator()));
            int count = bottoms[0]->count();
            const float *input_data = bottoms[0]->gpu_data();
            float *output_data = tops[0]->mutable_gpu_data();

            if (bottoms[0]->order() == memory::NCHW)
            {
               SigmoidCUDAKernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, input_data, output_data);
            }
            else
            {
                NOT_IMPLEMENTED;
            }
            CUDA_POST_KERNEL_CHECK;
        }

#ifdef USE_CUDNN
        INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_sigmoid);
#else
        INSTANTIATE_OPERATION_CUDA_FWDF32(operation_sigmoid);
#endif

#endif // USE_CUDA

    }
}