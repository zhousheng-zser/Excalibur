#include "../../../include/Excalibur/operation_hardswish.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
    namespace excalibur
    {
#ifdef USE_CUDA

        __global__ void hardswish_forward(int N, const float *bottom_data, float *top_data, float offset_, float threshold_, float scale_)
        {
            CUDA_KERNEL_LOOP(i, N)
            {
                top_data[i] = bottom_data[i] * (bottom_data[i] + offset_ < 0 ? 0 : bottom_data[i] + offset_ > threshold_ ? threshold_ : bottom_data[i] + offset_) / scale_;
            }
        }

        template<class Dtype>
        void operation_hardswish<Dtype>::forward_gpu_f32(
            cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
            cudnnHandle_t cudnn_handle,
#endif
            const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
            std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            int count = bottoms[0]->count();
            const float *bottom_data = bottoms[0]->gpu_data();
            tops[0].reset(new memory::tensor<float>(bottoms[0]->data_shape(), bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            float *top_data = tops[0]->mutable_gpu_data();
            
            hardswish_forward<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bottom_data, top_data, offset_, threshold_, scale_);
        }

#ifdef USE_CUDNN
    INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_hardswish);
#else
    INSTANTIATE_OPERATION_CUDA_FWDF32(operation_hardswish);
#endif

#endif
    }
}