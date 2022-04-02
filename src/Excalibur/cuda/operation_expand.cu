#include "../../../include/Excalibur/operation_expand.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
    namespace excalibur
    {
#ifdef USE_CUDA

        template<typename T>
        __global__ void expand_kernel(int N, const T* value, T* top_data)
        {
            CUDA_KERNEL_LOOP(i, N)
            {
                top_data[i] = *value;
            }
        }

        template <class Dtype>
        void operation_expand<Dtype>::forward_gpu_f32(
            cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
            cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
            const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
            std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            // int bottom_w = bottoms[0]->width();
            // int bottom_h = bottoms[0]->height();
            // int bottom_c = bottoms[0]->channels();
            int num = bottoms[0]->num();
            // int bottom_cstep = bottom_w * bottom_h;

            if (bottoms[0]->order() == memory::NCHW)
                tops[0].reset(new memory::tensor<float>(std::vector<int>{num, c_, h_, w_}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            else
                tops[0].reset(new memory::tensor<float>(std::vector<int>{num, h_, w_, c_}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));

            for (int n = 0; n < num; ++n)
            {
                const float* bottom_data = bottoms[0]->gpu_data() + bottoms[0]->offset(n);
                float* top_data = tops[0]->mutable_gpu_data() + tops[0]->offset(n);

                if (bottoms[n]->count() == 1) // input 1
                {
                    expand_kernel << <CUDA_GET_BLOCKS(tops[n]->count()), CUDA_NUM_THREADS >> > (tops[n]->count(), bottom_data, top_data);
                }
            }
        }

#ifdef USE_CUDNN
        INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_expand);
#else
        INSTANTIATE_OPERATION_CUDA_FWDF32(operation_expand);
#endif

#endif
    }
}