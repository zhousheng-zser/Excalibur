#include "../../../include/Excalibur/operation_interp.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

#include <cfloat>

namespace glasssix
{
    namespace excalibur
    {
#ifdef USE_CUDA
        __global__ void interp_forward_nearest(int N, const float *bottom_data, float *top_data, float scalew, float scaleh, int iw, int ih, int ow, int oh)
        {
            CUDA_KERNEL_LOOP(index, N)
            {
                int x = index % ow;
                int tmp = index / ow;
                int y = tmp % oh;
                int z = tmp / oh;
                int ix = min(max(0, (int)(x * scalew)), iw - 1);
                int iy = min(max(0, (int)(y * scaleh)), ih - 1);
                top_data[index] = bottom_data[z * ih * iw + iy * iw + ix];
            }
        }

        template <class Dtype>
        void operation_interp<Dtype>::forward_gpu_f32(
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

            int num = bottoms[0]->num();
            int bottom_w = bottoms[0]->width();
            int bottom_h = bottoms[0]->height();
            int bottom_c = bottoms[0]->channels();
            int outw = output_width_;
            int outh = output_height_;
            if (output_width_ == 0 || output_height_ == 0)
            {
                outw = static_cast<int>(bottom_w * width_scale_);
                outh = static_cast<int>(bottom_h * height_scale_);
            }
            tops[0].reset(new memory::tensor<float>(std::vector<int>{num, bottom_c, outh, outw}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
            int count = tops[0]->count();
            const float *bottom_data = bottoms[0]->gpu_data();
            float *top_data = tops[0]->mutable_gpu_data();
            const float hs = outh ? bottom_h / (float)outh : 1.f / height_scale_;
            const float ws = outw ? bottom_w / (float)outw : 1.f / width_scale_;

            interp_forward_nearest<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bottom_data, top_data, ws, hs, bottom_w, bottom_h, outw, outh);
        }

#ifdef USE_CUDNN
        INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_interp);
#else
        INSTANTIATE_OPERATION_CUDA_FWDF32(operation_interp);
#endif

#endif
    }
}