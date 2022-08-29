#include "../../../include/Excalibur/operation_pad.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"
#include "../../../include/Excalibur/math_functions.hpp"

namespace glasssix
{
    namespace excalibur
    {
#ifdef USE_CUDA
        __global__ void constant_mode_fill_kernel_nchw(const int n, const float* src, int bottom_n_stride, int bottom_c_stride, int bottom_h_stride,
            float* dst, int top_n_stride, int top_c_stride, int top_h_stride, float constant_value, int pad_front, int pad_top, int pad_left)
        {
            CUDA_KERNEL_LOOP(index, n)
            {
                int offset = index / bottom_n_stride * top_n_stride;
                offset += pad_front * top_c_stride;
                offset += index % bottom_n_stride / bottom_c_stride * top_c_stride;
                offset += pad_top * top_h_stride;
                offset += index % bottom_n_stride % bottom_c_stride / bottom_h_stride * top_h_stride;
                offset += pad_left;
                offset += index % bottom_n_stride % bottom_c_stride % bottom_h_stride;

                dst[offset] = src[index];
            }
        }

        template <typename Dtype>
        void operation_pad<Dtype>::forward_gpu_f32(
            cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
            cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
            const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
            std::vector<std::shared_ptr<memory::tensor<float>>>& tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            //check pad num
            if (!(top_ || bottom_ || left_ || right_ || front_ || behind_))
            {
                tops[0] = bottoms[0];
                return;
            }
            if (bottoms[0]->order() == memory::NCHW)
            {
                tops[0].reset(new memory::tensor<float>(std::vector<int>{bottoms[0]->num(), bottoms[0]->channels() + front_ + behind_, bottoms[0]->height() + top_ + bottom_, bottoms[0]->width() + left_ + right_}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));

                if (mode_ == CONSTANT)
                {
                    math_functions::gpu_set(tops[0]->count(), constant_value_, tops[0]->mutable_gpu_data());
                    constant_mode_fill_kernel_nchw << <CUDA_GET_BLOCKS(bottoms[0]->count()), CUDA_NUM_THREADS >> > (bottoms[0]->count(),
                        bottoms[0]->gpu_data(), bottoms[0]->count(1, 4), bottoms[0]->count(2, 4), bottoms[0]->count(3, 4),
                        tops[0]->mutable_gpu_data(), tops[0]->count(1, 4), tops[0]->count(2, 4), tops[0]->count(3, 4),
                        constant_value_, front_, top_, left_);
                }
                else
                    NOT_IMPLEMENTED;
            }
            else
                NOT_IMPLEMENTED;
        }

#ifdef USE_CUDNN
        INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_pad);
#else
        INSTANTIATE_OPERATION_CUDA_FWDF32(operation_pad);
#endif

#endif // USE_CUDA

    }
}