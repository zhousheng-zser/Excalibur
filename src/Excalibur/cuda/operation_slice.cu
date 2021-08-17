#include "../../../include/Excalibur/operation_slice.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"

namespace glasssix
{
    namespace excalibur
    {
#ifdef USE_CUDA
        __global__ void slice_kernel(const int n, const float *in, float *out)
        {
            CUDA_KERNEL_LOOP(index, n)
            {
                out[index] = in[index];
            }
        }

        template <typename Dtype>
        void operation_slice<Dtype>::forward_gpu_f32(
            cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
            cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
            const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
            std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);

            std::vector<int> shape = bottoms[0]->data_shape();
            CHECK_GE(axis_, 0);
            CHECK_LT(axis_, shape.size());
            int input_dim = shape[axis_];
            CHECK_GE(start_, 0);
            CHECK_LE(end_, input_dim);

            int output_dim = end_ - start_;
            int inner_size = 1;
            for (int i = axis_ + 1; i < shape.size(); ++i)
            {
                inner_size *= shape[i];
            }
            int input_stride = input_dim * inner_size;
            std::vector<int> output_shape(shape);
            output_shape[axis_] = output_dim;
            tops[0].reset(new memory::tensor<float>(output_shape, this->params_.device_, bottoms[0]->order(), bottoms[0]->allocator()));
            int outer_size = 1;
            for (int i = 0; i < axis_; ++i)
            {
                outer_size *= shape[i];
            }
            int output_stride = output_dim * inner_size;
            int offset = start_ * inner_size;
            const float *input_data = bottoms[0]->gpu_data();
            float *output_data = tops[0]->mutable_gpu_data();

            if (bottoms[0]->order() == memory::NCHW)
            {
                for (int i = 0; i < outer_size; ++i)
                {
                    const float *input_base = input_data + i * input_stride + offset;
                    float *output_base = output_data + i * output_stride;
                    slice_kernel<<<CUDA_GET_BLOCKS(output_stride), CUDA_NUM_THREADS>>>(output_stride, input_base, output_base);
                }
            }
            else
            {
                NOT_IMPLEMENTED;
            }
            CUDA_POST_KERNEL_CHECK;
        }

#ifdef USE_CUDNN
        INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_slice);
#else
        INSTANTIATE_OPERATION_CUDA_FWDF32(operation_slice);
#endif

#endif // USE_CUDA

    }
}