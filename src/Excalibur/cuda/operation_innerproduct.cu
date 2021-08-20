#include "../../../include/Excalibur/operation_innerproduct.hpp"
#include "../../../include/Excalibur/operation_reflector.hpp"
#include "../../../include/Excalibur/math_functions.hpp"
#include "../../../include/Excalibur/im2col.hpp"

namespace glasssix
{
    namespace excalibur
    {
#ifdef USE_CUDA

        template <typename Dtype>
        void operation_innerproduct<Dtype>::forward_gpu_f32(
            cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
            cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
            const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
            std::vector<std::shared_ptr<memory::tensor<float>>> &tops)
        {
            CHECK_EQ(bottoms.size(), 1);
            CHECK_EQ(tops.size(), 1);
            int m = bottoms[0]->num();
            int n = num_output_;
            int k = bottoms[0]->count(1, 4);
            memory::orderType order = bottoms[0]->order();
            if (bias_term_)
            {
                bias_multiplier_.reset(new memory::tensor<float>(std::vector<int>{m}, this->params_.device_, bottoms[0]->order(), bottoms[0]->allocator()));
                math_functions::gpu_set(m, 1.0f, bias_multiplier_->mutable_gpu_data());
            }
            const float *bottom_data = bottoms[0]->gpu_data();
            const float *weight = this->weights_f32_[0]->gpu_data();
            const int width = bottoms[0]->width();
            const int height = bottoms[0]->height();
            const int channels = bottoms[0]->channels();
            if (order == memory::NCHW)
            {
                int dims = (width == 1 ? 0 : 1) + (height == 1 ? 0 : 1) + (channels == 1 ? 0 : 1);
                if (dims == 2)
                {
                    // bottom->dim = 3
                    tops[0].reset(new memory::tensor<float>(std::vector<int>{m, 1, height, n}, bottoms[0]->device(), bottoms[0]->order(), bottoms[0]->allocator()));
                    float *top_data = tops[0]->mutable_gpu_data();
                    m = height;
                    k = width;
                    math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasTrans, m, n, k, 1.0f,
                                              bottom_data, weight, 0.0f, top_data);
                }
                else
                {
                    tops[0].reset(new memory::tensor<float>(std::vector<int>{m, n, 1, 1}, this->params_.device_, order, bottoms[0]->allocator()));
                    float *top_data = tops[0]->mutable_gpu_data();
                    math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasTrans, m, n, k, 1.0f,
                                              bottom_data, weight, 0.0f, top_data);
                    if (bias_term_)
                    {
                        math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, m, n, 1,
                                                  1.0f, bias_multiplier_->gpu_data(), this->weights_f32_[1]->gpu_data(), 1.0f, top_data);
                    }
                }
            }
            else if (order == memory::NHWC)
            {
                tops[0].reset(new memory::tensor<float>(std::vector<int>{m, 1, 1, n}, this->params_.device_, order, bottoms[0]->allocator()));
                //
                const float *bottom_data = bottoms[0]->gpu_data();
                float *top_data = tops[0]->mutable_gpu_data();
                const float *weight = this->weights_f32_[0]->gpu_data();

                if (bottoms[0]->height() != 1 || bottoms[0]->width() != 1)
                {
                    std::shared_ptr<memory::tensor<float>> col_buff;
                    col_buff.reset(new memory::tensor<float>(bottoms[0]->data_shape(), this->params_.device_, order, bottoms[0]->allocator()));
                    float *col_buff_data = col_buff->mutable_gpu_data();

                    im2col_gpu<float>(bottom_data, bottoms[0]->channels(), bottoms[0]->height(), bottoms[0]->width(), 1,
                                      1, 0, 0, 1, 1, 1, 1, col_buff_data, order, m);
                    math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasTrans, m, n, k, 1.0f,
                                              col_buff_data, weight, 0.0f, top_data);
                }
                else
                {
                    math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasTrans, m, n, k, 1.0f,
                                              bottom_data, weight, 0.0f, top_data);
                }
                //

                if (bias_term_)
                {
                    math_functions::gpu_sgemm(cublas_handle_, CblasNoTrans, CblasNoTrans, m, n, 1,
                                              1.0f, bias_multiplier_->gpu_data(), this->weights_f32_[1]->gpu_data(), 1.0f, top_data);
                }
            }
            else
            {
                NOT_IMPLEMENTED;
            }
        }

#ifdef USE_CUDNN
        INSTANTIATE_OPERATION_CUDNN_FWDF32(operation_innerproduct);
#else
        INSTANTIATE_OPERATION_CUDA_FWDF32(operation_innerproduct);
#endif

#endif
    }
}