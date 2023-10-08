#ifndef _OPERATION_INTERP_HPP_
#define _OPERATION_INTERP_HPP_

#include "operation.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        class operation_interp : public operation<Dtype>
        {
        public:
            operation_interp(const operation_param &param);
            virtual const char *type() { return this->params_.type_.c_str(); }
            virtual ~operation_interp() {}

        protected:
            virtual void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops);
            virtual void forward_gpu_f32(
#ifdef USE_CUDA
                cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
                cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
                const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms,
                std::vector<std::shared_ptr<memory::tensor<float>>> &tops);
        private:
            // param
            int resize_type_; //1=nearest  2=bilinear  3=bicubic
            float width_scale_ = 1.0f;
            float height_scale_ = 1.0f;
            int output_width_ = 0;
            int output_height_ = 0;
            int dynamic_target_size_;
            int align_corner_ = 0;
        };
    }
}
#endif