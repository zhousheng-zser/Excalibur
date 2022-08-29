#ifndef _OPERATION_PAD_HPP_
#define _OPERATION_PAD_HPP_
#include "operation.hpp"

namespace glasssix
{
    namespace excalibur
    { 
        template <class Dtype>
        class operation_pad : public operation<Dtype>
        {
        public:
            operation_pad(const operation_param &param);
            virtual const char *type() { return this->params_.type_.c_str(); }
            virtual ~operation_pad() {}

        protected:
            virtual void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, 
                std::vector<std::shared_ptr<memory::tensor<float>>> &tops);
            virtual void forward_gpu_f32(
#ifdef USE_CUDA
                cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
                cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
                const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
                std::vector<std::shared_ptr<memory::tensor<float>>>& tops);
        private:
            void copy_make_border_image(const std::shared_ptr<memory::tensor<float>> &bottoms, std::shared_ptr<memory::tensor<float>> &tops);

            enum pad_mode { CONSTANT = 0, EDGE, REFLECT };
            //top left bottom right
            int top_ = 0;
            int bottom_ = 0;
            int left_ = 0;
            int right_ = 0;
            int mode_ = pad_mode::CONSTANT;
            float constant_value_ = 0.f;
            int front_ = 0;
            int behind_ = 0;
        };
    }
}
#endif