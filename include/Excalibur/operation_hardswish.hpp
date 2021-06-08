#ifndef _OPERATION_HARDSWISH_HPP_
#define _OPERATION_HARDSWISH_HPP_

#include "operation.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        class operation_hardswish : public operation<Dtype>
        {
        public:
            operation_hardswish(const operation_param &param);
            virtual const char *type() const { return this->params_.type_.c_str(); }
            virtual ~operation_hardswish() {}

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
            float threshold_;
            float scale_;
            float offset_;
        };
    }
}
#endif