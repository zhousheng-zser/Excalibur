#ifndef _OPERATION_CLIP_HPP_
#define _OPERATION_CLIP_HPP_

#include "operation.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        class operation_clip : public operation<Dtype>
        {
        public:
            operation_clip(const operation_param &param);
            virtual const char *type() const { return this->params_.type_.c_str(); }
            virtual ~operation_clip() {}

        protected:
            virtual void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops);
#ifdef USE_CUDA
            virtual void forward_gpu_f32(
                cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
                cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
                const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
                std::vector<std::shared_ptr<memory::tensor<float>>>& tops);
#endif //!USE_CUDA

        private:
            float min_;
            float max_;
        };
    }
}

#endif