#ifndef _OPERATION_ELU_H_
#define _OPERATION_ELU_H_

#include "Excalibur/operation.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        class operation_elu : public operation<Dtype>
        {
        public:
            operation_elu(const operation_param &param);
            virtual const char *type() const { return this->params_.type_.c_str(); }
            virtual ~operation_elu() {}
        
        protected:
            void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops);

            virtual void forward_gpu_f32(
#ifdef USE_CUDA
                cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
                cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
                const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
                std::vector<std::shared_ptr<memory::tensor<float>>>& tops);

#ifdef USE_CUDA
            virtual void forward_gpu_f16(
                cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
                cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
                const std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& bottoms,
                std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& tops);
#endif //!USE_CUDA
        
        private:
            float alpha_;
        };
    }
}

#endif