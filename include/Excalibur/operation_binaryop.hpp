#ifndef _OPERATION_BINARYOP_HPP_
#define _OPERATION_BINARYOP_HPP_

#include "operation.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        class operation_binaryop : public operation<Dtype>
        {
        public:
            operation_binaryop(const operation_param &param);
            virtual const char *type() { return this->params_.type_.c_str(); }
            virtual ~operation_binaryop() {}

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
            enum OperationType
            {
                Operation_ADD = 0,
                Operation_SUB = 1,
                Operation_MUL = 2,
                Operation_DIV = 3,
                Operation_MAX = 4,
                Operation_MIN = 5,
                Operation_POW = 6,
                Operation_RSUB = 7,
                Operation_RDIV = 8
            };
            OperationType op_type_;
            int with_scalar_ = 0;
            float coef_;
        };
    }
}
#endif