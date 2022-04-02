#ifndef _OPERATION_REDUCTION_HPP_
#define _OPERATION_REDUCTION_HPP_
#include <climits>
#include "operation.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        class operation_reduction : public operation<Dtype>
        {
        public:
            operation_reduction(const operation_param &param);
            virtual const char *type() { return this->params_.type_.c_str(); }
            virtual ~operation_reduction() {}

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
            enum ReductionOp
            {
                ReductionOp_SUM = 0,
                ReductionOp_ASUM = 1,
                ReductionOp_SUMSQ = 2,
                ReductionOp_MEAN = 3,
                ReductionOp_MAX = 4,
                ReductionOp_MIN = 5,
                ReductionOp_PROD = 6,
                ReductionOp_L1 = 7,
                ReductionOp_L2 = 8,
                ReductionOp_LogSum = 9,
                ReductionOp_LogSumExp = 10
            };
            // operator type
            int operation_ = 0;
            int axes_ = 0;
            int keepdims_ = 0;
        };
    }
}
#endif