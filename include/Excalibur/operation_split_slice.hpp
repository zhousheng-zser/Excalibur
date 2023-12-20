#ifndef _OPERATION_SPLIT_SLICE_HPP_
#define _OPERATION_SPLIT_SLICE_HPP_
#include <climits>
#include "operation.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        class operation_split_slice : public operation<Dtype>
        {
        public:
            operation_split_slice(const operation_param& param);
            virtual const char* type() { return this->params_.type_.c_str(); }
            virtual ~operation_split_slice() {}

        protected:
            virtual void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms, std::vector<std::shared_ptr<memory::tensor<float>>>& tops);
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
            // order: c h w
            size_t output_size_ = 0;
            int axis_ = 1;
            std::vector<int> split_;
        };
    }
}
#endif