#ifndef _OPERATION_LEAKYRELU_HPP_
#define _OPERATION_LEAKYRELU_HPP_
#include "operation.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        class operation_leakyrelu : public operation<Dtype>
        {
        public:
            operation_leakyrelu(const operation_param &param);
            virtual const char *type() { return this->params_.type_.c_str(); }
            virtual ~operation_leakyrelu() {}

        protected:
            void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops);
        
        private:
            float alpha_;
        };
    }
}
#endif