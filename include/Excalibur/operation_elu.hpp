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
        
        private:
            float alpha_;
        };
    }
}

#endif