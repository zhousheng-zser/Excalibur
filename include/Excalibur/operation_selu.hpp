#ifndef _OPERATION_SELU_H_
#define _OPERATION_SELU_H_

#include "operation.hpp"

namespace glasssix
{
	namespace excalibur
	{
        template <class Dtype>
        class operation_selu: public operation<Dtype>
        {
        public:
            operation_selu(const operation_param &param);
            virtual const char *type() { return this->params_.type_.c_str(); }
            virtual ~operation_selu() {}

        protected:
            void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops);

        private:
            float alpha_ = 1.67326;
            float gamma_ = 1.0507;
        };
    }
}

#endif