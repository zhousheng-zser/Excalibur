#ifndef _OPERATION_HARDSIGMOID_HPP_
#define _OPERATION_HARDSIGMOID_HPP_
#include "operation.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        class operation_hardsigmoid : public operation<Dtype>
        {
        public:
            operation_hardsigmoid(const operation_param &param);
            virtual const char *type() { return this->params_.type_.c_str(); }
            virtual ~operation_hardsigmoid() {}

        protected:
            void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops);

        private:
            float alpha_ = 0.2;
            float beta_ = 0.5;
        };
    }
}
#endif