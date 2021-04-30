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

        private:
            float threshold_ = 6.0;
            float scale_ = 6.0;
            float offset_ = 3.0;
        };
    }
}
#endif