#ifndef _OPERATION_EXPAND_HPP_
#define _OPERATION_EXPAND_HPP_

#include "operation.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        class operation_expand: public operation<Dtype>
        {
        public:
            operation_expand(const operation_param &param);
            virtual const char *type() { return this->params_.type_.c_str(); }
            virtual ~operation_expand() {}

        private:
            int w_;
            int h_;
            int c_;

        protected:
            void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops);
        };
    }
}
#endif