#ifndef _OPERATION_TRANSPOSE_HPP_
#define _OPERATION_TRANSPOSE_HPP_

#include "operation.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        class operation_transpose : public operation<Dtype>
        {
        public:
            operation_transpose(const operation_param &param);
            virtual const char *type() { return this->params_.type_.c_str(); }
            virtual ~operation_transpose() {}

        protected:
            void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops);

        private:
            // w h c
            std::vector<int> perms_{0, 1, 2};
        };
    }
}
#endif