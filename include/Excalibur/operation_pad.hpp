#ifndef _OPERATION_PAD_HPP_
#define _OPERATION_PAD_HPP_
#include "operation.hpp"

namespace glasssix
{
    namespace excalibur
    { 
        template <class Dtype>
        class operation_pad : public operation<Dtype>
        {
        public:
            operation_pad(const operation_param &param);
            virtual const char *type() { return this->params_.type_.c_str(); }
            virtual ~operation_pad() {}

        protected:
            void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops);

        private:
            void copy_make_border_image(const std::shared_ptr<memory::tensor<float>> &bottoms, std::shared_ptr<memory::tensor<float>> &tops);

            enum pad_type { CONSTANT, REFLECT, EDGE };
            //top left bottom right
            std::vector<int> pads_{0, 0, 0, 0};
            float constant_value_ = 0;
            pad_type type_ = pad_type::CONSTANT;
        };
    }
}
#endif