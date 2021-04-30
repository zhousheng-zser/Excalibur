#ifndef _OPERATION_CLIP_H_
#define _OPERATION_CLIP_H_

#include "operation.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        class operation_clip : public operation<Dtype>
        {
        public:
            operation_clip(const operation_param &param);
            virtual const char *type() const { return this->params_.type_.c_str(); }
            virtual ~operation_clip() {}

        protected:
            void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops);
        
        private:
            float min_;
            float max_;
        };
    }
}

#endif