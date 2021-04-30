#ifndef _OPERATION_SCATTERND_H_
#define _OPERATION_SCATTERND_H_

#include "operation.hpp"

namespace glasssix
{
	namespace excalibur
	{
        template <class Dtype>
        class operation_scatternd: public operation<Dtype>
        {
        public:
            operation_scatternd(const operation_param &param);
            virtual const char *type() { return this->params_.type_.c_str(); }
            virtual ~operation_scatternd() {}

        protected:
            void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops);

        private:
            std::vector<int> indices_;
        };
    }
}
#endif