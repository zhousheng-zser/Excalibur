#ifndef _OPERATION_SLICE_HPP_
#define _OPERATION_SLICE_HPP_
#include <climits>
#include "operation.hpp"
 
namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        class operation_slice : public operation<Dtype>
        {
        public:
            operation_slice(const operation_param &param);
            virtual const char *type() { return this->params_.type_.c_str(); }
            virtual ~operation_slice() {}

        protected:
            virtual void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops);
        
        private:
            // order: w h c
            std::vector<int> starts_;
            std::vector<int> ends_;
            std::vector<int> steps_;
        };
    }
}
#endif