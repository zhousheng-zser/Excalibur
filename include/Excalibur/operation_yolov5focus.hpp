#ifndef _OPERATION_YOLOV5FOCUS_H_
#define _OPERATION_YOLOV5FOCUS_H_

#include "operation.hpp"
 
namespace glasssix
{
	namespace excalibur
	{
        template <class Dtype>
        class operation_yolov5focus: public operation<Dtype>
        {
        public:
            operation_yolov5focus(const operation_param &param);
            virtual const char *type() { return this->params_.type_.c_str(); }
            virtual ~operation_yolov5focus() {}

        protected:
            void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops);
        };
    }
}
#endif