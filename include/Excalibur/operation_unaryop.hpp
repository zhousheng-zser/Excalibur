#ifndef _OPERATION_UNARYOP_HPP_
#define _OPERATION_UNARYOP_HPP_
#include <climits>
#include "operation.hpp"

namespace glasssix
{
    namespace excalibur
    {
        template <class Dtype>
        class operation_unaryop : public operation<Dtype>
        {
        public:
            operation_unaryop(const operation_param &param);
            virtual const char *type() { return this->params_.type_.c_str(); }
            virtual ~operation_unaryop() {}

        protected:
            virtual void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>> &bottoms, std::vector<std::shared_ptr<memory::tensor<float>>> &tops);

        private:
            enum OperationType
            {
                Operation_ABS = 0,
                Operation_NEG = 1,
                Operation_FLOOR = 2,
                Operation_CEIL = 3,
                Operation_SQUARE = 4,
                Operation_SQRT = 5,
                Operation_RSQRT = 6,
                Operation_EXP = 7,
                Operation_LOG = 8,
                Operation_SIN = 9,
                Operation_COS = 10,
                Operation_TAN = 11,
                Operation_ASIN = 12,
                Operation_ACOS = 13,
                Operation_ATAN = 14,
                Operation_RECIPROCAL = 15,
                Operation_TANH = 16
            };
            // operator type
            int op_type_;
        };
    }
}
#endif