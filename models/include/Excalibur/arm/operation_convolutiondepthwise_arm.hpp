#pragma once
#ifndef _OPERATION_CONVOLUTIONDEPTHWISE_ARM_HPP_
#define _OPERATION_CONVOLUTIONDEPTHWISE_ARM_HPP_

#include "../operation_general_conv.hpp"
#include "../operation_convolutiondepthwise.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		class operation_convolutiondepthwise_arm : public operation_general_conv<Dtype>
		{
		public:
			operation_convolutiondepthwise_arm(const operation_param& param);
			virtual ~operation_convolutiondepthwise_arm();

			virtual int init_weights();

			virtual int init_weights(FILE *fp);

		protected:
			virtual void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
				std::vector<std::shared_ptr<memory::tensor<float>>>& tops);

			virtual void forward_cpu_i8(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
				std::vector<std::shared_ptr<memory::tensor<float>>>& tops);

		private:
			void convdw3x3s1_neon(const std::shared_ptr < memory::tensor<float>>& bottom,
				std::shared_ptr < memory::tensor<float>>& top);
			void convdw3x3s2_neon(const std::shared_ptr < memory::tensor<float>>& bottom,
				std::shared_ptr < memory::tensor<float>>& top);

			void convdw3x3s1_int8_neon(const std::shared_ptr<memory::tensor<int8_t>>& bottom, std::shared_ptr<memory::tensor<int>>& top);
			void convdw3x3s2_int8_neon(const std::shared_ptr<memory::tensor<int8_t>>& bottom, std::shared_ptr<memory::tensor<int>>& top);

			operation<float>* op;
		};
	}
}
#endif // !_OPERATION_CONVOLUTION_ARM_HPP_
