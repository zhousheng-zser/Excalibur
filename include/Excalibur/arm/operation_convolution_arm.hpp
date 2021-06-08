#pragma once
#ifndef _OPERATION_CONVOLUTION_ARM_HPP_
#define _OPERATION_CONVOLUTION_ARM_HPP_

#include "../operation_general_conv.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		class operation_convolution_arm : public operation_general_conv<Dtype>
		{
		public:
			operation_convolution_arm(const operation_param& param);

			virtual int init_weights();

			virtual int init_weights(FILE *fp);

		protected:
			virtual void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
				std::vector<std::shared_ptr<memory::tensor<float>>>& tops);

			virtual void forward_cpu_i8(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
				std::vector<std::shared_ptr<memory::tensor<float>>>& tops);

		private:

			void conv1x1s1_sgemm_transform_kernel_neon();
			void conv3x3s1_winograd64_transform_kernel_neon5();
			void conv3x3s2_transform_kernel_neon();
			void conv_im2col_sgemm_transform_kernel_neon();

			void conv1x1s1_sgemm_neon(const std::shared_ptr < memory::tensor<float>>& bottom,
				std::shared_ptr < memory::tensor<float>>& top);
			void conv1x1s1_neon(const std::shared_ptr < memory::tensor<float>>& bottom,
				std::shared_ptr < memory::tensor<float>>& top);
			void conv3x3s1_winograd64_neon5(const std::shared_ptr < memory::tensor<float>>& bottom,
				std::shared_ptr < memory::tensor<float>>& top);
			void conv3x3s1_neon(const std::shared_ptr < memory::tensor<float>>& bottom,
				std::shared_ptr < memory::tensor<float>>& top);
			void conv3x3s2_packed_neon(const std::shared_ptr < memory::tensor<float>>& bottom,
				std::shared_ptr < memory::tensor<float>>& top);
			void conv_im2col_sgemm_neon(const std::shared_ptr < memory::tensor<float>>& bottom,
				std::shared_ptr < memory::tensor<float>>& top);

			//f32 convolution multiplication
			std::shared_ptr<memory::tensor<float>> kernel_tm_;
			std::shared_ptr<memory::tensor<float>> kernel_tm_gemm_;

			//for int8
			void im2col_sgemm_int8_neon(const int8_t* kernel_tm_gemm_int8_data, int kernel_tm_gemm_int8_cstep, 
				const int8_t *bottom_im2col, int size, int maxk, int inch, 
				int* top, int outw, int outh, int outch);
			void conv3x3s1_winograd43_transform_kernel_int8_neon();
			void conv_im2col_sgemm_transform_kernel_int8_neon();
			void conv1x1s1_sgemm_int8_neon(const std::shared_ptr<memory::tensor<int8_t>>& bottom,
				std::shared_ptr<memory::tensor<int>>& top);
			void conv1x1s2_int8_neon(const std::shared_ptr<memory::tensor<int8_t>>& bottom,
				std::shared_ptr<memory::tensor<int>>& top);
			void conv3x3s1_winograd43_int8_neon(const std::shared_ptr<memory::tensor<int8_t>>& bottom, 
				std::shared_ptr<memory::tensor<int>>& top);
			void conv_im2col_sgemm_int8_neon(const std::shared_ptr<memory::tensor<int8_t>>& bottom, 
				std::shared_ptr<memory::tensor<int>>& top);
			std::vector<std::shared_ptr<memory::tensor<short>>> kernel_tm_winograd_int8_;
			std::shared_ptr<memory::tensor<int8_t>> kernel_tm_gemm_int8_;
		};
	}
}
#endif // !_OPERATION_CONVOLUTION_ARM_HPP_
