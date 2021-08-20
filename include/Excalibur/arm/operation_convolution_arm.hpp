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
			//float32 pack1
			std::shared_ptr<memory::tensor<float>> kernel_tm_;
			std::shared_ptr<memory::tensor<float>> kernel_tm_gemm_;

#ifdef __ARM_NEON
			//float32 pack4
			void conv3x3s1_winograd64_transform_kernel_pack4_neon();
			void conv3x3s1_winograd42_transform_kernel_pack4_neon();
			void convolution_transform_kernel_pack4_neon();

			std::shared_ptr<memory::tensor<float>> weight_data_pack4_;
			std::shared_ptr<memory::tensor<float>> weight_sgemm_data_pack4_;
			std::shared_ptr<memory::tensor<float>> weight_3x3_winograd42_data_pack4_;

			//float32 pack1to4
			void convolution_transform_kernel_pack1to4_neon();

			std::shared_ptr<memory::tensor<float>> weight_data_pack1to4_;

			//pack4to1
			void conv1x1s1_sgemm_transform_kernel_pack4to1_neon();
			void conv3x3s1_winograd64_transform_kernel_pack4to1_neon();
			void convolution_transform_kernel_pack4to1_neon();

			std::shared_ptr<memory::tensor<float>> weight_data_pack4to1_;
#endif


			//int8 pack1
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
