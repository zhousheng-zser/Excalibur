#pragma once
#ifndef _OPERATION_CONVOLUTION_HPP_
#define _OPERATION_CONVOLUTION_HPP_

#include "operation_general_conv.hpp"

namespace glasssix
{
	namespace excalibur
	{
		template<typename Dtype>
		class operation_convolution : public operation_general_conv<Dtype>
		{
		public:
			operation_convolution(const operation_param& param);
			virtual ~operation_convolution() {}
			virtual int init_weights();

			virtual int init_weights(FILE *fp);

		protected:
			virtual void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
				std::vector<std::shared_ptr<memory::tensor<float>>>& tops);

			virtual void forward_cpu_i8(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
				std::vector<std::shared_ptr<memory::tensor<float>>>& tops);

			virtual void forward_gpu_f32(
#ifdef USE_CUDA
				cublasHandle_t &cublas_handle_,
#ifdef USE_CUDNN
				cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
#endif //!USE_CUDA
				const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
				std::vector<std::shared_ptr<memory::tensor<float>>>& tops);

#ifdef USE_CUDA
			virtual void forward_gpu_f16(
				cublasHandle_t& cublas_handle_,
#ifdef USE_CUDNN
				cudnnHandle_t cudnn_handle,
#endif //!USE_CUDNN
				const std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& bottoms,
				std::vector<std::shared_ptr<memory::tensor<unsigned short>>>& tops);
#endif //!USE_CUDA

		private:

			void conv3x3s1_winograd23_tr_kernel();

			void conv3x3s1_winograd23(const std::shared_ptr<memory::tensor<float> >& bottom_blob, 
				std::shared_ptr<memory::tensor<float> >& top_blob);

			void forward_im2col_tr_kernel();

			void forward_im2col(const std::shared_ptr<memory::tensor<float> >& bottom_blob, 
				std::shared_ptr<memory::tensor<float> >& top_blob);

			void conv3x3s1_winograd23_tr_kernel_int8();

			void conv3x3s1_winograd23_int8(const std::shared_ptr<memory::tensor<signed char> >& bottom_blob, 
				std::shared_ptr<memory::tensor<int> >& top_blob);

			void forward_im2col_int8_tr_kernel();

			void conv_im2col_sgemm_int8_dequant_sse(const std::shared_ptr<memory::tensor<signed char> >& bottom_blob, 
				std::shared_ptr<memory::tensor<float> >& top_blob, std::vector<float>& scale_dequants);


			//others
			void forward_cpu_sgemm(const float* input, const float* weights, float* output, memory::orderType order);

			void forward_cpu_sbias(float* output, const float* bias, memory::orderType order);

#ifdef USE_CUDA
			void forward_gpu_sgemm(cublasHandle_t &cublas_handle, const float* input, const float* weights, float* output, memory::orderType order);

			void forward_gpu_sbias(cublasHandle_t &cublas_handle, float* output, const float* bias, memory::orderType order);

			void forward_gpu_hgemm(cublasHandle_t& cublas_handle, const unsigned short* input, const unsigned short* weights, unsigned short* output, memory::orderType order);

			void forward_gpu_hbias(cublasHandle_t& cublas_handle, unsigned short* output, const unsigned short* bias, memory::orderType order);
#endif //!USE_CUDA

			void forward_cpu_k1s1_f32(const std::shared_ptr < memory::tensor<float>>& bottom,
				std::shared_ptr < memory::tensor<float>>& top);

			//f32 convolution multiplication
			std::shared_ptr<memory::tensor<float>> weights1x1_;
			std::shared_ptr<memory::tensor<float>> col_buffer_;
			std::shared_ptr<memory::tensor<float>> bias_multiplier_;
			std::vector<std::shared_ptr<memory::tensor<float>>> kernel_tm_;
			std::vector<std::shared_ptr<memory::tensor<float>>> kernel_tm_gemm_;
			std::shared_ptr<memory::tensor<float>> tmp_;
			std::shared_ptr<memory::tensor<float>> bottom_im2col_;
			std::shared_ptr<memory::tensor<float>> bottom_tm_;
			std::shared_ptr<memory::tensor<float>> border_bottom_;
			std::shared_ptr<memory::tensor<float> > top_blob_bordered_;
			std::shared_ptr<memory::tensor<float> > bottom_blob_bordered_;
			std::shared_ptr<memory::tensor<float> > bottom_blob_tm_;

			float* col_buffer_data;
			float* bias_multiplier_data;


			//int8 convolution multiplication
			std::shared_ptr<memory::tensor<int>> top_int32_;            
			std::shared_ptr<memory::tensor<signed char>> bottom_int8_;
			std::shared_ptr<memory::tensor<signed char>> bottom_int8_bordered_;
			std::vector<std::shared_ptr<memory::tensor<signed char>>> kernel_tm_int8_sgemm_;
			std::vector<std::shared_ptr<memory::tensor<short>>> kernel_tm_int8_;
			std::shared_ptr<memory::tensor<signed char>> weights1x1_int8_;
			std::shared_ptr<memory::tensor<signed char>> col_buffer_int8_;
			std::shared_ptr<memory::tensor<short> > bottom_blob_int8_tm_;
			std::shared_ptr<memory::tensor<signed char>> bottom_im2col_int8_;
			std::shared_ptr<memory::tensor<signed char>> bottom_tm_int8_;
			std::shared_ptr<memory::tensor<signed char>> bottom_im2row_;
			

			//fp16
			std::shared_ptr<memory::tensor<unsigned short>> col_buffer_f16_;
			std::shared_ptr<memory::tensor<unsigned short>> bias_multiplier_f16_;
			unsigned short* col_buffer_f16_data_;
			unsigned short* bias_multiplier_f16_data_;
		};
	}
}
#endif // !_OPERATION_CONVOLUTION_HPP_
