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

			static void cut_border_cpu(const std::shared_ptr<memory::tensor<Dtype>>& src,
				std::shared_ptr<memory::tensor<Dtype>>& dst, int top, int bottom, int left, int right);

#ifdef HARDCODE
			virtual void init_weights() {}
#else
			virtual int init_weights(FILE *fp);
#endif //!HARDCODE

		protected:
			virtual void forward_cpu_f32(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
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


		private:
			void forward_cpu_int8(const std::vector<std::shared_ptr<memory::tensor<float>>>& bottoms,
				std::vector<std::shared_ptr<memory::tensor<float>>>& tops);

			void conv3x3s1_winograd23_tr_kernel(const std::shared_ptr<memory::tensor<float> >& kernel,
				std::shared_ptr<memory::tensor<float> >& kernel_tm, int inch, int ou);

			void conv3x3s1_winograd23(const std::shared_ptr<memory::tensor<float> >& bottom_blob, std::shared_ptr<memory::tensor<float> >& top_blob, 
				const std::shared_ptr<memory::tensor<float> >& kernel_tm, bool bias_term_);

			void forward_im2col_tr_kernel(std::shared_ptr<memory::tensor<float> >& kernel_tm);

			void forward_im2col(const std::shared_ptr<memory::tensor<float> >& bottom_blob, std::shared_ptr<memory::tensor<float> >& top_blob,
				const std::shared_ptr<memory::tensor<float> >& kernel_tm, const std::shared_ptr<memory::tensor<float> >& _bias, 
				const int kernel_w,const int kernel_h, const int stride_w, const int stride_h, bool bias_term_);

			void conv3x3s1_winograd23_tr_kernel_int8(const std::shared_ptr<memory::tensor<signed char> >& kernel,
				std::shared_ptr<memory::tensor<short> >& kernel_tm, int inch, int ou);

			void conv3x3s1_winograd23_int8_(const std::shared_ptr<memory::tensor<signed char> >& bottom_blob, std::shared_ptr<memory::tensor<int> >& top_blob,
				const std::shared_ptr<memory::tensor<short> >& kernel_tm, bool bias_term_);

			void conv_im2col_sgemm_int8_dequant_sse(const std::shared_ptr<memory::tensor<signed char> >& bottom_blob, 
				std::shared_ptr<memory::tensor<float> >& top_blob, std::shared_ptr<memory::tensor<signed char> >& _kernel,
				const std::shared_ptr<memory::tensor<float> >& _bias, const int kernel_h, const int kernel_w, 
				const int stride_h, const int stride_w, float scale_dequant, bool bias_term_);


			//others
			void forward_sgemm(const float* input, const float* weights, float* output, memory::orderType order);

			void forward_sbias(float* output, const float* bias, memory::orderType order);

			void forward_k1s1_f32(const std::shared_ptr < memory::tensor<float>>& bottom,
				std::shared_ptr < memory::tensor<float>>& top);

			void quantize_float32_to_int8(const std::shared_ptr<memory::tensor<float>>& src,
				std::shared_ptr<memory::tensor<signed char>>& dst, float scale);

			void dequantize_int32_to_float32(std::shared_ptr<memory::tensor<int>>& src,std::shared_ptr<memory::tensor<float>>& dst,
				float scale, const float* bias, int bias_data_size);


			//f32 convolution multiplication
			std::shared_ptr<memory::tensor<float>> weights1x1_;
			std::shared_ptr<memory::tensor<float>> col_buffer_;
			std::shared_ptr<memory::tensor<float>> bias_multiplier_;
			std::shared_ptr<memory::tensor<float>> kernel_tm;
			std::shared_ptr<memory::tensor<float>> kernel_tm_gemm;
			std::shared_ptr<memory::tensor<float>> tmp;
			std::shared_ptr<memory::tensor<float>> bottom_im2col;
			std::shared_ptr<memory::tensor<float>> bottom_tm;
			std::shared_ptr<memory::tensor<float>> border_bottom;
			std::shared_ptr<memory::tensor<float> > top_blob_bordered;
			std::shared_ptr<memory::tensor<float> > bottom_blob_bordered;
			std::shared_ptr<memory::tensor<float> > bottom_blob_tm;


			//int8 convolution multiplication
			std::shared_ptr<memory::tensor<int>> top_int32;            
			std::shared_ptr<memory::tensor<signed char>> bottom_int8_;
			std::shared_ptr<memory::tensor<signed char>> bottom_int8_bordered;
			std::shared_ptr<memory::tensor<short>> kernel_tm_int8;
			std::shared_ptr<memory::tensor<signed char>> weights1x1_int8_;
			std::shared_ptr<memory::tensor<signed char>> col_buffer_int8_;
			std::shared_ptr<memory::tensor<short> > bottom_blob_int8_tm;
			std::shared_ptr<memory::tensor<signed char>> kernel_tm_int8_sgemm;
			std::shared_ptr<memory::tensor<signed char>> bottom_im2row;
			std::shared_ptr<memory::tensor<signed char>> bottom_tm_int8;

			

			float* col_buffer_data;
			float* bias_multiplier_data;
		};
	}
}
#endif // !_OPERATION_CONVOLUTION_HPP_
