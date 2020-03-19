#include "pnet_mobile_nir.hpp"
#include <iostream>

namespace glasssix
{
	namespace longinus
	{
		pnet_mobile_nir::pnet_mobile_nir(int device)
		{
			device_ = device;
			if (device_ >= 0)
			{
				int8_quantization_ = false;//do not use int8 in GPU mode
			}

#if SIMD_TYPE >= SIMDTYPE_SSE
			//use for Copy_Int8_Params
			std::shared_ptr<tensor<float>> bottom_round_ = std::make_shared<tensor<float>>(std::vector<int>{mm_align_size});
			float* bottom_round_data_ = bottom_round_->mutable_cpu_data();
#endif // SIMD_TYPE >= SIMDTYPE_SSE

			float quantize_level = INT_MAX;
			//copy float32_data directly
			Copy_Params(conv1_weights, PNet_mobile_nir, quantize_level);
			Copy_Params(conv1_bias, PNet_mobile_nir, quantize_level);
			Copy_Params(prelu1_weights, PNet_mobile_nir, quantize_level);
			Copy_Params(conv2_dw_weights, PNet_mobile_nir, quantize_level);
			Copy_Params(conv2_dw_bias, PNet_mobile_nir, quantize_level);
			Copy_Params(prelu2_dw_weights, PNet_mobile_nir, quantize_level);
			Copy_Params(conv2_sep_weights, PNet_mobile_nir, quantize_level);
			Copy_Params(conv2_sep_bias, PNet_mobile_nir, quantize_level);
			Copy_Params(prelu2_weights, PNet_mobile_nir, quantize_level);
			Copy_Params(conv3_dw_weights, PNet_mobile_nir, quantize_level);
			Copy_Params(conv3_dw_bias, PNet_mobile_nir, quantize_level);
			Copy_Params(prelu3_dw_weights, PNet_mobile_nir, quantize_level);
			Copy_Params(conv3_sep_weights, PNet_mobile_nir, quantize_level);
			Copy_Params(conv3_sep_bias, PNet_mobile_nir, quantize_level);
			Copy_Params(prelu3_weights, PNet_mobile_nir, quantize_level);
			Copy_Params(conv4_dw_weights, PNet_mobile_nir, quantize_level);
			Copy_Params(conv4_dw_bias, PNet_mobile_nir, quantize_level);
			Copy_Params(prelu4_dw_weights, PNet_mobile_nir, quantize_level);
			Copy_Params(conv4_1_weights, PNet_mobile_nir, quantize_level);
			Copy_Params(conv4_1_bias, PNet_mobile_nir, quantize_level);
			Copy_Params(conv4_1_bn_weights, PNet_mobile_nir, quantize_level);
			Copy_Params(conv4_1_bn_bias, PNet_mobile_nir, quantize_level);
			Copy_Params(conv4_1_scale_weights, PNet_mobile_nir, quantize_level);
			Copy_Params(conv4_1_scale_bias, PNet_mobile_nir, quantize_level);

#ifdef USE_CUDA
			CUBLAS_CHECK(cublasCreate(&cublas_handle_));
#ifdef USE_CUDNN
			CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
			cudnn_ready_ = true;
#endif
#endif

#ifdef __ARM_NEON
			Init_Conv_arm_Params(conv1, 3, 8, 1, 3, 2, 0, true);
			Init_PReLU_arm_Params(prelu1, 8, false, false);
			Init_Conv_arm_Params(conv2_dw, 8, 8, 8, 3, 1, 0, true);
			Init_PReLU_arm_Params(prelu2_dw, 8, false, false);
			Init_Conv_arm_Params(conv2_sep, 8, 16, 1, 1, 1, 0, true);
			Init_PReLU_arm_Params(prelu2, 16, false, false);
			Init_Conv_arm_Params(conv3_dw, 16, 16, 16, 3, 2, 0, true);
			Init_PReLU_arm_Params(prelu3_dw, 16, false, false);
			Init_Conv_arm_Params(conv3_sep, 16, 24, 1, 1, 1, 0, true);
			Init_PReLU_arm_Params(prelu3, 24, false, false);
			Init_Conv_arm_Params(conv4_dw, 24, 24, 24, 3, 1, 0, true);
			Init_PReLU_arm_Params(prelu4_dw, 24, false, false);
			Init_Conv_arm_Params(conv4_1, 24, 2, 1, 1, 1, 0, true);
			Init_BatchNorm_arm_Params(conv4_1_bn, 2);
			Init_Scale_arm_Params(conv4_1_scale, 2, true);
			Init_Softmax_arm_Params(cls_prob, 2);
#else
			Init_Conv_Params(conv1, 3, 8, 1, 3, 2, 0, true);
			Init_PReLU_Params(prelu1, 8, false);
			Init_Conv_Params(conv2_dw, 8, 8, 8, 3, 1, 0, true);
			Init_PReLU_Params(prelu2_dw, 8, false);
			Init_Conv_Params(conv2_sep, 8, 16, 1, 1, 1, 0, true);
			Init_PReLU_Params(prelu2, 16, false);
			Init_Conv_Params(conv3_dw, 16, 16, 16, 3, 2, 0, true);
			Init_PReLU_Params(prelu3_dw, 16, false);
			Init_Conv_Params(conv3_sep, 16, 24, 1, 1, 1, 0, true);
			Init_PReLU_Params(prelu3, 24, false);
			Init_Conv_Params(conv4_dw, 24, 24, 24, 3, 1, 0, true);
			Init_PReLU_Params(prelu4_dw, 24, false);
			Init_Conv_Params(conv4_1, 24, 2, 1, 1, 1, 0, true);
			Init_BatchNorm_arm_Params(conv4_1_bn, 2);
			Init_Scale_arm_Params(conv4_1_scale, 2, true);
			Init_Softmax_Params(cls_prob, 2);
#endif // !__ARM_NEON
		}

		pnet_mobile_nir::~pnet_mobile_nir()
		{
			delete conv1;
			delete prelu1;
			delete conv2_dw;
			delete prelu2_dw;
			delete conv2_sep;
			delete prelu2;
			delete conv3_dw;
			delete prelu3_dw;
			delete conv3_sep;
			delete prelu3;
			delete conv4_dw;
			delete prelu4_dw;
			delete conv4_1;
			delete conv4_1_bn;
			delete conv4_1_scale;
			delete cls_prob;

			//conv_weights and bias free automatically, prelu_weights need to free explicitly
			FreeHost(prelu1_weights, false);
			FreeHost(prelu2_dw_weights, false);
			FreeHost(prelu2_weights, false);
			FreeHost(prelu3_dw_weights, false);
			FreeHost(prelu3_weights, false);
			FreeHost(prelu4_dw_weights, false);

#ifdef USE_CUDA
			if (cublas_handle_)
			{
				CUBLAS_CHECK(cublasDestroy(cublas_handle_));
			}
#ifdef USE_CUDNN
			if (cudnn_handle_)
			{
				CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
			}
#endif
#endif
		}

		void pnet_mobile_nir::Forward_cpu(const std::shared_ptr<tensor<float>> input_data)
		{
			conv1->Forward(input_data, conv1_top_data);
			prelu1->Forward_cpu(conv1_top_data);
			conv2_dw->Forward(conv1_top_data, conv2_dw_top_data);
			prelu2_dw->Forward_cpu(conv2_dw_top_data);
			conv2_sep->Forward(conv2_dw_top_data, conv2_sep_top_data);
			prelu2->Forward_cpu(conv2_sep_top_data);
			conv3_dw->Forward(conv2_sep_top_data, conv3_dw_top_data);
			prelu3_dw->Forward_cpu(conv3_dw_top_data);
			conv3_sep->Forward(conv3_dw_top_data, conv3_sep_top_data);
			prelu3->Forward_cpu(conv3_sep_top_data);
			conv4_dw->Forward(conv3_sep_top_data, conv4_dw_top_data);
			prelu4_dw->Forward_cpu(conv4_dw_top_data);
			conv4_1->Forward(conv4_dw_top_data, conv4_1_top_data);
			conv4_1_bn->Forward_cpu(conv4_1_top_data);
			conv4_1_scale->Forward_cpu(conv4_1_top_data);
			cls_prob->Forward_cpu(conv4_1_top_data, cls_prob_top_data);
		}

#ifdef USE_CUDA
		void pnet_mobile_nir::Forward_gpu_native(const std::shared_ptr<tensor<float>> input_data)
		{
			conv1->Forward(cublas_handle_, input_data, conv1_top_data);
			prelu1->Forward_gpu_native(conv1_top_data);
			conv2_dw->Forward(cublas_handle_, conv1_top_data, conv2_dw_top_data);
			prelu2_dw->Forward_gpu_native(conv2_dw_top_data);
			conv2_sep->Forward(cublas_handle_, conv2_dw_top_data, conv2_sep_top_data);
			prelu2->Forward_gpu_native(conv2_sep_top_data);
			conv3_dw->Forward(cublas_handle_, conv2_sep_top_data, conv3_dw_top_data);
			prelu3_dw->Forward_gpu_native(conv3_dw_top_data);
			conv3_sep->Forward(cublas_handle_, conv3_dw_top_data, conv3_sep_top_data);
			prelu3->Forward_gpu_native(conv3_sep_top_data);
			conv4_dw->Forward(cublas_handle_, conv3_sep_top_data, conv4_dw_top_data);
			prelu4_dw->Forward_gpu_native(conv4_dw_top_data);
			conv4_1->Forward(cublas_handle_, conv4_dw_top_data, conv4_1_top_data);
			conv4_1_bn->Forward_cpu(conv4_1_top_data);
			conv4_1_scale->Forward_cpu(conv4_1_top_data);
			cls_prob->Forward_cpu(conv4_1_top_data, cls_prob_top_data);
		}

#ifdef USE_CUDNN
		void pnet_mobile_nir::Forward_gpu_cudnn(const std::shared_ptr<tensor<float>> input_data)
		{
			conv1->Forward(cudnn_handle_, input_data, conv1_top_data);
			prelu1->Forward_gpu_native(conv1_top_data);
			conv2_dw->Forward(cudnn_handle_, conv1_top_data, conv2_dw_top_data);
			prelu2_dw->Forward_gpu_native(conv2_dw_top_data);
			conv2_sep->Forward(cudnn_handle_, conv2_dw_top_data, conv2_sep_top_data);
			prelu2->Forward_gpu_native(conv2_sep_top_data);
			conv3_dw->Forward(cudnn_handle_, conv2_sep_top_data, conv3_dw_top_data);
			prelu3_dw->Forward_gpu_native(conv3_dw_top_data);
			conv3_sep->Forward(cudnn_handle_, conv3_dw_top_data, conv3_sep_top_data);
			prelu3->Forward_gpu_native(conv3_sep_top_data);
			conv4_dw->Forward(cudnn_handle_, conv3_sep_top_data, conv4_dw_top_data);
			prelu4_dw->Forward_gpu_native(conv4_dw_top_data);
			conv4_1->Forward(cudnn_handle_, conv4_dw_top_data, conv4_1_top_data);
			conv4_1_bn->Forward_cpu(conv4_1_top_data);
			conv4_1_scale->Forward_cpu(conv4_1_top_data);
			cls_prob->Forward_cpu(conv4_1_top_data, cls_prob_top_data);
		}

#endif
#endif

		void pnet_mobile_nir::Forward(const std::shared_ptr<tensor<float>> input_data)
		{
			if (device_ < 0)
			{
				Forward_cpu(input_data);
			}
			else
			{
#ifdef USE_CUDA
#ifdef USE_CUDNN
				Forward_gpu_cudnn(input_data);
				return;
#endif
				Forward_gpu_native(input_data);
				return;
#else
				NO_GPU;
#endif
			}
		}
	}
}