#include "rnet_mobile.hpp"

namespace glasssix
{
	namespace longinus
	{
		rnet_mobile::rnet_mobile(int device)
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
			Copy_Params(conv1_weights, RNet_mobile, quantize_level);
			Copy_Params(conv1_bias, RNet_mobile, quantize_level);
			Copy_Params(prelu1_weights, RNet_mobile, quantize_level);
			Copy_Params(conv2_sep_weights, RNet_mobile, quantize_level);
			Copy_Params(conv2_sep_bias, RNet_mobile, quantize_level);
			Copy_Params(prelu2_weights, RNet_mobile, quantize_level);
			Copy_Params(conv3_sep_weights, RNet_mobile, quantize_level);
			Copy_Params(conv3_sep_bias, RNet_mobile, quantize_level);
			Copy_Params(prelu3_weights, RNet_mobile, quantize_level);
			Copy_Params(conv4_dw_weights, RNet_mobile, quantize_level);
			Copy_Params(conv4_dw_bias, RNet_mobile, quantize_level);
			Copy_Params(prelu4_dw_weights, RNet_mobile, quantize_level);
			Copy_Params(conv4_sep_weights, RNet_mobile, quantize_level);
			Copy_Params(conv4_sep_bias, RNet_mobile, quantize_level);
			Copy_Params(prelu4_weights, RNet_mobile, quantize_level);
			Copy_Params(conv5_dw_weights, RNet_mobile, quantize_level);
			Copy_Params(conv5_dw_bias, RNet_mobile, quantize_level);
			Copy_Params(prelu5_dw_weights, RNet_mobile, quantize_level);
			Copy_Params(conv5_1_weights, RNet_mobile, quantize_level);
			Copy_Params(conv5_1_bias, RNet_mobile, quantize_level);
			Copy_Params(conv5_1_bn_weights, RNet_mobile, quantize_level);
			Copy_Params(conv5_1_bn_bias, RNet_mobile, quantize_level);
			Copy_Params(conv5_1_sc_weights, RNet_mobile, quantize_level);
			Copy_Params(conv5_1_sc_bias, RNet_mobile, quantize_level);
			Copy_Params(conv5_2_weights, RNet_mobile, quantize_level);
			Copy_Params(conv5_2_bias, RNet_mobile, quantize_level);
			Copy_Params(conv5_2_bn_weights, RNet_mobile, quantize_level);
			Copy_Params(conv5_2_bn_bias, RNet_mobile, quantize_level);
			Copy_Params(conv5_2_sc_weights, RNet_mobile, quantize_level);
			Copy_Params(conv5_2_sc_bias, RNet_mobile, quantize_level);

			//

#ifdef USE_CUDA
			if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
				LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
			}
#ifdef USE_CUDNN
			if (cudnnCreate(&cudnn_handle_) != CUDNN_STATUS_SUCCESS) {
				LOG(ERROR) << "Cannot create Cudnn handle. Cudnn won't be available.";
			}
			cudnn_ready_ = true;
#endif
#endif
			//
#ifdef __ARM_NEON
			Init_Conv_arm_Params(conv1, 3, 16, 1, 3, 1, 0, true);
			Init_PReLU_arm_Params(prelu1, 16, false, false);
			Init_Pooling_arm_Params(pool1, 3, 2, 0, 0);
			Init_Conv_arm_Params(conv2_sep, 16, 32, 1, 1, 1, 0, true);
			Init_PReLU_arm_Params(prelu2, 32, false, false);
			Init_Pooling_arm_Params(pool2, 3, 2, 0, 0);
			Init_Conv_arm_Params(conv3_sep, 32, 64, 1, 1, 1, 0, true);
			Init_PReLU_arm_Params(prelu3, 64, false, false);
			Init_Conv_arm_Params(conv4_dw, 64, 64, 64, 3, 1, 0, true);
			Init_PReLU_arm_Params(prelu4_dw, 64, false, false);
			Init_Conv_arm_Params(conv4_sep, 64, 128, 1, 1, 1, 0, true);
			Init_PReLU_arm_Params(prelu4, 128, false, false);
			Init_Conv_arm_Params(conv5_dw, 128, 128, 128, 3, 1, 0, true);
			Init_PReLU_arm_Params(prelu5_dw, 128, false, false);
			Init_InnerProduct_arm_Params(conv5_1, 128, 1, 1, 2, true);
			Init_BatchNorm_arm_Params(conv5_1_bn, 2);
			Init_Scale_arm_Params(conv5_1_sc, 2, true);
			Init_Softmax_arm_Params(cls_prob, 2);
			Init_InnerProduct_arm_Params(conv5_2, 128, 1, 1, 4, true);
			Init_BatchNorm_arm_Params(conv5_2_bn, 4);
			Init_Scale_arm_Params(conv5_2_sc, 4, true);
#else
			Init_Conv_Params(conv1, 3, 16, 1, 3, 1, 0, true);
			Init_PReLU_Params(prelu1, 16, false);
			Init_Pooling_Params(pool1, 3, 2, 0, 0);
			Init_Conv_Params(conv2_sep, 16, 32, 1, 1, 1, 0, true);
			Init_PReLU_Params(prelu2, 32, false);
			Init_Pooling_Params(pool2, 3, 2, 0, 0);
			Init_Conv_Params(conv3_sep, 32, 64, 1, 1, 1, 0, true);
			Init_PReLU_Params(prelu3, 64, false);
			Init_Conv_Params(conv4_dw, 64, 64, 64, 3, 1, 0, true);
			Init_PReLU_Params(prelu4_dw, 64, false);
			Init_Conv_Params(conv4_sep, 64, 128, 1, 1, 1, 0, true);
			Init_PReLU_Params(prelu4, 128, false);
			Init_Conv_Params(conv5_dw, 128, 128, 128, 3, 1, 0, true);
			Init_PReLU_Params(prelu5_dw, 128, false);
			Init_InnerProduct_Params(conv5_1, 128, 1, 1, 2, true);
			Init_BatchNorm_arm_Params(conv5_1_bn, 2);
			Init_Scale_arm_Params(conv5_1_sc, 2, true);
			Init_Softmax_Params(cls_prob, 2);
			Init_InnerProduct_Params(conv5_2, 128, 1, 1, 4, true);
			Init_BatchNorm_arm_Params(conv5_2_bn, 4);
			Init_Scale_arm_Params(conv5_2_sc, 4, true);
#endif
		}


		rnet_mobile::~rnet_mobile()
		{
			delete conv1;
			delete prelu1;
			delete pool1;
			delete conv2_sep;
			delete prelu2;
			delete pool2;
			delete conv3_sep;
			delete prelu3;
			delete conv4_dw;
			delete prelu4_dw;
			delete conv4_sep;
			delete prelu4;
			delete conv5_dw;
			delete prelu5_dw;
			delete conv5_1;
			delete conv5_1_bn;
			delete conv5_1_sc;
			delete cls_prob;
			delete conv5_2;
			delete conv5_2_bn;
			delete conv5_2_sc;

			//conv_weights and bias free automatically, prelu_weights need to free explicitly
			FreeHost(prelu1_weights, false);
			FreeHost(prelu2_weights, false);
			FreeHost(prelu3_weights, false);
			FreeHost(prelu4_dw_weights, false);
			FreeHost(prelu4_weights, false);
			FreeHost(prelu5_dw_weights, false);

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

		void rnet_mobile::Forward_cpu(const std::shared_ptr<tensor<float>> input_data)
		{
			conv1->Forward(input_data, conv1_top_data);
			prelu1->Forward_cpu(conv1_top_data);
			pool1->Forward_cpu(conv1_top_data, pool1_top_data);
			conv2_sep->Forward(pool1_top_data, conv2_sep_top_data);
			prelu2->Forward_cpu(conv2_sep_top_data);
			pool2->Forward_cpu(conv2_sep_top_data, pool2_top_data);
			conv3_sep->Forward(pool2_top_data, conv3_sep_top_data);
			prelu3->Forward_cpu(conv3_sep_top_data);
			conv4_dw->Forward(conv3_sep_top_data, conv4_dw_top_data);
			prelu4_dw->Forward_cpu(conv4_dw_top_data);
			conv4_sep->Forward(conv4_dw_top_data, conv4_sep_top_data);
			prelu4->Forward_cpu(conv4_sep_top_data);
			conv5_dw->Forward(conv4_sep_top_data, conv5_dw_top_data);
			prelu5_dw->Forward_cpu(conv5_dw_top_data);
			conv5_1->Forward_cpu(conv5_dw_top_data, conv5_1_top_data);
			conv5_1_bn->Forward_cpu(conv5_1_top_data);
			conv5_1_sc->Forward_cpu(conv5_1_top_data);
			cls_prob->Forward_cpu(conv5_1_top_data, cls_prob_top_data);
			conv5_2->Forward_cpu(conv5_dw_top_data, conv5_2_top_data);
			conv5_2_bn->Forward_cpu(conv5_2_top_data);
			conv5_2_sc->Forward_cpu(conv5_2_top_data);
		}

#ifdef USE_CUDA
		void rnet_mobile::Forward_gpu_native(const std::shared_ptr<tensor<float>> input_data)
		{
			conv1->Forward(cublas_handle_, input_data, conv1_top_data);
			prelu1->Forward_gpu_native(conv1_top_data);
			pool1->Forward_gpu_native(conv1_top_data, pool1_top_data);
			conv2_sep->Forward(cublas_handle_, pool1_top_data, conv2_sep_top_data);
			prelu2->Forward_gpu_native(conv2_sep_top_data);
			pool2->Forward_gpu_native(conv2_sep_top_data, pool2_top_data);
			conv3_sep->Forward(cublas_handle_, pool2_top_data, conv3_sep_top_data);
			prelu3->Forward_gpu_native(conv3_sep_top_data);
			conv4_dw->Forward(cublas_handle_, conv3_sep_top_data, conv4_dw_top_data);
			prelu4_dw->Forward_gpu_native(conv4_dw_top_data);
			conv4_sep->Forward(cublas_handle_, conv4_dw_top_data, conv4_sep_top_data);
			prelu4->Forward_gpu_native(conv4_sep_top_data);
			conv5_dw->Forward(cublas_handle_, conv4_sep_top_data, conv5_dw_top_data);
			prelu5_dw->Forward_gpu_native(conv5_dw_top_data);
			conv5_1->Forward_gpu_native(cublas_handle_, conv5_dw_top_data, conv5_1_top_data);
			conv5_1_bn->Forward_cpu(conv5_1_top_data);
			conv5_1_sc->Forward_cpu(conv5_1_top_data);
			cls_prob->Forward_cpu(conv5_1_top_data, cls_prob_top_data);
			conv5_2->Forward_gpu_native(cublas_handle_, conv5_dw_top_data, conv5_2_top_data);
			conv5_2_bn->Forward_cpu(conv5_2_top_data);
			conv5_2_sc->Forward_cpu(conv5_2_top_data);
		}
#ifdef USE_CUDNN
		void rnet_mobile::Forward_gpu_cudnn(const std::shared_ptr<tensor<float>> input_data)
		{
			conv1->Forward(cudnn_handle_, input_data, conv1_top_data);
			prelu1->Forward_gpu_native(conv1_top_data);
			pool1->Forward_gpu_native(conv1_top_data, pool1_top_data);
			conv2_sep->Forward(cudnn_handle_, pool1_top_data, conv2_sep_top_data);
			prelu2->Forward_gpu_native(conv2_sep_top_data);
			pool2->Forward_gpu_native(conv2_sep_top_data, pool2_top_data);
			conv3_sep->Forward(cudnn_handle_, pool2_top_data, conv3_sep_top_data);
			prelu3->Forward_gpu_native(conv3_sep_top_data);
			conv4_dw->Forward(cudnn_handle_, conv3_sep_top_data, conv4_dw_top_data);
			prelu4_dw->Forward_gpu_native(conv4_dw_top_data);
			conv4_sep->Forward(cudnn_handle_, conv4_dw_top_data, conv4_sep_top_data);
			prelu4->Forward_gpu_native(conv4_sep_top_data);
			conv5_dw->Forward(cudnn_handle_, conv4_sep_top_data, conv5_dw_top_data);
			prelu5_dw->Forward_gpu_native(conv5_dw_top_data);
			conv5_1->Forward_gpu_native(cublas_handle_, conv5_dw_top_data, conv5_1_top_data);
			conv5_1_bn->Forward_cpu(conv5_1_top_data);
			conv5_1_sc->Forward_cpu(conv5_1_top_data);
			cls_prob->Forward_cpu(conv5_1_top_data, cls_prob_top_data);
			conv5_2->Forward_gpu_native(cublas_handle_, conv5_dw_top_data, conv5_2_top_data);
			conv5_2_bn->Forward_cpu(conv5_2_top_data);
			conv5_2_sc->Forward_cpu(conv5_2_top_data);
		}
#endif
#endif
		void rnet_mobile::Forward(const std::shared_ptr<tensor<float>> input_data)
		{
			if (device_<0)
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