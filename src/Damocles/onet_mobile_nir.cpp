#include "onet_mobile_nir.hpp"
#include "Primitives/memory.hpp"

using glasssix::memory::aligned_heap_free;

using namespace glasssix::memory;

namespace glasssix
{
	namespace longinus
	{
		onet_mobile_nir::onet_mobile_nir(int device)
		{
			device_ = device;
			if (device_ >= 0)
			{
				int8_quantization_ = false;//do not use int8 in GPU mode
			}

#ifdef __ARM_NEON
			int8_quantization_ = false;//do not use int8 in ARM mode
#endif

#if SIMD_TYPE >= SIMDTYPE_SSE
			//use for Copy_Int8_Params
			std::shared_ptr<memory::tensor<float>> bottom_round_ = std::make_shared<memory::tensor<float>>(std::vector<int>{mm_align_size});
			float* bottom_round_data_ = bottom_round_->mutable_cpu_data();
#endif // SIMD_TYPE >= SIMDTYPE_SSE

			float quantize_level = INT_MAX;
			//copy float32_data directly
			Copy_Params(conv1_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv1_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(relu1_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv1_dw_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv1_dw_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(relu1_dw_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv2_ex_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv2_ex_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(relu2_ex_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv2_dw_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv2_dw_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(relu2_dw_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv2_em_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv2_em_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(conv2_1_ex_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv2_1_ex_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(relu2_1_ex_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv2_1_dw_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv2_1_dw_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(relu2_1_dw_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv2_1_em_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv2_1_em_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(conv3_ex_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv3_ex_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(relu3_ex_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv3_dw_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv3_dw_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(relu3_dw_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv3_em_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv3_em_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(conv3_1_ex_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv3_1_ex_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(relu3_1_ex_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv3_1_dw_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv3_1_dw_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(relu3_1_dw_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv3_1_em_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv3_1_em_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(conv4_ex_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv4_ex_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(relu4_ex_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv4_dw_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv4_dw_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(relu4_dw_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv4_em_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv4_em_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(conv4_1_ex_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv4_1_ex_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(relu4_1_ex_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv4_1_dw_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv4_1_dw_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(relu4_1_dw_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv4_1_em_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv4_1_em_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(conv5_ex_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv5_ex_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(relu5_ex_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv5_dw_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv5_dw_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(conv6_1_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv6_1_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(conv6_2_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv6_2_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(conv6_3_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv6_3_bias, ONet_mobile_nir, quantize_level);
			Copy_Params(conv6_4_weights, ONet_mobile_nir, quantize_level);
			Copy_Params(conv6_4_bias, ONet_mobile_nir, quantize_level);

#ifdef USE_CUDA
			CUBLAS_CHECK(cublasCreate(&cublas_handle_));
#ifdef USE_CUDNN
			CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
			cudnn_ready_ = true;
#endif
#endif

#ifdef __ARM_NEON
			Init_Conv_arm_Params(conv1, 3, 32, 1, 3, 2, 1, true);
			Init_PReLU_arm_Params(relu1, 32, false, false);
			Init_Conv_arm_Params(conv1_dw, 32, 32, 32, 3, 1, 1, true);
			Init_PReLU_arm_Params(relu1_dw, 32, false, false);
			Init_Conv_arm_Params(conv2_ex, 32, 64, 1, 1, 1, 0, true);
			Init_PReLU_arm_Params(relu2_ex, 64, false, false);
			Init_Conv_arm_Params(conv2_dw, 64, 64, 64, 3, 2, 1, true);
			Init_PReLU_arm_Params(relu2_dw, 64, false, false);
			Init_Conv_arm_Params(conv2_em, 64, 32, 1, 1, 1, 0, true);
			Init_Conv_arm_Params(conv2_1_ex, 32, 64, 1, 1, 1, 0, true);
			Init_PReLU_arm_Params(relu2_1_ex, 64, false, false);
			Init_Conv_arm_Params(conv2_1_dw, 64, 64, 64, 3, 1, 1, true);
			Init_PReLU_arm_Params(relu2_1_dw, 64, false, false);
			Init_Conv_arm_Params(conv2_1_em, 64, 32, 1, 1, 1, 0, true);
			Init_Eltwise_arm_Params(res2, 0);
			Init_Conv_arm_Params(conv3_ex, 32, 128, 1, 1, 1, 0, true);
			Init_PReLU_arm_Params(relu3_ex, 128, false, false);
			Init_Conv_arm_Params(conv3_dw, 128, 128, 128, 3, 2, 1, true);
			Init_PReLU_arm_Params(relu3_dw, 128, false, false);
			Init_Conv_arm_Params(conv3_em, 128, 64, 1, 1, 1, 0, true);
			Init_Conv_arm_Params(conv3_1_ex, 64, 128, 1, 1, 1, 0, true);
			Init_PReLU_arm_Params(relu3_1_ex, 128, false, false);
			Init_Conv_arm_Params(conv3_1_dw, 128, 128, 128, 3, 1, 1, true);
			Init_PReLU_arm_Params(relu3_1_dw, 128, false, false);
			Init_Conv_arm_Params(conv3_1_em, 128, 64, 1, 1, 1, 0, true);
			Init_Eltwise_arm_Params(res3, 0);
			Init_Conv_arm_Params(conv4_ex, 64, 256, 1, 1, 1, 0, true);
			Init_PReLU_arm_Params(relu4_ex, 256, false, false);
			Init_Conv_arm_Params(conv4_dw, 256, 256, 256, 3, 2, 1, true);
			Init_PReLU_arm_Params(relu4_dw, 256, false, false);
			Init_Conv_arm_Params(conv4_em, 256, 64, 1, 1, 1, 0, true);
			Init_Conv_arm_Params(conv4_1_ex, 64, 128, 1, 1, 1, 0, true);
			Init_PReLU_arm_Params(relu4_1_ex, 128, false, false);
			Init_Conv_arm_Params(conv4_1_dw, 128, 128, 128, 3, 1, 1, true);
			Init_PReLU_arm_Params(relu4_1_dw, 128, false, false);
			Init_Conv_arm_Params(conv4_1_em, 128, 64, 1, 1, 1, 0, true);
			Init_Eltwise_arm_Params(res4, 0);
			Init_Conv_arm_Params(conv5_ex, 64, 256, 1, 1, 1, 0, true);
			Init_PReLU_arm_Params(relu5_ex, 256, false, false);
			Init_Conv_arm_Params(conv5_dw, 256, 256, 256, 4, 1, 0, true);
			Init_InnerProduct_arm_Params(conv6_1, 256, 1, 1, 1, true);
			Init_Sigmoid_arm_Params(prob1);
			Init_InnerProduct_arm_Params(conv6_2, 256, 1, 1, 4, true);
			Init_InnerProduct_arm_Params(conv6_3, 256, 1, 1, 3, true);
			Init_InnerProduct_arm_Params(conv6_4, 256, 1, 1, 10, true);
#else
			Init_Conv_Params(conv1, 3, 32, 1, 3, 2, 1, true);
			Init_PReLU_Params(relu1, 32, false);
			Init_Conv_Params(conv1_dw, 32, 32, 32, 3, 1, 1, true);
			Init_PReLU_Params(relu1_dw, 32, false);
			Init_Conv_Params(conv2_ex, 32, 64, 1, 1, 1, 0, true);
			Init_PReLU_Params(relu2_ex, 64, false);
			Init_Conv_Params(conv2_dw, 64, 64, 64, 3, 2, 1, true);
			Init_PReLU_Params(relu2_dw, 64, false);
			Init_Conv_Params(conv2_em, 64, 32, 1, 1, 1, 0, true);
			Init_Conv_Params(conv2_1_ex, 32, 64, 1, 1, 1, 0, true);
			Init_PReLU_Params(relu2_1_ex, 64, false);
			Init_Conv_Params(conv2_1_dw, 64, 64, 64, 3, 1, 1, true);
			Init_PReLU_Params(relu2_1_dw, 64, false);
			Init_Conv_Params(conv2_1_em, 64, 32, 1, 1, 1, 0, true);
			Init_Eltwise_Params(res2, 0);
			Init_Conv_Params(conv3_ex, 32, 128, 1, 1, 1, 0, true);
			Init_PReLU_Params(relu3_ex, 128, false);
			Init_Conv_Params(conv3_dw, 128, 128, 128, 3, 2, 1, true);
			Init_PReLU_Params(relu3_dw, 128, false);
			Init_Conv_Params(conv3_em, 128, 64, 1, 1, 1, 0, true);
			Init_Conv_Params(conv3_1_ex, 64, 128, 1, 1, 1, 0, true);
			Init_PReLU_Params(relu3_1_ex, 128, false);
			Init_Conv_Params(conv3_1_dw, 128, 128, 128, 3, 1, 1, true);
			Init_PReLU_Params(relu3_1_dw, 128, false);
			Init_Conv_Params(conv3_1_em, 128, 64, 1, 1, 1, 0, true);
			Init_Eltwise_Params(res3, 0);
			Init_Conv_Params(conv4_ex, 64, 256, 1, 1, 1, 0, true);
			Init_PReLU_Params(relu4_ex, 256, false);
			Init_Conv_Params(conv4_dw, 256, 256, 256, 3, 2, 1, true);
			Init_PReLU_Params(relu4_dw, 256, false);
			Init_Conv_Params(conv4_em, 256, 64, 1, 1, 1, 0, true);
			Init_Conv_Params(conv4_1_ex, 64, 128, 1, 1, 1, 0, true);
			Init_PReLU_Params(relu4_1_ex, 128, false);
			Init_Conv_Params(conv4_1_dw, 128, 128, 128, 3, 1, 1, true);
			Init_PReLU_Params(relu4_1_dw, 128, false);
			Init_Conv_Params(conv4_1_em, 128, 64, 1, 1, 1, 0, true);
			Init_Eltwise_Params(res4, 0);
			Init_Conv_Params(conv5_ex, 64, 256, 1, 1, 1, 0, true);
			Init_PReLU_Params(relu5_ex, 256, false);
			Init_Conv_Params(conv5_dw, 256, 256, 256, 4, 1, 0, true);
			Init_InnerProduct_Params(conv6_1, 256, 1, 1, 1, true);
			Init_Sigmoid_Params(prob1);
			Init_InnerProduct_Params(conv6_2, 256, 1, 1, 4, true);
			Init_InnerProduct_Params(conv6_3, 256, 1, 1, 3, true);
			Init_InnerProduct_Params(conv6_4, 256, 1, 1, 10, true);
#endif
		}

		onet_mobile_nir::~onet_mobile_nir()
		{
			delete conv1;
			delete relu1;
			delete conv1_dw;
			delete relu1_dw;
			delete conv2_ex;
			delete relu2_ex;
			delete conv2_dw;
			delete relu2_dw;
			delete conv2_em;
			delete conv2_1_ex;
			delete relu2_1_ex;
			delete conv2_1_dw;
			delete relu2_1_dw;
			delete conv2_1_em;
			delete res2;
			delete conv3_ex;
			delete relu3_ex;
			delete conv3_dw;
			delete relu3_dw;
			delete conv3_em;
			delete conv3_1_ex;
			delete relu3_1_ex;
			delete conv3_1_dw;
			delete relu3_1_dw;
			delete conv3_1_em;
			delete res3;
			delete conv4_ex;
			delete relu4_ex;
			delete conv4_dw;
			delete relu4_dw;
			delete conv4_em;
			delete conv4_1_ex;
			delete relu4_1_ex;
			delete conv4_1_dw;
			delete relu4_1_dw;
			delete conv4_1_em;
			delete res4;
			delete conv5_ex;
			delete relu5_ex;
			delete conv5_dw;
			delete conv6_1;
			delete conv6_2;
			delete conv6_3;
			delete conv6_4;
			delete prob1;

			//conv_weights and bias free automatically, prelu_weights need to free explicitly
			aligned_heap_free(relu1_weights);
			aligned_heap_free(relu1_dw_weights);
			aligned_heap_free(relu2_ex_weights);
			aligned_heap_free(relu2_dw_weights);
			aligned_heap_free(relu2_1_ex_weights);
			aligned_heap_free(relu2_1_dw_weights);
			aligned_heap_free(relu3_ex_weights);
			aligned_heap_free(relu3_dw_weights);
			aligned_heap_free(relu3_1_ex_weights);
			aligned_heap_free(relu3_1_dw_weights);
			aligned_heap_free(relu4_ex_weights);
			aligned_heap_free(relu4_dw_weights);
			aligned_heap_free(relu4_1_ex_weights);
			aligned_heap_free(relu4_1_dw_weights);
			aligned_heap_free(relu5_ex_weights);

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

		void onet_mobile_nir::Forward_cpu(const std::shared_ptr<memory::tensor<float>> input_data)
		{
			conv1->Forward(input_data, conv1_top_data);
			relu1->Forward_cpu(conv1_top_data);
			conv1_dw->Forward(conv1_top_data, conv1_dw_top_data);
			relu1_dw->Forward_cpu(conv1_dw_top_data);
			conv2_ex->Forward(conv1_dw_top_data, conv2_ex_top_data);
			relu2_ex->Forward_cpu(conv2_ex_top_data);
			conv2_dw->Forward(conv2_ex_top_data, conv2_dw_top_data);
			relu2_dw->Forward_cpu(conv2_dw_top_data);
			conv2_em->Forward(conv2_dw_top_data, conv2_em_top_data);
			conv2_1_ex->Forward(conv2_em_top_data, conv2_1_ex_top_data);
			relu2_1_ex->Forward_cpu(conv2_1_ex_top_data);
			conv2_1_dw->Forward(conv2_1_ex_top_data, conv2_1_dw_top_data);
			relu2_1_dw->Forward_cpu(conv2_1_dw_top_data);
			conv2_1_em->Forward(conv2_1_dw_top_data, conv2_1_em_top_data);
			res2->Forward_cpu(std::vector<std::shared_ptr<memory::tensor<float>>>{conv2_em_top_data, conv2_1_em_top_data}, res2_top_data);
			conv3_ex->Forward(res2_top_data, conv3_ex_top_data);
			relu3_ex->Forward_cpu(conv3_ex_top_data);
			conv3_dw->Forward(conv3_ex_top_data, conv3_dw_top_data);
			relu3_dw->Forward_cpu(conv3_dw_top_data);
			conv3_em->Forward(conv3_dw_top_data, conv3_em_top_data);
			conv3_1_ex->Forward(conv3_em_top_data, conv3_1_ex_top_data);
			relu3_1_ex->Forward_cpu(conv3_1_ex_top_data);
			conv3_1_dw->Forward(conv3_1_ex_top_data, conv3_1_dw_top_data);
			relu3_1_dw->Forward_cpu(conv3_1_dw_top_data);
			conv3_1_em->Forward(conv3_1_dw_top_data, conv3_1_em_top_data);
			res3->Forward_cpu(std::vector<std::shared_ptr<memory::tensor<float>>>{conv3_em_top_data, conv3_1_em_top_data}, res3_top_data);
			conv4_ex->Forward(res3_top_data, conv4_ex_top_data);
			relu4_ex->Forward_cpu(conv4_ex_top_data);
			conv4_dw->Forward(conv4_ex_top_data, conv4_dw_top_data);
			relu4_dw->Forward_cpu(conv4_dw_top_data);
			conv4_em->Forward(conv4_dw_top_data, conv4_em_top_data);
			conv4_1_ex->Forward(conv4_em_top_data, conv4_1_ex_top_data);
			relu4_1_ex->Forward_cpu(conv4_1_ex_top_data);
			conv4_1_dw->Forward(conv4_1_ex_top_data, conv4_1_dw_top_data);
			relu4_1_dw->Forward_cpu(conv4_1_dw_top_data);
			conv4_1_em->Forward(conv4_1_dw_top_data, conv4_1_em_top_data);
			res4->Forward_cpu(std::vector<std::shared_ptr<memory::tensor<float>>>{conv4_em_top_data, conv4_1_em_top_data}, res4_top_data);
			conv5_ex->Forward(res4_top_data, conv5_ex_top_data);
			relu5_ex->Forward_cpu(conv5_ex_top_data);
			conv5_dw->Forward(conv5_ex_top_data, conv5_dw_top_data);
			conv6_1->Forward_cpu(conv5_dw_top_data, conv6_1_top_data);
			prob1->Forward_cpu(conv6_1_top_data);
			conv6_2->Forward_cpu(conv5_dw_top_data, conv6_2_top_data);
			conv6_3->Forward_cpu(conv5_dw_top_data, conv6_3_top_data);
			conv6_4->Forward_cpu(conv5_dw_top_data, conv6_4_top_data);
		}

#ifdef USE_CUDA
		void onet_mobile_nir::Forward_gpu_native(const std::shared_ptr<memory::tensor<float>> input_data)
		{
			conv1->Forward(cublas_handle_, input_data, conv1_top_data);
			relu1->Forward_gpu_native(conv1_top_data);
			conv1_dw->Forward(cublas_handle_, conv1_top_data, conv1_dw_top_data);
			relu1_dw->Forward_gpu_native(conv1_dw_top_data);
			conv2_ex->Forward(cublas_handle_, conv1_dw_top_data, conv2_ex_top_data);
			relu2_ex->Forward_gpu_native(conv2_ex_top_data);
			conv2_dw->Forward(cublas_handle_, conv2_ex_top_data, conv2_dw_top_data);
			relu2_dw->Forward_gpu_native(conv2_dw_top_data);
			conv2_em->Forward(cublas_handle_, conv2_dw_top_data, conv2_em_top_data);
			conv2_1_ex->Forward(cublas_handle_, conv2_em_top_data, conv2_1_ex_top_data);
			relu2_1_ex->Forward_gpu_native(conv2_1_ex_top_data);
			conv2_1_dw->Forward(cublas_handle_, conv2_1_ex_top_data, conv2_1_dw_top_data);
			relu2_1_dw->Forward_gpu_native(conv2_1_dw_top_data);
			conv2_1_em->Forward(cublas_handle_, conv2_1_dw_top_data, conv2_1_em_top_data);
			res2->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<memory::tensor<float>>>{conv2_em_top_data, conv2_1_em_top_data}, res2_top_data);
			conv3_ex->Forward(cublas_handle_, res2_top_data, conv3_ex_top_data);
			relu3_ex->Forward_gpu_native(conv3_ex_top_data);
			conv3_dw->Forward(cublas_handle_, conv3_ex_top_data, conv3_dw_top_data);
			relu3_dw->Forward_gpu_native(conv3_dw_top_data);
			conv3_em->Forward(cublas_handle_, conv3_dw_top_data, conv3_em_top_data);
			conv3_1_ex->Forward(cublas_handle_, conv3_em_top_data, conv3_1_ex_top_data);
			relu3_1_ex->Forward_gpu_native(conv3_1_ex_top_data);
			conv3_1_dw->Forward(cublas_handle_, conv3_1_ex_top_data, conv3_1_dw_top_data);
			relu3_1_dw->Forward_gpu_native(conv3_1_dw_top_data);
			conv3_1_em->Forward(cublas_handle_, conv3_1_dw_top_data, conv3_1_em_top_data);
			res3->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<memory::tensor<float>>>{conv3_em_top_data, conv3_1_em_top_data}, res3_top_data);
			conv4_ex->Forward(cublas_handle_, res3_top_data, conv4_ex_top_data);
			relu4_ex->Forward_gpu_native(conv4_ex_top_data);
			conv4_dw->Forward(cublas_handle_, conv4_ex_top_data, conv4_dw_top_data);
			relu4_dw->Forward_gpu_native(conv4_dw_top_data);
			conv4_em->Forward(cublas_handle_, conv4_dw_top_data, conv4_em_top_data);
			conv4_1_ex->Forward(cublas_handle_, conv4_em_top_data, conv4_1_ex_top_data);
			relu4_1_ex->Forward_gpu_native(conv4_1_ex_top_data);
			conv4_1_dw->Forward(cublas_handle_, conv4_1_ex_top_data, conv4_1_dw_top_data);
			relu4_1_dw->Forward_gpu_native(conv4_1_dw_top_data);
			conv4_1_em->Forward(cublas_handle_, conv4_1_dw_top_data, conv4_1_em_top_data);
			res4->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<memory::tensor<float>>>{conv4_em_top_data, conv4_1_em_top_data}, res4_top_data);
			conv5_ex->Forward(cublas_handle_, res4_top_data, conv5_ex_top_data);
			relu5_ex->Forward_gpu_native(conv5_ex_top_data);
			conv5_dw->Forward(cublas_handle_, conv5_ex_top_data, conv5_dw_top_data);
			conv6_1->Forward_gpu_native(cublas_handle_, conv5_dw_top_data, conv6_1_top_data);
			prob1->Forward_gpu_native(conv6_1_top_data);
			conv6_2->Forward_gpu_native(cublas_handle_, conv5_dw_top_data, conv6_2_top_data);
			conv6_3->Forward_gpu_native(cublas_handle_, conv5_dw_top_data, conv6_3_top_data);
			conv6_4->Forward_gpu_native(cublas_handle_, conv5_dw_top_data, conv6_4_top_data);
		}
#ifdef USE_CUDNN
		void onet_mobile_nir::Forward_gpu_cudnn(const std::shared_ptr<memory::tensor<float>> input_data)
		{
			conv1->Forward(cudnn_handle_, input_data, conv1_top_data);
			relu1->Forward_gpu_native(conv1_top_data);
			conv1_dw->Forward(cudnn_handle_, conv1_top_data, conv1_dw_top_data);
			relu1_dw->Forward_gpu_native(conv1_dw_top_data);
			conv2_ex->Forward(cudnn_handle_, conv1_dw_top_data, conv2_ex_top_data);
			relu2_ex->Forward_gpu_native(conv2_ex_top_data);
			conv2_dw->Forward(cudnn_handle_, conv2_ex_top_data, conv2_dw_top_data);
			relu2_dw->Forward_gpu_native(conv2_dw_top_data);
			conv2_em->Forward(cudnn_handle_, conv2_dw_top_data, conv2_em_top_data);
			conv2_1_ex->Forward(cudnn_handle_, conv2_em_top_data, conv2_1_ex_top_data);
			relu2_1_ex->Forward_gpu_native(conv2_1_ex_top_data);
			conv2_1_dw->Forward(cudnn_handle_, conv2_1_ex_top_data, conv2_1_dw_top_data);
			relu2_1_dw->Forward_gpu_native(conv2_1_dw_top_data);
			conv2_1_em->Forward(cudnn_handle_, conv2_1_dw_top_data, conv2_1_em_top_data);
			res2->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<memory::tensor<float>>>{conv2_em_top_data, conv2_1_em_top_data}, res2_top_data);
			conv3_ex->Forward(cudnn_handle_, res2_top_data, conv3_ex_top_data);
			relu3_ex->Forward_gpu_native(conv3_ex_top_data);
			conv3_dw->Forward(cudnn_handle_, conv3_ex_top_data, conv3_dw_top_data);
			relu3_dw->Forward_gpu_native(conv3_dw_top_data);
			conv3_em->Forward(cudnn_handle_, conv3_dw_top_data, conv3_em_top_data);
			conv3_1_ex->Forward(cudnn_handle_, conv3_em_top_data, conv3_1_ex_top_data);
			relu3_1_ex->Forward_gpu_native(conv3_1_ex_top_data);
			conv3_1_dw->Forward(cudnn_handle_, conv3_1_ex_top_data, conv3_1_dw_top_data);
			relu3_1_dw->Forward_gpu_native(conv3_1_dw_top_data);
			conv3_1_em->Forward(cudnn_handle_, conv3_1_dw_top_data, conv3_1_em_top_data);
			res3->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<memory::tensor<float>>>{conv3_em_top_data, conv3_1_em_top_data}, res3_top_data);
			conv4_ex->Forward(cudnn_handle_, res3_top_data, conv4_ex_top_data);
			relu4_ex->Forward_gpu_native(conv4_ex_top_data);
			conv4_dw->Forward(cudnn_handle_, conv4_ex_top_data, conv4_dw_top_data);
			relu4_dw->Forward_gpu_native(conv4_dw_top_data);
			conv4_em->Forward(cudnn_handle_, conv4_dw_top_data, conv4_em_top_data);
			conv4_1_ex->Forward(cudnn_handle_, conv4_em_top_data, conv4_1_ex_top_data);
			relu4_1_ex->Forward_gpu_native(conv4_1_ex_top_data);
			conv4_1_dw->Forward(cudnn_handle_, conv4_1_ex_top_data, conv4_1_dw_top_data);
			relu4_1_dw->Forward_gpu_native(conv4_1_dw_top_data);
			conv4_1_em->Forward(cudnn_handle_, conv4_1_dw_top_data, conv4_1_em_top_data);
			res4->Forward_gpu_native(cublas_handle_, std::vector<std::shared_ptr<memory::tensor<float>>>{conv4_em_top_data, conv4_1_em_top_data}, res4_top_data);
			conv5_ex->Forward(cudnn_handle_, res4_top_data, conv5_ex_top_data);
			relu5_ex->Forward_gpu_native(conv5_ex_top_data);
			conv5_dw->Forward(cudnn_handle_, conv5_ex_top_data, conv5_dw_top_data);
			conv6_1->Forward_gpu_native(cublas_handle_, conv5_dw_top_data, conv6_1_top_data);
			prob1->Forward_gpu_native(conv6_1_top_data);
			conv6_2->Forward_gpu_native(cublas_handle_, conv5_dw_top_data, conv6_2_top_data);
			conv6_3->Forward_gpu_native(cublas_handle_, conv5_dw_top_data, conv6_3_top_data);
			conv6_4->Forward_gpu_native(cublas_handle_, conv5_dw_top_data, conv6_4_top_data);
		}
#endif
#endif

		void onet_mobile_nir::Forward(const std::shared_ptr<memory::tensor<float>> input_data)
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