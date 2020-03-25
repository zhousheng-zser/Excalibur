#include "deconv_net.hpp"
#include "deconv_net_data.hpp"
#include <iostream>
#include <vector>
#include "../../include/Julius/simd_helper.hpp"
#include "../../include/Excalibur/conv_winograd_cpu.hpp"

using namespace glasssix::excalibur;

namespace glasssix
{
	namespace longinus
	{
		deconvT::deconvT(int device)
		{
			device_ = device;
			conv_native_cpu *tt = new conv_native_cpu(12, 12, 1, 3, 1, 1, true, -1, false);
			conv_winograd_cpu *gg = new conv_winograd_cpu(12, 12, 1, 3, 1, 1, true, -1, false);
			if (device_ >= 0)
			{
				int8_quantization_ = false;//do not use int8 in GPU mode
			}

#if SIMD_TYPE >= SIMDTYPE_SSE
			std::shared_ptr<tensor<float>> bottom_round_ = std::make_shared<tensor<float>>(std::vector<int>{mm_align_size});
			float* bottom_round_data_ = bottom_round_->mutable_cpu_data();
#endif // SIMD_TYPE >= SIMDTYPE_SSE
			
			float quantize_level = INT_MAX;

			if (int8_quantization_)
			{
				Copy_Int8_Params(conv1, deconv_net);
				Copy_Params(relu1_weights, deconv_net, quantize_level);
				Copy_Int8_Params(conv2, deconv_net);
				Copy_Params(relu2_weights, deconv_net, quantize_level);
				Copy_Int8_Params(conv22, deconv_net);
				Copy_Params(relu22_weights, deconv_net, quantize_level);
				Copy_Int8_Params(conv23, deconv_net);
				Copy_Params(relu23_weights, deconv_net, quantize_level);
				Copy_Int8_Params(conv24, deconv_net);
				Copy_Params(relu24_weights, deconv_net, quantize_level);
				Copy_Int8_Params(conv25, deconv_net);
				Copy_Params(relu25_weights, deconv_net, quantize_level);
				Copy_Int8_Params(conv26, deconv_net);
				Copy_Params(relu26_weights, deconv_net, quantize_level);
				Copy_Params(conv3_weights, deconv_net, quantize_level);
				Copy_Params(conv3_bias, deconv_net, quantize_level);
			}
			else
			{
				Copy_Params(conv1_weights, deconv_net, quantize_level);
				Copy_Params(conv1_bias, deconv_net, quantize_level);
				Copy_Params(relu1_weights, deconv_net, quantize_level);
				Copy_Params(conv2_weights, deconv_net, quantize_level);
				Copy_Params(conv2_bias, deconv_net, quantize_level);
				Copy_Params(relu2_weights, deconv_net, quantize_level);
				Copy_Params(conv22_weights, deconv_net, quantize_level);
				Copy_Params(conv22_bias, deconv_net, quantize_level);
				Copy_Params(relu22_weights, deconv_net, quantize_level);
				Copy_Params(conv23_weights, deconv_net, quantize_level);
				Copy_Params(conv23_bias, deconv_net, quantize_level);
				Copy_Params(relu23_weights, deconv_net, quantize_level);
				Copy_Params(conv24_weights, deconv_net, quantize_level);
				Copy_Params(conv24_bias, deconv_net, quantize_level);
				Copy_Params(relu24_weights, deconv_net, quantize_level);
				Copy_Params(conv25_weights, deconv_net, quantize_level);
				Copy_Params(conv25_bias, deconv_net, quantize_level);
				Copy_Params(relu25_weights, deconv_net, quantize_level);
				Copy_Params(conv26_weights, deconv_net, quantize_level);
				Copy_Params(conv26_bias, deconv_net, quantize_level);
				Copy_Params(relu26_weights, deconv_net, quantize_level);
				Copy_Params(conv3_weights, deconv_net, quantize_level);
				Copy_Params(conv3_bias, deconv_net, quantize_level);
			}
			

#ifdef USE_CUDA
			if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS)
			{
				LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
			}

#ifdef USE_CUDNN
			if (cudnnCreate(&cudnn_handle_) != CUDNN_STATUS_SUCCESS)
			{
				LOG(ERROR) << "Cannot create Cudnn handle. Cudnn won't be available.";
			}
			cudnn_ready_ = true;
#endif // USE_CUDNN
#endif

			

			Init_Conv_Params(conv1, 1, 56, 1, 5, 1, 0, true);
			Init_PReLU_Shared_Params(relu1, 56, false, device_, true);
			Init_Conv_Params(conv2, 56, 12, 1, 1, 1, 0, true);
			Init_PReLU_Shared_Params(relu2, 12, false, device_, true);
			Init_Conv_Params(conv22, 12, 12, 1, 3, 1, 1, true);
			Init_PReLU_Shared_Params(relu22, 12, false, device_, true);
			Init_Conv_Params(conv23, 12, 12, 1, 3, 1, 1, true);
			Init_PReLU_Shared_Params(relu23, 12, false, device_, true);
			Init_Eltwise_Params(res2_3, 0);
			Init_Conv_Params(conv24, 12, 12, 1, 3, 1, 1, true);
			Init_PReLU_Shared_Params(relu24, 12, false, device_, true);
			Init_Conv_Params(conv25, 12, 12, 1, 3, 1, 1, true);
			Init_PReLU_Shared_Params(relu25, 12, false, device_, true);
			Init_Eltwise_Params(res2_5, 0);
			Init_Conv_Params(conv26, 12, 56, 1, 1, 1, 0, true);
			Init_PReLU_Shared_Params(relu26, 56, false, device_, true);
			Init_Deconv_Params(conv3, 56, 1, 1, 9, 3, 4, true);
		}


		deconvT::~deconvT()
		{
			delete conv1;
			delete relu1;
			delete conv2;
			delete relu2;
			delete conv22;
			delete relu22;
			delete conv23;
			delete relu23;
			delete res2_3;
			delete conv24;
			delete relu24;
			delete conv25;
			delete relu25;
			delete res2_5;
			delete conv26;
			delete relu26;
			delete conv3;

			FreeHost(relu1_weights, false);
			FreeHost(relu2_weights, false);
			FreeHost(relu22_weights, false);
			FreeHost(relu23_weights, false);
			FreeHost(relu24_weights, false);
			FreeHost(relu25_weights, false);
			FreeHost(relu26_weights, false);

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

		void deconvT::Forward_cpu(const std::shared_ptr<tensor<float>> input_data)
		{
			conv1->Forward(input_data, conv1_top_data);
			relu1->Forward_cpu(conv1_top_data);
			conv2->Forward(conv1_top_data, conv2_top_data);
			relu2->Forward_cpu(conv2_top_data);
			conv22->Forward(conv2_top_data, conv22_top_data);
			relu22->Forward_cpu(conv22_top_data);
			conv23->Forward(conv22_top_data, conv23_top_data);
			relu23->Forward_cpu(conv23_top_data);
			res2_3->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{conv2_top_data, conv23_top_data}, res2_3_top_data);
			conv24->Forward(res2_3_top_data, conv24_top_data);
			relu24->Forward_cpu(conv24_top_data);
			conv25->Forward(conv24_top_data, conv25_top_data);
			relu25->Forward_cpu(conv25_top_data);
			res2_5->Forward_cpu(std::vector<std::shared_ptr<tensor<float>>>{res2_3_top_data, conv25_top_data}, res2_5_top_data);
			conv26->Forward(res2_5_top_data, conv26_top_data);
			relu26->Forward_cpu(conv26_top_data);
			conv3->Forward(conv26_top_data, conv3_top_data);
		}

#ifdef USE_CUDA
		void Banshee::Forward_gpu_native(const std::shared_ptr<tensor<float>> input_data)
		{
#ifdef _DEBUG
			CHECK_EQ(input_data->width(), 48);
			CHECK_EQ(input_data->height(), 48);
			CHECK_EQ(input_data->channels(), 1);
#endif
			conv1->Forward(cublas_handle_, input_data, conv1_top_data);
			prelu1->Forward_gpu_native(conv1_top_data);
			conv1_dw->Forward(cublas_handle_, conv1_top_data, conv1_dw_top_data);
			prelu1_dw->Forward_gpu_native(conv1_dw_top_data);
			conv2->Forward(cublas_handle_, conv1_dw_top_data, conv2_top_data);
			conv2_dw->Forward(cublas_handle_, conv2_top_data, conv2_dw_top_data);
			prelu2_dw->Forward_gpu_native(conv2_dw_top_data);
			conv3->Forward(cublas_handle_, conv2_dw_top_data, conv3_top_data);
			conv3_dw->Forward(cublas_handle_, conv3_top_data, conv3_dw_top_data);
			prelu3_dw->Forward_gpu_native(conv3_dw_top_data);
			conv4->Forward(cublas_handle_, conv3_dw_top_data, conv4_top_data);
			conv4_dw->Forward(cublas_handle_, conv4_top_data, conv4_dw_top_data);
			prelu4_dw->Forward_gpu_native(conv4_dw_top_data);
			conv5->Forward_gpu_native(cublas_handle_, conv4_dw_top_data, conv5_top_data);
			prelu5->Forward_gpu_native(conv5_top_data);
			conv6_1->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_1_top_data);
			sigmoid1->Forward_gpu_native(conv6_1_top_data);
			conv6_2->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_2_top_data);
			conv6_3->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_3_top_data);
		}

#ifdef USE_CUDNN
		void Banshee::Forward_gpu_cudnn(const std::shared_ptr<tensor<float>> input_data)
		{
#ifdef _DEBUG
			CHECK_EQ(input_data->width(), 48);
			CHECK_EQ(input_data->height(), 48);
			CHECK_EQ(input_data->channels(), 1);
#endif
			conv1->Forward(cudnn_handle_, input_data, conv1_top_data);
			prelu1->Forward_gpu_native(conv1_top_data);
			conv1_dw->Forward(cudnn_handle_, conv1_top_data, conv1_dw_top_data);
			prelu1_dw->Forward_gpu_native(conv1_dw_top_data);
			conv2->Forward(cudnn_handle_, conv1_dw_top_data, conv2_top_data);
			conv2_dw->Forward(cudnn_handle_, conv2_top_data, conv2_dw_top_data);
			prelu2_dw->Forward_gpu_native(conv2_dw_top_data);
			conv3->Forward(cudnn_handle_, conv2_dw_top_data, conv3_top_data);
			conv3_dw->Forward(cudnn_handle_, conv3_top_data, conv3_dw_top_data);
			prelu3_dw->Forward_gpu_native(conv3_dw_top_data);
			conv4->Forward(cudnn_handle_, conv3_dw_top_data, conv4_top_data);
			conv4_dw->Forward(cudnn_handle_, conv4_top_data, conv4_dw_top_data);
			prelu4_dw->Forward_gpu_native(conv4_dw_top_data);
			conv5->Forward_gpu_native(cublas_handle_, conv4_dw_top_data, conv5_top_data);
			prelu5->Forward_gpu_native(conv5_top_data);
			conv6_1->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_1_top_data);
			sigmoid1->Forward_gpu_native(conv6_1_top_data);
			conv6_2->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_2_top_data);
			conv6_3->Forward_gpu_native(cublas_handle_, conv5_top_data, conv6_3_top_data);
		}
#endif 
#endif

	}
}