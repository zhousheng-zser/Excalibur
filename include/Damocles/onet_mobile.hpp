#ifndef _DAILIR_ONET_HPP_
#define _DAILIR_ONET_HPP_
#include "../Excalibur/support_layers.hpp"
#include "onet_mobile_data.hpp"

using namespace glasssix::excalibur;

namespace glasssix
{
	namespace longinus
	{
		class onet_mobile
		{
			Declear_Params(conv1);
			Declear_Params(prelu1);
			Declear_Params(conv1_dw);
			Declear_Params(prelu1_dw);
			Declear_Params(conv2);
			Declear_Params(conv2_dw);
			Declear_Params(prelu2_dw);
			Declear_Params(conv3);
			Declear_Params(conv3_dw);
			Declear_Params(prelu3_dw);
			Declear_Params(conv4);
			Declear_Params(conv4_dw);
			Declear_Params(prelu4_dw);
			Declear_Params(conv5);
			Declear_Params(prelu5);
			Declear_Params(conv6_1);
			Declear_Params(conv6_2);
			Declear_Params(conv6_3);
			Declear_Params(conv6_4);
			//
			std::shared_ptr<tensor<float>> tensor_data;

			Declear_Opration(baseconv, conv1);
			Neuron_Name(conv1);
			Declear_Opration(prelu, prelu1);
			Neuron_Name(prelu1);
			Declear_Opration(baseconv, conv1_dw);
			Neuron_Name(conv1_dw);
			Declear_Opration(prelu, prelu1_dw);
			Neuron_Name(prelu1_dw);
			Declear_Opration(baseconv, conv2);
			Neuron_Name(conv2);
			Declear_Opration(baseconv, conv2_dw);
			Neuron_Name(conv2_dw);
			Declear_Opration(prelu, prelu2_dw);
			Neuron_Name(prelu2_dw);
			Declear_Opration(baseconv, conv3);
			Neuron_Name(conv3);
			Declear_Opration(baseconv, conv3_dw);
			Neuron_Name(conv3_dw);
			Declear_Opration(prelu, prelu3_dw);
			Neuron_Name(prelu3_dw);
			Declear_Opration(baseconv, conv4);
			Neuron_Name(conv4);
			Declear_Opration(baseconv, conv4_dw);
			Neuron_Name(conv4_dw);
			Declear_Opration(prelu, prelu4_dw);
			Neuron_Name(prelu4_dw);
			Declear_Opration(inner_product, conv5);
			Neuron_Name(conv5);
			Declear_Opration(prelu, prelu5);
			Neuron_Name(prelu5);
			Declear_Opration(inner_product, conv6_1);
			Neuron_Name(conv6_1);
			Declear_Opration(inner_product, conv6_2);
			Neuron_Name(conv6_2);
			Declear_Opration(inner_product, conv6_3);
			Neuron_Name(conv6_3);
			Declear_Opration(inner_product, conv6_4);
			Neuron_Name(conv6_4);
			Declear_Opration(sigmoid, prob1);
			Neuron_Name(prob1);
			//
			int device_;
			bool cudnn_ready_ = false;
			bool int8_quantization_ = false;
			void Forward_cpu(const std::shared_ptr<tensor<float>> input_data);
#ifdef USE_CUDA
			cublasHandle_t cublas_handle_ = nullptr;
			void Forward_gpu_native(const std::shared_ptr<tensor<float>> input_data);
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle_ = nullptr;
			void Forward_gpu_cudnn(const std::shared_ptr<tensor<float>> input_data);
#endif 
#endif
		public:
			onet_mobile(int device);
			~onet_mobile();
			void Forward(const std::shared_ptr<tensor<float>> input_data);
		};
	}
}

#endif