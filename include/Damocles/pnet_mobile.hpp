#pragma once

#ifndef _DAILIR_PNET_MOBILE_HPP_
#define _DAILIR_PNET_MOBILE_HPP_

#include "pnet_mobile_data.hpp"
#include "Excalibur/support_layers.hpp"
#include "Primitives/tensor.hpp"

using namespace glasssix::excalibur;

namespace glasssix
{
	namespace longinus
	{
		class pnet_mobile
		{
			Declear_Params(conv1);
			Declear_Params(prelu1);
			Declear_Params(conv2_dw);
			Declear_Params(prelu2_dw);
			Declear_Params(conv2_sep);
			Declear_Params(prelu2);
			Declear_Params(conv3_dw);
			Declear_Params(prelu3_dw);
			Declear_Params(conv3_sep);
			Declear_Params(prelu3);
			Declear_Params(conv4_dw);
			Declear_Params(prelu4_dw);
			Declear_Params(conv4_1);

			Declear_Opration(baseconv, conv1);
			Neuron_Name(conv1);
			Declear_Opration(prelu, prelu1);
			Neuron_Name(prelu1);
			Declear_Opration(baseconv, conv2_dw);
			Neuron_Name(conv2_dw);
			Declear_Opration(prelu, prelu2_dw);
			Neuron_Name(prelu2_dw);
			Declear_Opration(baseconv, conv2_sep);
			Neuron_Name(conv2_sep);
			Declear_Opration(prelu, prelu2);
			Neuron_Name(prelu2);
			Declear_Opration(baseconv, conv3_dw);
			Neuron_Name(conv3_dw);
			Declear_Opration(prelu, prelu3_dw);
			Neuron_Name(prelu3_dw);
			Declear_Opration(baseconv, conv3_sep);
			Neuron_Name(conv3_sep);
			Declear_Opration(prelu, prelu3);
			Neuron_Name(prelu3);
			Declear_Opration(baseconv, conv4_dw);
			Neuron_Name(conv4_dw);
			Declear_Opration(prelu, prelu4_dw);
			Neuron_Name(prelu4_dw);
			Declear_Opration(baseconv, conv4_1);
			Neuron_Name(conv4_1);
			Declear_Opration(softmax, cls_prob);
			Neuron_Name(cls_prob);

			std::shared_ptr<memory::tensor<float>> tensor_data;

			int device_;
			bool cudnn_ready_ = false;
			bool int8_quantization_ = false;
			void Forward_cpu(const std::shared_ptr<memory::tensor<float>> input_data);
#ifdef USE_CUDA
			cublasHandle_t cublas_handle_ = nullptr;
			void Forward_gpu_native(const std::shared_ptr<memory::tensor<float>> input_data);
#ifdef USE_CUDNN
			cudnnHandle_t cudnn_handle_ = nullptr;
			void Forward_gpu_cudnn(const std::shared_ptr<memory::tensor<float>> input_data);
#endif 
#endif

		public:
			pnet_mobile(int device);
			~pnet_mobile();
			void Forward(const std::shared_ptr<memory::tensor<float>> input_data);
		};
	}
}

#endif