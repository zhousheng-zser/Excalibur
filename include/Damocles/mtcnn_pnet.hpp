#pragma once
#ifndef _MTCNN_PNET_HPP_
#define _MTCNN_PNET_HPP_

#include "mtcnn_pnet_data.hpp"
#include "Excalibur/support_layers.hpp"
#include "Primitives/tensor.hpp"

using namespace glasssix::excalibur;

namespace glasssix
{
	namespace longinus
	{
		class mtcnn_pnet
		{
			Declear_Params(conv1)
			Declear_Params(prelu1)
			Declear_Params(conv2)
			Declear_Params(prelu2)
			Declear_Params(conv3)
			Declear_Params(prelu3)
			Declear_Params(conv4_1)
			Declear_Params(conv4_2)

			//
			Declear_Opration(baseconv, conv1)
			Declear_Opration(prelu, prelu1)
			Declear_Opration(pooling, pool1)
			Declear_Opration(baseconv, conv2)
			Declear_Opration(prelu, prelu2)
			Declear_Opration(baseconv, conv3)
			Declear_Opration(prelu, prelu3)
			Declear_Opration(baseconv, conv4_1)
			Declear_Opration(baseconv, conv4_2)
			Declear_Opration(softmax, prob1)

			//
			std::shared_ptr<memory::tensor<float>> tensor_data;
			Neuron_Name(conv1)
			Neuron_Name(pool1)
			Neuron_Name(conv2)
			Neuron_Name(conv3)
			Neuron_Name(conv4_1)
			Neuron_Name(conv4_2)
			Neuron_Name(prob1)
			//
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
			mtcnn_pnet(int device);
			~mtcnn_pnet();
			void Forward(const std::shared_ptr<memory::tensor<float>> input_data);
		};
	}
}

#endif