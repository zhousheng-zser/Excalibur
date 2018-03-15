#pragma once
#ifndef _MTCNN_PNET_HPP_
#define _MTCNN_PNET_HPP_
#include "mtcnn_pnet_data.hpp"
#include "../Excalibur/support_layers.hpp"

using namespace excalibur;

namespace glasssix
{
	class mtcnn_pnet
	{
		Declear_Params(conv1_weights)
		Declear_Params(conv1_bias)
		Declear_Params(prelu1_weights)
		Declear_Params(conv2_weights)
		Declear_Params(conv2_bias)
		Declear_Params(prelu2_weights)
		Declear_Params(conv3_weights)
		Declear_Params(conv3_bias)
		Declear_Params(prelu3_weights)
		Declear_Params(conv4_1_weights)
		Declear_Params(conv4_1_bias)
		Declear_Params(conv4_2_weights)
		Declear_Params(conv4_2_bias)

		//
		Declear_Opration(cudnn_convolution, conv1)
		Declear_Opration(prelu, prelu1)
		Declear_Opration(pooling, pool1)
		Declear_Opration(convolution, conv2)
		Declear_Opration(prelu, prelu2)
		Declear_Opration(convolution, conv3)
		Declear_Opration(prelu, prelu3)
		Declear_Opration(convolution, conv4_1)
		Declear_Opration(convolution, conv4_2)
		Declear_Opration(softmax, prob1)
		
		//
		std::shared_ptr<tensor> tensor_data = nullptr;
		Neuron_Name(conv1)
		Neuron_Name(pool1)
		Neuron_Name(conv2)
		Neuron_Name(conv3)
		Neuron_Name(conv4_1)
		Neuron_Name(conv4_2)
		Neuron_Name(prob1)
		//
		int device_;
		void Forward_cpu(const std::shared_ptr<tensor> input_data);
#ifdef USE_CUDA
		cublasHandle_t cublas_handle_ = nullptr;
		void Forward_native_gpu(const std::shared_ptr<tensor> input_data);
#ifdef USE_CUDNN
		cudnnHandle_t cudnn_handle_ = nullptr;
		void Forward_cudnn_gpu(const std::shared_ptr<tensor> input_data);
#endif 
#endif
		
	public:
		mtcnn_pnet(int device);
		~mtcnn_pnet();
		void Forward(const std::shared_ptr<tensor> input_data);
	};
}

#endif