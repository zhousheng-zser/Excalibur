#pragma once
#ifndef _MTCNN_ONET_HPP_
#define _MTCNN_ONET_HPP_
#include "../Excalibur/io.hpp"
#include "../Excalibur/support_layers.hpp"
#include "mtcnn_onet_data.hpp"

using namespace excalibur;

namespace glasssix
{
	class mtcnn_onet
	{
		Declear_Params(conv1_weights);
		Declear_Params(conv1_bias);
		Declear_Params(prelu1_weights);
		Declear_Params(conv2_weights);
		Declear_Params(conv2_bias);
		Declear_Params(prelu2_weights);
		Declear_Params(conv3_weights);
		Declear_Params(conv3_bias);
		Declear_Params(prelu3_weights);
		Declear_Params(conv4_weights);
		Declear_Params(conv4_bias);
		Declear_Params(prelu4_weights);
		Declear_Params(conv5_weights);
		Declear_Params(conv5_bias);
		Declear_Params(prelu5_weights);
		Declear_Params(conv6_1_weights);
		Declear_Params(conv6_1_bias);
		Declear_Params(conv6_2_weights);
		Declear_Params(conv6_2_bias);
		Declear_Params(conv6_3_weights);
		Declear_Params(conv6_3_bias);
		//
		std::shared_ptr<tensor> tensor_data = nullptr;
		Declear_Opration(convolution, conv1);
		Neuron_Name(conv1);
		Declear_Opration(prelu, prelu1);
		Neuron_Name(prelu1);
		Declear_Opration(pooling, pool1);
		Neuron_Name(pool1);
		Declear_Opration(convolution, conv2);
		Neuron_Name(conv2);
		Declear_Opration(prelu, prelu2);
		Neuron_Name(prelu2);
		Declear_Opration(pooling, pool2);
		Neuron_Name(pool2);
		Declear_Opration(convolution, conv3);
		Neuron_Name(conv3);
		Declear_Opration(prelu, prelu3);
		Neuron_Name(prelu3);
		Declear_Opration(pooling, pool3);
		Neuron_Name(pool3);
		Declear_Opration(convolution, conv4);
		Neuron_Name(conv4);
		Declear_Opration(prelu, prelu4);
		Neuron_Name(prelu4);
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
		Declear_Opration(softmax, prob1);
		Neuron_Name(prob1);
		//
		int device_;
		void Forward_cpu(const std::shared_ptr<tensor> input_data);
#ifdef USE_CUDA
		cublasHandle_t cublas_handle_ = nullptr;
		void Forward_native_gpu(const std::shared_ptr<tensor> input_data);
#ifdef USE_CUDNN
		void Forward_cudnn_gpu(const std::shared_ptr<tensor> input_data);
#endif 
#endif
	public:
		mtcnn_onet(int device);
		~mtcnn_onet();
		void Forward(const std::shared_ptr<tensor> input_data);
	};
}

#endif