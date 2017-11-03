#pragma once
#ifndef _IPTS_NET_HPP_
#define _IPTS_NET_HPP_

#include "ipts_v2_data.hpp"
#include "../Excalibur/support_layers.hpp"

using namespace excalibur;

namespace glasssix
{
	class ipts_net
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
		Declear_Params(fc2_weights);
		Declear_Params(fc2_bias);
		//
		int device_;
		std::shared_ptr<tensor> tensor_data = nullptr;
		//
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
		Declear_Opration(inner_product, fc2);
		Neuron_Name(fc2);

#ifdef USE_CUDA
		cublasHandle_t cublas_handle_ = nullptr;
		void Forward_native_gpu(const std::shared_ptr<tensor> input_data);
#ifdef USE_CUDNN
		void Forward_cudnn_gpu(const std::shared_ptr<tensor> input_data);
#endif 
#endif
		void Forward_cpu(const std::shared_ptr<tensor> input_data);

	public:
		ipts_net(int device);
		~ipts_net();
		
		void Forward(const std::shared_ptr<tensor> input_data);

		static int get_input_channel()
		{
			return 3;
		}
		static int get_input_width()
		{
			return 60;
		}
		static int get_input_height()
		{
			return 60;
		}
	};
}

#endif