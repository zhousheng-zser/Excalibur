#pragma once
#ifndef _UNICORN_NET_HPP_
#define _UNICORN_NET_HPP_

#include "unicorn_data.hpp"
#include "../Excalibur/support_layers.hpp"

using namespace excalibur;

namespace glasssix
{
	class unicorn_net
	{
		Declear_Params(conv1a_weights);
		Declear_Params(conv1a_bias);
		Declear_Params(relu1a_weights);
		Declear_Params(conv1b_weights);
		Declear_Params(conv1b_bias);
		Declear_Params(relu1b_weights);
		Declear_Params(conv2_1_weights);
		Declear_Params(conv2_1_bias);
		Declear_Params(relu2_1_weights);
		Declear_Params(conv2_2_weights);
		Declear_Params(conv2_2_bias);
		Declear_Params(relu2_2_weights);
		Declear_Params(conv2_weights);
		Declear_Params(conv2_bias);
		Declear_Params(relu2_weights);
		Declear_Params(conv3_1_weights);
		Declear_Params(conv3_1_bias);
		Declear_Params(relu3_1_weights);
		Declear_Params(conv3_2_weights);
		Declear_Params(conv3_2_bias);
		Declear_Params(relu3_2_weights);
		Declear_Params(conv3_3_weights);
		Declear_Params(conv3_3_bias);
		Declear_Params(relu3_3_weights);
		Declear_Params(conv3_4_weights);
		Declear_Params(conv3_4_bias);
		Declear_Params(relu3_4_weights);
		Declear_Params(conv3_weights);
		Declear_Params(conv3_bias);
		Declear_Params(relu3_weights);
		Declear_Params(conv4_1_weights);
		Declear_Params(conv4_1_bias);
		Declear_Params(relu4_1_weights);
		Declear_Params(conv4_2_weights);
		Declear_Params(conv4_2_bias);
		Declear_Params(relu4_2_weights);
		Declear_Params(conv4_3_weights);
		Declear_Params(conv4_3_bias);
		Declear_Params(relu4_3_weights);
		Declear_Params(conv4_4_weights);
		Declear_Params(conv4_4_bias);
		Declear_Params(relu4_4_weights);
		Declear_Params(conv4_5_weights);
		Declear_Params(conv4_5_bias);
		Declear_Params(relu4_5_weights);
		Declear_Params(conv4_6_weights);
		Declear_Params(conv4_6_bias);
		Declear_Params(relu4_6_weights);
		Declear_Params(conv4_7_weights);
		Declear_Params(conv4_7_bias);
		Declear_Params(relu4_7_weights);
		Declear_Params(conv4_8_weights);
		Declear_Params(conv4_8_bias);
		Declear_Params(relu4_8_weights);
		Declear_Params(conv4_9_weights);
		Declear_Params(conv4_9_bias);
		Declear_Params(relu4_9_weights);
		Declear_Params(conv4_10_weights);
		Declear_Params(conv4_10_bias);
		Declear_Params(relu4_10_weights);
		Declear_Params(conv4_weights);
		Declear_Params(conv4_bias);
		Declear_Params(relu4_weights);
		Declear_Params(conv5_1_weights);
		Declear_Params(conv5_1_bias);
		Declear_Params(relu5_1_weights);
		Declear_Params(conv5_2_weights);
		Declear_Params(conv5_2_bias);
		Declear_Params(relu5_2_weights);
		Declear_Params(conv5_3_weights);
		Declear_Params(conv5_3_bias);
		Declear_Params(relu5_3_weights);
		Declear_Params(conv5_4_weights);
		Declear_Params(conv5_4_bias);
		Declear_Params(relu5_4_weights);
		Declear_Params(conv5_5_weights);
		Declear_Params(conv5_5_bias);
		Declear_Params(relu5_5_weights);
		Declear_Params(conv5_6_weights);
		Declear_Params(conv5_6_bias);
		Declear_Params(relu5_6_weights);
		Declear_Params(conv5_weights);
		Declear_Params(conv5_bias);
		Declear_Params(relu5_weights);
		//
		int device_;
		std::shared_ptr<tensor> tensor_data = nullptr;
		//
		Declear_Opration(convolution, conv1a);
		Neuron_Name(conv1a);
		Declear_Opration(prelu, relu1a);
		Neuron_Name(relu1a);
		Declear_Opration(convolution, conv1b);
		Neuron_Name(conv1b);
		Declear_Opration(prelu, relu1b);
		Neuron_Name(relu1b);
		Declear_Opration(pooling, pool1b);
		Neuron_Name(pool1b);
		Declear_Opration(convolution, conv2_1);
		Neuron_Name(conv2_1);
		Declear_Opration(prelu, relu2_1);
		Neuron_Name(relu2_1);
		Declear_Opration(convolution, conv2_2);
		Neuron_Name(conv2_2);
		Declear_Opration(prelu, relu2_2);
		Neuron_Name(relu2_2);
		Declear_Opration(eltwise, res2_2);
		Neuron_Name(res2_2);
		Declear_Opration(convolution, conv2);
		Neuron_Name(conv2);
		Declear_Opration(prelu, relu2);
		Neuron_Name(relu2);
		Declear_Opration(pooling, pool2);
		Neuron_Name(pool2);
		Declear_Opration(convolution, conv3_1);
		Neuron_Name(conv3_1);
		Declear_Opration(prelu, relu3_1);
		Neuron_Name(relu3_1);
		Declear_Opration(convolution, conv3_2);
		Neuron_Name(conv3_2);
		Declear_Opration(prelu, relu3_2);
		Neuron_Name(relu3_2);
		Declear_Opration(eltwise, res3_2);
		Neuron_Name(res3_2);
		Declear_Opration(convolution, conv3_3);
		Neuron_Name(conv3_3);
		Declear_Opration(prelu, relu3_3);
		Neuron_Name(relu3_3);
		Declear_Opration(convolution, conv3_4);
		Neuron_Name(conv3_4);
		Declear_Opration(prelu, relu3_4);
		Neuron_Name(relu3_4);
		Declear_Opration(eltwise, res3_4);
		Neuron_Name(res3_4);
		Declear_Opration(convolution, conv3);
		Neuron_Name(conv3);
		Declear_Opration(prelu, relu3);
		Neuron_Name(relu3);
		Declear_Opration(pooling, pool3);
		Neuron_Name(pool3);
		Declear_Opration(convolution, conv4_1);
		Neuron_Name(conv4_1);
		Declear_Opration(prelu, relu4_1);
		Neuron_Name(relu4_1);
		Declear_Opration(convolution, conv4_2);
		Neuron_Name(conv4_2);
		Declear_Opration(prelu, relu4_2);
		Neuron_Name(relu4_2);
		Declear_Opration(eltwise, res4_2);
		Neuron_Name(res4_2);
		Declear_Opration(convolution, conv4_3);
		Neuron_Name(conv4_3);
		Declear_Opration(prelu, relu4_3);
		Neuron_Name(relu4_3);
		Declear_Opration(convolution, conv4_4);
		Neuron_Name(conv4_4);
		Declear_Opration(prelu, relu4_4);
		Neuron_Name(relu4_4);
		Declear_Opration(eltwise, res4_4);
		Neuron_Name(res4_4);
		Declear_Opration(convolution, conv4_5);
		Neuron_Name(conv4_5);
		Declear_Opration(prelu, relu4_5);
		Neuron_Name(relu4_5);
		Declear_Opration(convolution, conv4_6);
		Neuron_Name(conv4_6);
		Declear_Opration(prelu, relu4_6);
		Neuron_Name(relu4_6);
		Declear_Opration(eltwise, res4_6);
		Neuron_Name(res4_6);
		Declear_Opration(convolution, conv4_7);
		Neuron_Name(conv4_7);
		Declear_Opration(prelu, relu4_7);
		Neuron_Name(relu4_7);
		Declear_Opration(convolution, conv4_8);
		Neuron_Name(conv4_8);
		Declear_Opration(prelu, relu4_8);
		Neuron_Name(relu4_8);
		Declear_Opration(eltwise, res4_8);
		Neuron_Name(res4_8);
		Declear_Opration(convolution, conv4_9);
		Neuron_Name(conv4_9);
		Declear_Opration(prelu, relu4_9);
		Neuron_Name(relu4_9);
		Declear_Opration(convolution, conv4_10);
		Neuron_Name(conv4_10);
		Declear_Opration(prelu, relu4_10);
		Neuron_Name(relu4_10);
		Declear_Opration(eltwise, res4_10);
		Neuron_Name(res4_10);
		Declear_Opration(convolution, conv4);
		Neuron_Name(conv4);
		Declear_Opration(prelu, relu4);
		Neuron_Name(relu4);
		Declear_Opration(pooling, pool4);
		Neuron_Name(pool4);
		Declear_Opration(convolution, conv5_1);
		Neuron_Name(conv5_1);
		Declear_Opration(prelu, relu5_1);
		Neuron_Name(relu5_1);
		Declear_Opration(convolution, conv5_2);
		Neuron_Name(conv5_2);
		Declear_Opration(prelu, relu5_2);
		Neuron_Name(relu5_2);
		Declear_Opration(eltwise, res5_2);
		Neuron_Name(res5_2);
		Declear_Opration(convolution, conv5_3);
		Neuron_Name(conv5_3);
		Declear_Opration(prelu, relu5_3);
		Neuron_Name(relu5_3);
		Declear_Opration(convolution, conv5_4);
		Neuron_Name(conv5_4);
		Declear_Opration(prelu, relu5_4);
		Neuron_Name(relu5_4);
		Declear_Opration(eltwise, res5_4);
		Neuron_Name(res5_4);
		Declear_Opration(convolution, conv5_5);
		Neuron_Name(conv5_5);
		Declear_Opration(prelu, relu5_5);
		Neuron_Name(relu5_5);
		Declear_Opration(convolution, conv5_6);
		Neuron_Name(conv5_6);
		Declear_Opration(prelu, relu5_6);
		Neuron_Name(relu5_6);
		Declear_Opration(eltwise, res5_6);
		Neuron_Name(res5_6);
		Declear_Opration(convolution, conv5);
		Neuron_Name(conv5);
		Declear_Opration(prelu, relu5);
		Neuron_Name(relu5);
		Declear_Opration(pooling, pool5);
		Neuron_Name(pool5);
		
#ifdef USE_CUDA
		cublasHandle_t cublas_handle_ = nullptr;
		void Forward_native_gpu(const std::shared_ptr<tensor> input_data);
#ifdef USE_CUDNN
		void Forward_cudnn_gpu(const std::shared_ptr<tensor> input_data);
#endif 
#endif
		void Forward_cpu(const std::shared_ptr<tensor> input_data);
	public:
		unicorn_net(int device);
		~unicorn_net();

		void Forward(const std::shared_ptr<tensor> input_data);

		static int get_input_channel()
		{
			return 3;
		}

		static int get_input_width()
		{
			return 128;
		}

		static int get_input_height()
		{
			return 128;
		}
	};
}

#endif