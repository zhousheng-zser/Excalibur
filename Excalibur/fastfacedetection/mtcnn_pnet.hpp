#pragma once
#ifndef _MTCNN_PNET_HPP_
#define _MTCNN_PNET_HPP_
#include "mtcnn_pnet_data.hpp"
#include "../Excalibur/support_layers.hpp"
#define Neuron_Name(name) private: \
std::shared_ptr<tensor> name##_top_data = nullptr;\
public: std::shared_ptr<tensor> get_##name(){\
return name##_top_data;\
}\
private:

#define  Declear_Opration(op, name) op##* name;

#define Declear_PARAMS(layer_para) float* layer_para;

#ifdef USE_MKL
#define COPY_PARAMS(layer_para, netname) layer_para =  (float*)mkl_malloc(sizeof(netname##_##layer_para) ?sizeof(netname##_##layer_para) :1, 64); \
memcpy(layer_para, netname##_##layer_para, sizeof(netname##_##layer_para));
#else
#define COPY_PARAMS(layer_para, const_layer_param) layer_para =  (float*)malloc(sizeof(const_layer_param)); \
memcpy(layer_para, const_layer_param, sizeof(const_layer_param));
#endif

using namespace excalibur;

namespace fastface
{
	class mtcnn_pnet
	{
		Declear_PARAMS(conv1_weights);
		Declear_PARAMS(conv1_bias);
		Declear_PARAMS(prelu1_weights);
		Declear_PARAMS(conv2_weights);
		Declear_PARAMS(conv2_bias);
		Declear_PARAMS(prelu2_weights);
		Declear_PARAMS(conv3_weights);
		Declear_PARAMS(conv3_bias);
		Declear_PARAMS(prelu3_weights);
		Declear_PARAMS(conv4_1_weights);
		Declear_PARAMS(conv4_1_bias);
		Declear_PARAMS(conv4_2_weights);
		Declear_PARAMS(conv4_2_bias);

		//
		Declear_Opration(convolution, conv1)
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