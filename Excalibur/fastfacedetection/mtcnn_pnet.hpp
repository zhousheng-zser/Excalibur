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

#define Declear_Params(layer_para) float* layer_para;

#ifdef USE_MKL
#define Copy_Params(layer_para, netname)\
layer_para =  (float*)mkl_malloc(sizeof(netname##_##layer_para) ?sizeof(netname##_##layer_para) :1, 64); \
memcpy(layer_para, netname##_##layer_para, sizeof(netname##_##layer_para));
#else
#define COPY_PARAMS(layer_para, const_layer_param)\
layer_para =  (float*)malloc(sizeof(const_layer_param)); \
memcpy(layer_para, const_layer_param, sizeof(const_layer_param));
#endif

#define Declear_Conv_Params(conv_name, input_channel, output_channel, kernel_size, stride, pad, bias_term) \
conv_name = new convolution(input_channel, output_channel, kernel_size, stride, pad, bias_term, device_);\
conv_name->set_weights(conv_name##_##weights);\
conv_name->set_bias(conv_name##_##bias);\

#define Declear_PReLU_Params(prelu_name, input_channel, isrelu)\
prelu_name = new prelu(input_channel, isrelu, device_);\
prelu_name->setslope(prelu_name##_##weights);

#define Declear_Pooling_Params(pooling_name, kernel, stride, pad, type)\
pooling_name = new pooling(kernel, stride, pad, type, device_);

#define Declear_Softmax_Params(softmax_name, input_channel)\
softmax_name = new softmax(input_channel, device_);

using namespace excalibur;

namespace fastface
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