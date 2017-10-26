#pragma once
#ifndef _SUPPORT_LAYERS_HPP_
#define _SUPPORT_LAYERS_HPP_
#include "convolution.hpp"
#include "cudnn_convolution.hpp"
#include "prelu.hpp"
#include "pooling.hpp"
#include "eltwise.hpp"
#include "inner_product.hpp"
#include "softmax.hpp"


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

#endif //_SUPPORT_LAYERS_HPP_