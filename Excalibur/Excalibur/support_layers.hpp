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
#include "slice.hpp"
#include "flip.hpp"
#include "concat.hpp"
#include "normalize.hpp"
#include "mirrormax.hpp"


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
#define Copy_Params(layer_para, netname)\
layer_para =  (float*)malloc(sizeof(netname##_##layer_para)); \
memcpy(layer_para, netname##_##layer_para, sizeof(netname##_##layer_para));
#endif

#define Init_Conv_Params(conv_name, input_channel, output_channel, kernel_size, stride, pad, bias_term) \
conv_name = new convolution(input_channel, output_channel, kernel_size, stride, pad, bias_term, device_);\
conv_name->set_weights(conv_name##_##weights);\
conv_name->set_bias(conv_name##_##bias);\

#define Init_PReLU_Params(prelu_name, input_channel, isrelu)\
prelu_name = new prelu(input_channel, isrelu, device_);\
prelu_name->setslope(prelu_name##_##weights);

#define Init_ReLU_Params(prelu_name, input_channel, isrelu)\
prelu_name = new prelu(input_channel, isrelu, device_);

#define Init_Pooling_Params(pooling_name, kernel, stride, pad, type)\
pooling_name = new pooling(kernel, stride, pad, type, device_);

#define Init_Softmax_Params(softmax_name, input_channel)\
softmax_name = new softmax(input_channel, device_);

#define Init_Eltwise_Params(eltwise_name, type)\
eltwise_name = new eltwise(type, device_);

#define Init_InnerProduct_Params(ip_name, input_channel, input_height, input_width, num_output, bias_term)\
ip_name = new inner_product(std::vector<int>{1, input_channel, input_height, input_width}, num_output, bias_term, device_);\
ip_name->set_weights(ip_name##_##weights);\
ip_name->set_bias(ip_name##_##bias);

#define Init_Flip_Params(fliper_name, flip_height, flip_width)\
fliper_name = new flip(flip_height, flip_width, device_);

#define Init_Concat_Params(concat_name, concat_axis)\
concat_name = new concat(concat_axis, device_);

#define Init_Slice_Params(slice_name, slice_axis)\
slice_name = new slice(slice_axis, device_);

#define Init_Normalize_Params(norm_name, type, rescale)\
norm_name = new normalize(type, rescale, device_);

#define Init_MirrorMax_Param(mm_name, mirror_axis)\
mm_name = new mirrormax(mirror_axis, device_);

#endif //_SUPPORT_LAYERS_HPP_