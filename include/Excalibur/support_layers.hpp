#pragma once
#ifndef _SUPPORT_LAYERS_HPP_
#define _SUPPORT_LAYERS_HPP_
#include "../../include/Julius/simd_helper.hpp"
#include "base_conv.hpp"
#include "conv_cudnn_gpu.hpp"
#include "conv_native_cpu.hpp"
#include "conv_native_gpu.hpp"
#include "conv_winograd_cpu.hpp"
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
#include "sigmoid.hpp"
#include "deconv.hpp"


#define Neuron_Name(name) private: \
std::shared_ptr<tensor<float>> name##_top_data;\
public: std::shared_ptr<tensor<float>> get_##name(){\
return name##_top_data;\
}\
private:

#define  Declear_Opration(op, name) op##* name;

#define Declear_Params(layername) float *layername##_##bias, *layername##_##weights, *layername##_##scales;\
signed char *layername##_##weights_int8;

#ifdef USE_MKL
#define Copy_Params(layer_para, netname, datatype)\
if(datatype == INT_MAX){\
layer_para =  (float*)mkl_malloc(sizeof(netname##_##layer_para) ? sizeof(netname##_##layer_para) :1, 64); \
memcpy(layer_para, netname##_##layer_para, sizeof(netname##_##layer_para));}\
if(datatype == USHRT_MAX){\
layer_para =  (float*)mkl_malloc(sizeof(netname##_##layer_para) ? sizeof(netname##_##layer_para) / sizeof(unsigned short) * sizeof(float) :1, 64); \
half2float((unsigned short*)netname##_##layer_para,layer_para,sizeof(netname##_##layer_para) / sizeof(unsigned short));}
#else
#define Copy_Params(layer_para, netname, datatype)\
if(datatype == INT_MAX){\
layer_para =  (float*)_aligned_malloc(sizeof(netname##_##layer_para), MALLOC_ALIGN); \
memcpy(layer_para, netname##_##layer_para, sizeof(netname##_##layer_para));}\
if(datatype == USHRT_MAX) {\
layer_para =  (float*)_aligned_malloc(sizeof(netname##_##layer_para) / sizeof(unsigned short) * sizeof(float), MALLOC_ALIGN); \
half2float((unsigned short*)netname##_##layer_para,layer_para,sizeof(netname##_##layer_para) / sizeof(unsigned short));}
#endif



#define Copy_Int8_FP32_Params(layername, netname)\
layername##_##weights =  (float*)_aligned_malloc(sizeof(netname##_##layername##_##weights) / sizeof(signed char) * sizeof(float), MALLOC_ALIGN); \
int8_to_float((const signed char*)netname##_##layername##_##weights,(const float*)netname##_##layername##_##scales_weight,(float*)layername##_##weights,\
    sizeof(netname##_##layername##_##weights) / sizeof(signed char),sizeof(netname##_##layername##_##scales_weight) / sizeof(float));


#ifdef INT8_DATA //copy directely, do not caculate

#define Copy_Int8_Params(layername, netname)\
layername##_##bias =  (float*)_aligned_malloc(sizeof(netname##_##layername##_##bias), MALLOC_ALIGN); \
memcpy(layername##_##bias, netname##_##layername##_##bias, sizeof(netname##_##layername##_##bias));\
layername##_##weights_int8 =  (signed char*)_aligned_malloc(sizeof(netname##_##layername##_##weights), MALLOC_ALIGN); \
memcpy(layername##_##weights_int8, netname##_##layername##_##weights, sizeof(netname##_##layername##_##weights));\
layername##_##scales =  (float*)_aligned_malloc(sizeof(netname##_##layername##_##scales_bottom) + sizeof(netname##_##layername##_##scales_weight), MALLOC_ALIGN); \
layername##_##scales[0] =  netname##_##layername##_##scales_bottom[0];\
for (int i = 0; i < sizeof(netname##_##layername##_##scales_weight) / sizeof(float); i++) {\
layername##_##scales[i + 1] = netname##_##layername##_##scales_weight[i];}

#else

#if SIMD_TYPE >= SIMDTYPE_SSE

#define Copy_Int8_Params(layername, netname)\
layername##_##bias =  (float*)_aligned_malloc(sizeof(netname##_##layername##_##bias), MALLOC_ALIGN); \
memcpy(layername##_##bias, netname##_##layername##_##bias, sizeof(netname##_##layername##_##bias));\
layername##_##weights_int8 =  (signed char*)_aligned_malloc(sizeof(netname##_##layername##_##weights) / sizeof(float) * sizeof(signed char), MALLOC_ALIGN); \
    for (int j = 0, num_weights = sizeof(netname##_##layername##_##weights) / sizeof(float), group = sizeof(netname##_##layername##_##scales_weight) / sizeof(float); j < group; j++){\
        int offset = num_weights / group;\
        mm_type scale = mm_set1_ps(netname##_##layername##_##scales_weight[j]);\
        int circle_num = offset / mm_align_size;\
        int index = 0;\
		for (; index < circle_num; index++){\
			int index_offset = index * mm_align_size;\
			mm_type data = mm_load_ps(const_cast<float*>(netname##_##layername##_##weights + j * offset + index_offset));\
			mm_type res_mul = mm_mul_ps(data, scale);\
			mm_type res_round = mm_round_ps(res_mul, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);\
			mm_store_ps(bottom_round_data_, res_round);\
			for (int k = 0; k < mm_align_size; k++){\
				layername##_##weights_int8[j * offset + index_offset + k] = (signed char)(bottom_round_data_[k]);}}\
			for (index = mm_align_size * index; index < offset; index++){\
				layername##_##weights_int8[j * offset + index] = round(netname##_##layername##_##weights[j * offset + index] * netname##_##layername##_##scales_weight[j]);}}\
layername##_##scales =  (float*)_aligned_malloc(sizeof(netname##_##layername##_##scales_bottom) + sizeof(netname##_##layername##_##scales_weight), MALLOC_ALIGN); \
layername##_##scales[0] =  netname##_##layername##_##scales_bottom[0];\
for (int i = 0; i < sizeof(netname##_##layername##_##scales_weight) / sizeof(float); i++) {\
layername##_##scales[i + 1] = netname##_##layername##_##scales_weight[i];}

#else

#define Copy_Int8_Params(layername, netname)\
layername##_##bias =  (float*)_aligned_malloc(sizeof(netname##_##layername##_##bias), MALLOC_ALIGN); \
memcpy(layername##_##bias, netname##_##layername##_##bias, sizeof(netname##_##layername##_##bias));\
layername##_##weights_int8 =  (signed char*)_aligned_malloc(sizeof(netname##_##layername##_##weights) / sizeof(float) * sizeof(signed char), MALLOC_ALIGN); \
	for (int j = 0, num_weights = sizeof(netname##_##layername##_##weights) / sizeof(float), group = sizeof(netname##_##layername##_##scales_weight) / sizeof(float); j < group; j++){\
        int offset = j * num_weights / group;\
        for(int index = 0; index < num_weights / group; index++){\
			layername##_##weights_int8[offset + index] = round(netname##_##layername##_##weights[offset + index] * netname##_##layername##_##scales_weight[j]);}}\
layername##_##scales =  (float*)_aligned_malloc(sizeof(netname##_##layername##_##scales_bottom) + sizeof(netname##_##layername##_##scales_weight), MALLOC_ALIGN); \
layername##_##scales[0] =  netname##_##layername##_##scales_bottom[0];\
for (int i = 0; i < sizeof(netname##_##layername##_##scales_weight) / sizeof(float); i++) {\
layername##_##scales[i + 1] = netname##_##layername##_##scales_weight[i];}

#endif

#endif // INT8_DATA

#define Init_Conv_Params(conv_name, input_channel, output_channel, group, kernel_size, stride, pad, bias_term) \
if(device_ < 0){\
if(kernel_size == 3 && stride == 1){\
conv_name = new conv_winograd_cpu(input_channel, output_channel, group, kernel_size, stride, pad, bias_term, device_, int8_quantization_);}\
else{\
conv_name = new conv_native_cpu(input_channel, output_channel, group, kernel_size, stride, pad, bias_term, device_, int8_quantization_);}}\
else{\
if(cudnn_ready_){\
conv_name = new conv_cudnn_gpu(input_channel, output_channel, group, kernel_size, stride, pad, bias_term, device_); }\
else{\
conv_name = new conv_native_gpu(input_channel, output_channel, group, kernel_size, stride, pad, bias_term, device_); }}\
conv_name->set_bias(conv_name##_##bias);\
if(int8_quantization_){\
conv_name->set_weights(conv_name##_##weights_int8);\
conv_name->set_scales(conv_name##_##scales);}\
else{\
conv_name->set_weights(conv_name##_##weights);}


#define Init_Deconv_Params(deconv_name, input_channel, output_channel, group, kernel_size, stride, pad, bias_term)\
deconv_name = new deconv(input_channel, output_channel, group, kernel_size, stride, pad, bias_term, device_);\
deconv_name->set_weights(deconv_name##_##weights);\
deconv_name->set_bias(deconv_name##_##bias);


#define Init_PReLU_Shared_Params(prelu_name, input_channel, isrelu, is_shared)\
prelu_name = new prelu(input_channel, isrelu, device_, is_shared);\
prelu_name->setslope(prelu_name##_##weights);

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