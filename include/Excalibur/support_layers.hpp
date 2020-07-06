#pragma once
#ifndef _SUPPORT_LAYERS_HPP_
#define _SUPPORT_LAYERS_HPP_

#include "Primitives/tensor.hpp"
#include "Primitives/simd_types.hpp"
#include "base_conv.hpp"
#include "conv_cudnn_gpu.hpp"
#include "conv_native_cpu.hpp"
#include "conv_native_gpu.hpp"
#include "conv_1x1s1_cpu.hpp"
#include "convdw_3x3s1_cpu.hpp"
#include "convdw_3x3s2_cpu.hpp"
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
#include "axpy.hpp"
#include "deconv.hpp"
#include "upsample.hpp"
#include "reshape.hpp"
#include "hswish.hpp"

#include "arm/conv_arm.hpp"
#include "arm/inner_product_arm.hpp"
#include "arm/pooling_arm.hpp"
#include "arm/prelu_arm.hpp"
#include "arm/softmax_arm.hpp"
#include "arm/sigmoid_arm.hpp"
#include "arm/batchnorm_arm.hpp"
#include "arm/scale_arm.hpp"
#include "arm/eltwise_arm.hpp"
#include "arm/deconv_arm.hpp"

#include <climits>

#define Neuron_Name(name) private: \
std::shared_ptr<glasssix::memory::tensor<float>> name##_top_data;\
public: std::shared_ptr<glasssix::memory::tensor<float>> get_##name(){\
return name##_top_data;\
}\
private:

#define  Declear_Opration(op, name) excalibur::op* name;

#define Declear_Params(layername) float *layername##_##bias, *layername##_##weights, *layername##_##scales;\
signed char *layername##_##weights_int8;

#ifdef x86
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
layer_para =  (float*)glasssix::memory::aligned_heap_alloc(sizeof(netname##_##layer_para), MALLOC_ALIGN); \
memcpy(layer_para, netname##_##layer_para, sizeof(netname##_##layer_para));}\
if(datatype == USHRT_MAX) {\
layer_para =  (float*)glasssix::memory::aligned_heap_alloc(sizeof(netname##_##layer_para) / sizeof(unsigned short) * sizeof(float), MALLOC_ALIGN); \
half2float((unsigned short*)netname##_##layer_para,layer_para,sizeof(netname##_##layer_para) / sizeof(unsigned short));}
#endif
#else
#define Copy_Params(layer_para, netname, datatype)\
if(datatype == INT_MAX){\
layer_para =  (float*)glasssix::memory::aligned_heap_alloc(sizeof(netname##_##layer_para), MALLOC_ALIGN); \
memcpy(layer_para, netname##_##layer_para, sizeof(netname##_##layer_para));}\
if(datatype == USHRT_MAX) {\
NOT_IMPLEMENTED;}
#endif

#define Copy_Int8_to_FP32_Params(layername, netname)\
layername##_##weights =  (float*)glasssix::memory::aligned_heap_alloc(sizeof(netname##_##layername##_##weights) / sizeof(signed char) * sizeof(float), MALLOC_ALIGN); \
int8_to_float((const signed char*)netname##_##layername##_##weights,(const float*)netname##_##layername##_##scales_weight,(float*)layername##_##weights,\
    sizeof(netname##_##layername##_##weights) / sizeof(signed char),sizeof(netname##_##layername##_##scales_weight) / sizeof(float));

//#define INT8_DATA
#ifdef INT8_DATA //copy directely, do not caculate

#define Copy_Int8_Params(layername, netname)\
layername##_##bias =  (float*)glasssix::memory::aligned_heap_alloc(sizeof(netname##_##layername##_##bias), MALLOC_ALIGN); \
memcpy(layername##_##bias, netname##_##layername##_##bias, sizeof(netname##_##layername##_##bias));\
layername##_##weights_int8 =  (signed char*)glasssix::memory::aligned_heap_alloc(sizeof(netname##_##layername##_##weights), MALLOC_ALIGN); \
memcpy(layername##_##weights_int8, netname##_##layername##_##weights, sizeof(netname##_##layername##_##weights));\
layername##_##scales =  (float*)glasssix::memory::aligned_heap_alloc(sizeof(netname##_##layername##_##scales_bottom) + sizeof(netname##_##layername##_##scales_weight), MALLOC_ALIGN); \
layername##_##scales[0] =  netname##_##layername##_##scales_bottom[0];\
for (int i = 0; i < sizeof(netname##_##layername##_##scales_weight) / sizeof(float); i++) {\
layername##_##scales[i + 1] = netname##_##layername##_##scales_weight[i];}

#else

#if SIMD_TYPE >= SIMDTYPE_SSE

#define Copy_Int8_Params(layername, netname)\
layername##_##bias =  (float*)glasssix::memory::aligned_heap_alloc(sizeof(netname##_##layername##_##bias), MALLOC_ALIGN); \
memcpy(layername##_##bias, netname##_##layername##_##bias, sizeof(netname##_##layername##_##bias));\
layername##_##weights_int8 =  (signed char*)glasssix::memory::aligned_heap_alloc(sizeof(netname##_##layername##_##weights) / sizeof(float) * sizeof(signed char), MALLOC_ALIGN); \
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
layername##_##scales =  (float*)glasssix::memory::aligned_heap_alloc(sizeof(netname##_##layername##_##scales_bottom) + sizeof(netname##_##layername##_##scales_weight), MALLOC_ALIGN); \
layername##_##scales[0] =  netname##_##layername##_##scales_bottom[0];\
for (int i = 0; i < sizeof(netname##_##layername##_##scales_weight) / sizeof(float); i++) {\
layername##_##scales[i + 1] = netname##_##layername##_##scales_weight[i];}

#else

#define Copy_Int8_Params(layername, netname)\
layername##_##bias =  (float*)glasssix::memory::aligned_heap_alloc(sizeof(netname##_##layername##_##bias), MALLOC_ALIGN); \
memcpy(layername##_##bias, netname##_##layername##_##bias, sizeof(netname##_##layername##_##bias));\
layername##_##weights_int8 =  (signed char*)glasssix::memory::aligned_heap_alloc(sizeof(netname##_##layername##_##weights) / sizeof(float) * sizeof(signed char), MALLOC_ALIGN); \
	for (int j = 0, num_weights = sizeof(netname##_##layername##_##weights) / sizeof(float), group = sizeof(netname##_##layername##_##scales_weight) / sizeof(float); j < group; j++){\
        int offset = j * num_weights / group;\
        for(int index = 0; index < num_weights / group; index++){\
			layername##_##weights_int8[offset + index] = round(netname##_##layername##_##weights[offset + index] * netname##_##layername##_##scales_weight[j]);}}\
layername##_##scales =  (float*)glasssix::memory::aligned_heap_alloc(sizeof(netname##_##layername##_##scales_bottom) + sizeof(netname##_##layername##_##scales_weight), MALLOC_ALIGN); \
layername##_##scales[0] =  netname##_##layername##_##scales_bottom[0];\
for (int i = 0; i < sizeof(netname##_##layername##_##scales_weight) / sizeof(float); i++) {\
layername##_##scales[i + 1] = netname##_##layername##_##scales_weight[i];}

#endif

#endif // INT8_DATA


#define Init_Conv_Params(conv_name, input_channel, output_channel, group, kernel_size, stride, pad, bias_term) \
if(device_ < 0){\
    bool int8_quantization = int8_quantization_;\
    if((group > 1) || (kernel_size == 1)) { int8_quantization = false;}\
	if((stride == 1) && (kernel_size == 1)){\
		conv_name = new excalibur::conv_1x1s1_cpu(input_channel, output_channel, group, kernel_size, stride, pad, bias_term, device_, int8_quantization);}\
	else if((stride == 1) && (kernel_size == 3) && (group > 1)){\
		conv_name = new excalibur::convdw_3x3s1_cpu(input_channel, output_channel, group, kernel_size, stride, pad, bias_term, device_, int8_quantization);}\
	else if((stride == 2) && (kernel_size == 3) && (group > 1)){\
		conv_name = new excalibur::convdw_3x3s2_cpu(input_channel, output_channel, group, kernel_size, stride, pad, bias_term, device_, int8_quantization);}\
	else if((stride == 1) && (kernel_size == 3)){\
		conv_name = new excalibur::conv_winograd_cpu(input_channel, output_channel, group, kernel_size, stride, pad, bias_term, device_, int8_quantization);}\
    else{\
        conv_name = new excalibur::conv_native_cpu(input_channel, output_channel, group, kernel_size, stride, pad, bias_term, device_, int8_quantization);}\
    conv_name->set_bias(conv_name##_##bias);\
    if(int8_quantization){\
        conv_name->set_weights(conv_name##_##weights_int8);\
        conv_name->set_scales(conv_name##_##scales);}\
    else{conv_name->set_weights(conv_name##_##weights);}}\
else{\
    bool int8_quantization = int8_quantization_;\
    if((group > 1) || (kernel_size == 1)) { int8_quantization = false;}\
    if(cudnn_ready_){\
        conv_name = new excalibur::conv_cudnn_gpu(input_channel, output_channel, group, kernel_size, stride, pad, bias_term, device_, int8_quantization);}\
    else{\
        conv_name = new excalibur::conv_native_gpu(input_channel, output_channel, group, kernel_size, stride, pad, bias_term, device_, int8_quantization);}\
    conv_name->set_bias(conv_name##_##bias);\
    if(int8_quantization){\
        conv_name->set_weights(conv_name##_##weights_int8);\
        conv_name->set_scales(conv_name##_##scales);}\
    else{conv_name->set_weights(conv_name##_##weights);}}


#define Init_Deconv_Params(deconv_name, input_channel, output_channel, group, kernel_size, stride, pad, bias_term)\
deconv_name = new excalibur::deconv(input_channel, output_channel, group, kernel_size, stride, pad, bias_term, device_);\
deconv_name->set_weights(deconv_name##_##weights);\
deconv_name->set_bias(deconv_name##_##bias);

#define Init_Upsample_Params(upsample_name, scale)\
upsample_name = new excalibur::upsample(scale, device_);

#define Init_PReLU_Shared_Params(prelu_name, input_channel, isrelu, is_shared)\
prelu_name = new excalibur::prelu(input_channel, isrelu, device_, is_shared);\
prelu_name->setslope(prelu_name##_##weights);

#define Init_PReLU_Params(prelu_name, input_channel, isrelu)\
prelu_name = new excalibur::prelu(input_channel, isrelu, device_);\
prelu_name->setslope(prelu_name##_##weights);

#define Init_ReLU_Params(prelu_name, input_channel, isrelu)\
prelu_name = new excalibur::prelu(input_channel, isrelu, device_);

#define Init_Pooling_Params(pooling_name, kernel, stride, pad, type)\
pooling_name = new excalibur::pooling(kernel, stride, pad, type, device_);

#define Init_Softmax_Params(softmax_name, input_channel)\
softmax_name = new excalibur::softmax(input_channel, device_);

#define Init_Eltwise_Params(eltwise_name, type)\
eltwise_name = new excalibur::eltwise(type, device_);

#define Init_InnerProduct_Params(ip_name, input_channel, input_height, input_width, num_output, bias_term)\
ip_name = new excalibur::inner_product(std::vector<int>{1, input_channel, input_height, input_width}, num_output, bias_term, device_);\
ip_name->set_weights(ip_name##_##weights);\
ip_name->set_bias(ip_name##_##bias);

#define Init_Flip_Params(fliper_name, flip_height, flip_width)\
fliper_name = new excalibur::flip(flip_height, flip_width, device_);

#define Init_Concat_Params(concat_name, concat_axis)\
concat_name = new excalibur::concat(concat_axis, device_);

#define Init_Sigmoid_Params(sigmoid_name)\
sigmoid_name = new excalibur::sigmoid();

#define Init_Reshape_Params(reshape_name, dimension1, dimension2, dimension3, dimension4)\
reshape_name = new excalibur::reshape(dimension1, dimension2, dimension3, dimension4, device_);

#define Init_Slice_Params(slice_name, slice_axis)\
slice_name = new excalibur::slice(slice_axis, device_);

#define Init_Normalize_Params(norm_name, type, rescale)\
norm_name = new excalibur::normalize(type, rescale, device_);

#define Init_MirrorMax_Param(mm_name, mirror_axis)\
mm_name = new excalibur::mirrormax(mirror_axis, device_);

#define Init_Conv_arm_Params(conv_name, input_channel, output_channel, group, kernel_size, stride, pad, bias_term) \
if(device_ < 0){\
    bool int8_quantization = int8_quantization_;\
    if((group > 1) || (kernel_size == 1)) { int8_quantization = false;}\
    if (!int8_quantization) {\
		if(group > 1 && kernel_size > 3){conv_name = new excalibur::conv_native_cpu(input_channel, output_channel, group, kernel_size, stride, pad, bias_term, device_, int8_quantization);}\
		else{conv_name = new excalibur::conv_arm(input_channel, output_channel, group, kernel_size, stride, pad, bias_term, device_, int8_quantization);}}\
	else { conv_name = new excalibur::conv_native_cpu(input_channel, output_channel, group, kernel_size, stride, pad, bias_term, device_, int8_quantization);}\
    conv_name->set_bias(conv_name##_##bias);\
    if(int8_quantization){\
        conv_name->set_weights(conv_name##_##weights_int8);\
        conv_name->set_scales(conv_name##_##scales);}\
    else{conv_name->set_weights(conv_name##_##weights);}}\
else{\
    NOT_IMPLEMENTED;}

#define Init_PReLU_arm_Params(prelu_name, input_channel, isrelu, is_shared)\
prelu_name = new excalibur::prelu_arm(input_channel, isrelu, -1, is_shared);\
prelu_name->setslope(prelu_name##_##weights);

#define Init_ReLU_arm_Params(prelu_name, input_channel, isrelu)\
prelu_name = new excalibur::prelu_arm(input_channel, isrelu, -1, false);

#define Init_Reshape_arm_Params(reshape_name, dim1, dim2, dim3, dim4)\
reshape_name = new excalibur::reshape(dim1, dim2, dim3, dim4);

#define Init_Pooling_arm_Params(pooling_name, kernel, stride, pad, type)\
pooling_name = new excalibur::pooling_arm(kernel, stride, pad, type, -1);

#define Init_Softmax_arm_Params(softmax_name, input_channel)\
softmax_name = new excalibur::softmax_arm(input_channel, -1);

#define Init_InnerProduct_arm_Params(ip_name, input_channel, input_height, input_width, num_output, bias_term)\
ip_name = new excalibur::inner_product_arm(std::vector<int>{1, input_channel, input_height, input_width}, num_output, bias_term, -1);\
ip_name->set_weights(ip_name##_##weights);\
ip_name->set_bias(ip_name##_##bias);

#define Init_Sigmoid_arm_Params(sigmoid_name)\
sigmoid_name = new excalibur::sigmoid_arm();

#define Init_BatchNorm_arm_Params(batchnorm_name, input_channel) \
batchnorm_name = new excalibur::batchnorm_arm(input_channel); \
batchnorm_name->set_weights(batchnorm_name##_##weights); \
batchnorm_name->set_bias(batchnorm_name##_##bias);

#define Init_Scale_arm_Params(scale_name, input_channel, bias_term) \
scale_name = new excalibur::scale_arm(input_channel, bias_term); \
scale_name->set_weights(scale_name##_##weights); \
scale_name->set_bias(scale_name##_##bias);

#define Init_Eltwise_arm_Params(eltwise_name, type)\
eltwise_name = new excalibur::eltwise_arm(type, device_);

#define Init_Deconv_arm_Params(deconv_name, input_channel, output_channel, group, kernel_size, stride, pad, bias_term)\
deconv_name = new excalibur::deconv_arm(input_channel, output_channel, group, kernel_size, stride, pad, bias_term, device_);\
deconv_name->set_weights(deconv_name##_##weights);\
deconv_name->set_bias(deconv_name##_##bias);

#endif //_SUPPORT_LAYERS_HPP_
