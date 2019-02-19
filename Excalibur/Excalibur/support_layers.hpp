#pragma once
#ifndef _SUPPORT_LAYERS_HPP_
#define _SUPPORT_LAYERS_HPP_
#include "convolution.hpp"
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


namespace glasssix
{

	namespace excalibur
	{
		// convert half precision floating point to float
		static float half2float(unsigned short value)
		{
			// 1 : 5 : 10
			unsigned short sign = (value & 0x8000) >> 15;
			unsigned short exponent = (value & 0x7c00) >> 10;
			unsigned short significand = value & 0x03FF;

			//     fprintf(stderr, "%d %d %d\n", sign, exponent, significand);

			// 1 : 8 : 23
			union
			{
				unsigned int u;
				float f;
			} tmp;
			if (exponent == 0)
			{
				if (significand == 0)
				{
					// zero
					tmp.u = (sign << 31);
				}
				else
				{
					// denormal
					exponent = 0;
					// find non-zero bit
					while ((significand & 0x200) == 0)
					{
						significand <<= 1;
						exponent++;
					}
					significand <<= 1;
					significand &= 0x3FF;
					tmp.u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
				}
			}
			else if (exponent == 0x1F)
			{
				// infinity or NaN
				tmp.u = (sign << 31) | (0xFF << 23) | (significand << 13);
			}
			else
			{
				// normalized
				tmp.u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
			}

			return tmp.f;
		}
	}
}

#define Neuron_Name(name) private: \
std::shared_ptr<tensor<float>> name##_top_data = nullptr;\
public: std::shared_ptr<tensor<float>> get_##name(){\
return name##_top_data;\
}\
private:

#define  Declear_Opration(op, name) op##* name;

#define Declear_Params(layer_para) float* layer_para;

#ifdef USE_MKL
#define Copy_Params(layer_para, netname, datatype)\
if(datatype == INT_MAX){\
layer_para =  (float*)mkl_malloc(sizeof(netname##_##layer_para) ? sizeof(netname##_##layer_para) :1, 64); \
memcpy(layer_para, netname##_##layer_para, sizeof(netname##_##layer_para));}\
if(datatype == USHRT_MAX){\
layer_para =  (float*)mkl_malloc(sizeof(netname##_##layer_para) ? sizeof(netname##_##layer_para) / sizeof(unsigned short) * sizeof(float) :1, 64); \
for (int i = 0; i < sizeof(netname##_##layer_para) / sizeof(unsigned short); i++){\
layer_para[i] = half2float(netname##_##layer_para[i]);}}
#else
#define Copy_Params(layer_para, netname, datatype)\
if(datatype == INT_MAX){\
layer_para =  (float*)_aligned_malloc(sizeof(netname##_##layer_para), MALLOC_ALIGN); \
memcpy(layer_para, netname##_##layer_para, sizeof(netname##_##layer_para));}\
if(datatype == USHRT_MAX) {\
layer_para =  (float*)_aligned_malloc(sizeof(netname##_##layer_para) / sizeof(unsigned short) * sizeof(float), MALLOC_ALIGN); \
for (int i = 0; i < sizeof(netname##_##layer_para) / sizeof(unsigned short); i++) {\
layer_para[i] = half2float(netname##_##layer_para[i]);}}
#endif

#define Init_Conv_Params(conv_name, input_channel, output_channel, kernel_size, stride, pad, bias_term) \
conv_name = new convolution(input_channel, output_channel, kernel_size, stride, pad, bias_term, device_);\
conv_name->set_weights(conv_name##_##weights);\
conv_name->set_bias(conv_name##_##bias);

#define Init_DepthConv_Params(conv_name, input_channel, output_channel, kernel_size, stride, pad, bias_term) \
conv_name = new convolution(input_channel, output_channel, kernel_size, output_channel, stride, pad, bias_term, device_);\
conv_name->set_weights(conv_name##_##weights);\
conv_name->set_bias(conv_name##_##bias);\
conv_name->set_depthwise();

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