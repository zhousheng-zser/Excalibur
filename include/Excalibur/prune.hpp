#pragma once
#ifndef _PRUNE_HPP_
#define _PRUNE_HPP_

#include <algorithm>
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <math.h>
using namespace std;


static bool my_cmp(const std::pair<int, float>& a, const std::pair<int, float>& b) {
	return a.second > b.second;
}


static unsigned short float2half(float value)
{
	// 1 : 8 : 23
	union
	{
		unsigned int u;
		float f;
	} tmp;

	tmp.f = value;

	// 1 : 8 : 23
	unsigned short sign = (tmp.u & 0x80000000) >> 31;
	unsigned short exponent = (tmp.u & 0x7F800000) >> 23;
	unsigned int significand = tmp.u & 0x7FFFFF;

	//     fprintf(stderr, "%d %d %d\n", sign, exponent, significand);

	// 1 : 5 : 10
	unsigned short fp16;
	if (exponent == 0)
	{
		// zero or denormal, always underflow
		fp16 = (sign << 15) | (0x00 << 10) | 0x00;
	}
	else if (exponent == 0xFF)
	{
		// infinity or NaN
		fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
	}
	else
	{
		// normalized
		short newexp = exponent + (-127 + 15);
		if (newexp >= 31)
		{
			// overflow, return infinity
			fp16 = (sign << 15) | (0x1F << 10) | 0x00;
		}
		else if (newexp <= 0)
		{
			// underflow
			if (newexp >= -10)
			{
				// denormal half-precision
				unsigned short sig = (significand | 0x800000) >> (14 - newexp);
				fp16 = (sign << 15) | (0x00 << 10) | sig;
			}
			else
			{
				// underflow
				fp16 = (sign << 15) | (0x00 << 10) | 0x00;
			}
		}
		else
		{
			fp16 = (sign << 15) | (newexp << 10) | (significand >> 13);
		}
	}

	return fp16;
}


static signed char float2int8(float v)
{
	int int32 = round(v);
	if (int32 > 127) return 127;
	if (int32 < -128) return -128;
	return (signed char)int32;
}


static void writedatafilehead(std::string filename, std::string namespace_, std::ofstream &out)
{
	std::transform(filename.begin(), filename.end(), filename.begin(), toupper);
	out << "#ifndef _" + filename + "_DATA_HPP_" << std::endl;
	out << "#define _" + filename + "_DATA_HPP_" << std::endl;
	out << std::endl << std::endl;
	out << "namespace glasssix {" << std::endl;
	out << "namespace " << namespace_ << " {" << std::endl;
}


static void writedatahead(std::string netname, std::string layername, std::string datatype, std::ofstream &out)
{
	out << "static const " + datatype + " " + netname + "_" + layername + "[] = {" << std::endl;
}


static void writedata(float* data, int len, int pos, std::ofstream &out)
{
	for (int i = 0; i < len; ++i, ++pos) {
		if (i == len - 1)
		{
			if (data[i] == 0.0f)
			{
				out << "0.0f";
			}
			else
			{
				out << data[i] << "f";
			}

		}
		else
		{
			if (data[i] == 0.0f)
			{
				out << "0.0f, ";
			}
			else
			{
				out << data[i] << "f, ";
			}
		}
		if (pos >= 15) {
			pos = 0;
			out << std::endl;
		}
	}
}


static void writedata(signed char* data, int len, int pos, std::ofstream &out)
{
	for (int i = 0; i < len; ++i, ++pos)
	{
		if (i == len - 1)
			out << int(data[i]);
		else
			out << int(data[i]) << ", ";
		if (pos >= 15) {
			pos = 0;
			out << std::endl;
		}
	}
}


static void writedataend(std::ofstream &out)
{
	out << "};" << std::endl << std::endl << std::endl;
}


static void writedatafileend(std::ofstream &out)
{
	out << "}" << std::endl;
	out << "}" << std::endl;
	out << "#endif" << std::endl;
}


#define Prune_Layer(layername, layername_str, input_channel, output_channel, kernel_size, sum, vec_map, ratio, pos, count)\
count = 0;\
pos = 0;\
vec_map.clear();\
for (int och = 0; och < output_channel; och++){\
sum[och] = 0;\
for (int ich = 0; ich < input_channel; ich++){\
for (int i = 0; i < kernel_size * kernel_size; i++){\
sum[och] += abs(layername##_##weights[och * input_channel * kernel_size * kernel_size + ich * kernel_size * kernel_size + i]);}}\
vec_map.push_back(std::make_pair(och,sum[och]));}\
sort(vec_map.begin(),vec_map.end(),my_cmp);\
vec_map.resize(floor(ratio * output_channel));\
for (int n = 0; n < floor(ratio * output_channel); n++){\
std::cout << vec_map[n].first << " " << vec_map[n].second << ",";}\
std::cout<<std::endl;\
sort(vec_map.begin(),vec_map.end());\
for (int n = 0; n < vec_map.size(); n++){\
std::cout << vec_map[n].first << " " << vec_map[n].second << std::endl;}\
writedatahead(std::string("Unicorn"), std::string(layername_str) + "_weights", std::string("float"), out);\
for (int n = 0; n < vec_map.size(); n++){\
	float* temp = layername##_##weights + vec_map[n].first * input_channel * kernel_size * kernel_size;\
        for(int i = 0; i < input_channel * kernel_size * kernel_size; i++, pos++){\
			if((n == floor(ratio * output_channel) - 1) && (i == input_channel * kernel_size * kernel_size - 1)){\
				if (temp[i] == 0.0f){\
					out << "0.0f";}\
				else{\
					out << temp[i] << "f";}}\
            else{\
				if (temp[i] == 0.0f){\
					out << "0.0f, ";}\
				else{\
					out << temp[i] << "f, ";}}\
            count++;\
			if (pos >= 15) {\
				pos = 0;\
				out << std::endl;}}}\
writedataend(out);\
std::cout<<layername_str<<" weights_num:"<<count<<std::endl;\
count = 0;\
pos = 0;\
writedatahead(std::string("Unicorn"), std::string(layername_str) + "_bias", std::string("float"), out);\
for (int n = 0; n < vec_map.size(); n++, pos++){\
	if(n == vec_map.size() - 1){\
		if (layername##_##bias[vec_map[n].first] == 0.0f){\
			out << "0.0f";}\
		else{\
			out << layername##_##bias[vec_map[n].first] << "f";}}\
    else{\
		if (layername##_##bias[vec_map[n].first] == 0.0f){\
			out << "0.0f, ";}\
		else{\
			out << layername##_##bias[vec_map[n].first] << "f, ";}}\
    count++;\
	if (pos >= 15) {\
		pos = 0;\
		out << std::endl;}}\
writedataend(out);\
std::cout<<layername_str<<" bias_num:"<<count<<std::endl;


#define Remain_Layer(layername, layername_str, input_channel, output_channel, kernel_size, vec_map, pos, count)\
count = 0;\
pos = 0;\
writedatahead(std::string("Unicorn"), std::string(layername_str) + "_weights", std::string("float"), out);\
for (int och = 0; och < output_channel; och++){\
	for (int n = 0; n < vec_map.size(); n++){\
		float* temp = layername##_##weights + och * input_channel * kernel_size * kernel_size + vec_map[n].first * kernel_size * kernel_size;\
		for(int i = 0; i < kernel_size * kernel_size; i++, pos++){\
			if((och == output_channel - 1) && (n == vec_map.size() - 1) && (i == kernel_size * kernel_size - 1)){\
				if (temp[i] == 0.0f){\
					out << "0.0f";}\
				else{\
					out << temp[i] << "f";}}\
            else{\
				if (temp[i] == 0.0f){\
					out << "0.0f, ";}\
				else{\
					out << temp[i] << "f, ";}}\
			count++;\
			if (pos >= 15) {\
				pos = 0;\
				out << std::endl;}}}}\
writedataend(out);\
std::cout<<layername_str<<" weights_num:"<<count<<std::endl;\
count = 0;\
pos = 0;\
writedatahead(std::string("Unicorn"), std::string(layername_str) + "_bias", std::string("float"), out);\
writedata(layername##_##bias, output_channel, pos, out);\
writedataend(out);\
std::cout<<layername_str<<" bias_num:"<<count<<std::endl;

#define Copy_Relu(layername, layername_str, vec_map, pos, count)\
count = 0;\
pos = 0;\
writedatahead(std::string("Unicorn"), std::string(layername_str) + "_weights", std::string("float"), out);\
for (int n = 0; n < vec_map.size(); n++, pos++){\
	if(n == vec_map.size() - 1){\
		if (layername##_##weights[vec_map[n].first] == 0.0f){\
			out << "0.0f";}\
		else{\
			out << layername##_##weights[vec_map[n].first] << "f";}}\
    else{\
		if (layername##_##weights[vec_map[n].first] == 0.0f){\
			out << "0.0f, ";}\
		else{\
			out << layername##_##weights[vec_map[n].first] << "f, ";}}\
    count++;\
	if (pos >= 15) {\
		pos = 0;\
		out << std::endl;}}\
writedataend(out);\
std::cout<<layername_str<<" relu_num:"<<count<<std::endl;


#define Write_Winograd_Param(layername, ptr, input_channel, output_channel, tile_length, pos, count)\
count = 0;\
pos = 0;\
writedatahead(std::string("Unicorn"), std::string(layername) + "_winograd_weights", std::string("float"), out);\
for (int och = 0; och < output_channel; och++){\
	for (int ich = 0; ich < input_channel; ich++){\
		float* temp = ptr + och * input_channel * tile_length + ich * tile_length;\
		for(int i = 0; i < tile_length; i++, pos++){\
			if((och == output_channel - 1) && (ich == input_channel - 1) && (i == tile_length - 1)){\
				if (temp[i] == 0.0f){\
					out << "0.0f";}\
				else{\
					out << temp[i] << "f";}}\
            else{\
				if (temp[i] == 0.0f){\
					out << "0.0f, ";}\
				else{\
					out << temp[i] << "f, ";}}\
			count++;\
			if (pos >= 15) {\
				pos = 0;\
				out << std::endl;}}}}\
writedataend(out);\
std::cout<<layername<<" winograd_weights_num:"<<count << ""<<std::endl;



#define Print_Star()\
std::cout<<"******************************"<<std::endl;

#define Print_Shared_Tensor(shared_tensor, pos, num)\
for (int i = pos; i < pos + num; i++){\
    std::cout<<float(shared_tensor->cpu_data()[i]) << " ";}\
std::cout<<std::endl;

#define Print_Tensor(tensor, pos, num)\
for (int i = pos; i < pos + num; i++){\
    std::cout<<float(tensor.cpu_data()[i]) << " ";}\
std::cout<<std::endl;


#define Copy_Params_Str(layer_para, netname, datatype, strlayername, strnetname)\
if(datatype == INT_MAX){\
layer_para =  (float*)_aligned_malloc(sizeof(netname##_##layer_para), MALLOC_ALIGN); \
memcpy(layer_para, netname##_##layer_para, sizeof(netname##_##layer_para));}\
writedatahead(strnetname, strlayername, std::string("float"), out); \
writedata(layer_para, sizeof(netname##_##layer_para) / sizeof(float), pos, out); \
writedataend(out);


#if SIMD_TYPE >= SIMDTYPE_SSE

#define Copy_Int8_Params_Str(layername, netname, strlayername, strnetname)\
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
layername##_##scales[i + 1] = netname##_##layername##_##scales_weight[i];}\
writedatahead(strnetname, std::string(strlayername) + "_weights", std::string("signed char"), out);\
writedata(layername##_##weights_int8, sizeof(netname##_##layername##_##weights) / sizeof(float), pos, out); \
writedataend(out);\
writedatahead(strnetname, std::string(strlayername) + "_bias", std::string("float"), out);\
writedata(layername##_##bias, sizeof(netname##_##layername##_##bias) / sizeof(float), pos, out);\
writedataend(out);


#else

#define Copy_Int8_Params_Str(layername, netname, strlayername, strnetname)\
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
layername##_##scales[i + 1] = netname##_##layername##_##scales_weight[i];}\
writedatahead(strnetname, std::string(strlayername) + "_weights", std::string("signed char"), out);\
writedata(layername##_##weights_int8, sizeof(netname##_##layername##_##weights) / sizeof(float), pos, out); \
writedataend(out);\
writedatahead(strnetname, std::string(strlayername) + "_bias", std::string("float"), out);\
writedata(layername##_##bias, sizeof(netname##_##layername##_##bias) / sizeof(float), pos, out);\
writedataend(out);

#endif

//{

//string prefix = "retina_net";
//string name_space = "longinus";
//ofstream out("D:/projects/data/detectModel/retinaFace/retina_net_int8_data.hpp");
//int pos = 0;
//writedatafilehead(prefix, name_space, out);

//Copy_Int8_Params_Str(mobilenet0_conv0_fwd, retina_net, "mobilenet0_conv0_fwd", prefix);
//Copy_Params_Str(mobilenet0_conv1_fwd_weights, retina_net, quantize_level, "mobilenet0_conv1_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv1_fwd_bias, retina_net, quantize_level, "mobilenet0_conv1_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv2_fwd_weights, retina_net, quantize_level, "mobilenet0_conv2_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv2_fwd_bias, retina_net, quantize_level, "mobilenet0_conv2_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv3_fwd_weights, retina_net, quantize_level, "mobilenet0_conv3_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv3_fwd_bias, retina_net, quantize_level, "mobilenet0_conv3_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv4_fwd_weights, retina_net, quantize_level, "mobilenet0_conv4_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv4_fwd_bias, retina_net, quantize_level, "mobilenet0_conv4_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv5_fwd_weights, retina_net, quantize_level, "mobilenet0_conv5_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv5_fwd_bias, retina_net, quantize_level, "mobilenet0_conv5_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv6_fwd_weights, retina_net, quantize_level, "mobilenet0_conv6_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv6_fwd_bias, retina_net, quantize_level, "mobilenet0_conv6_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv7_fwd_weights, retina_net, quantize_level, "mobilenet0_conv7_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv7_fwd_bias, retina_net, quantize_level, "mobilenet0_conv7_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv8_fwd_weights, retina_net, quantize_level, "mobilenet0_conv8_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv8_fwd_bias, retina_net, quantize_level, "mobilenet0_conv8_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv9_fwd_weights, retina_net, quantize_level, "mobilenet0_conv9_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv9_fwd_bias, retina_net, quantize_level, "mobilenet0_conv9_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv10_fwd_weights, retina_net, quantize_level, "mobilenet0_conv10_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv10_fwd_bias, retina_net, quantize_level, "mobilenet0_conv10_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv11_fwd_weights, retina_net, quantize_level, "mobilenet0_conv11_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv11_fwd_bias, retina_net, quantize_level, "mobilenet0_conv11_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv12_fwd_weights, retina_net, quantize_level, "mobilenet0_conv12_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv12_fwd_bias, retina_net, quantize_level, "mobilenet0_conv12_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv13_fwd_weights, retina_net, quantize_level, "mobilenet0_conv13_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv13_fwd_bias, retina_net, quantize_level, "mobilenet0_conv13_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv14_fwd_weights, retina_net, quantize_level, "mobilenet0_conv14_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv14_fwd_bias, retina_net, quantize_level, "mobilenet0_conv14_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv15_fwd_weights, retina_net, quantize_level, "mobilenet0_conv15_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv15_fwd_bias, retina_net, quantize_level, "mobilenet0_conv15_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv16_fwd_weights, retina_net, quantize_level, "mobilenet0_conv16_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv16_fwd_bias, retina_net, quantize_level, "mobilenet0_conv16_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv17_fwd_weights, retina_net, quantize_level, "mobilenet0_conv17_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv17_fwd_bias, retina_net, quantize_level, "mobilenet0_conv17_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv18_fwd_weights, retina_net, quantize_level, "mobilenet0_conv18_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv18_fwd_bias, retina_net, quantize_level, "mobilenet0_conv18_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv19_fwd_weights, retina_net, quantize_level, "mobilenet0_conv19_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv19_fwd_bias, retina_net, quantize_level, "mobilenet0_conv19_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv20_fwd_weights, retina_net, quantize_level, "mobilenet0_conv20_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv20_fwd_bias, retina_net, quantize_level, "mobilenet0_conv20_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv21_fwd_weights, retina_net, quantize_level, "mobilenet0_conv21_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv21_fwd_bias, retina_net, quantize_level, "mobilenet0_conv21_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv22_fwd_weights, retina_net, quantize_level, "mobilenet0_conv22_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv22_fwd_bias, retina_net, quantize_level, "mobilenet0_conv22_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv23_fwd_weights, retina_net, quantize_level, "mobilenet0_conv23_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv23_fwd_bias, retina_net, quantize_level, "mobilenet0_conv23_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv24_fwd_weights, retina_net, quantize_level, "mobilenet0_conv24_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv24_fwd_bias, retina_net, quantize_level, "mobilenet0_conv24_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv25_fwd_weights, retina_net, quantize_level, "mobilenet0_conv25_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv25_fwd_bias, retina_net, quantize_level, "mobilenet0_conv25_fwd_bias", prefix);
//Copy_Params_Str(mobilenet0_conv26_fwd_weights, retina_net, quantize_level, "mobilenet0_conv26_fwd_weights", prefix);
//Copy_Params_Str(mobilenet0_conv26_fwd_bias, retina_net, quantize_level, "mobilenet0_conv26_fwd_bias", prefix);
//Copy_Params_Str(rf_c3_lateral_weights, retina_net, quantize_level, "rf_c3_lateral_weights", prefix);
//Copy_Params_Str(rf_c3_lateral_bias, retina_net, quantize_level, "rf_c3_lateral_bias", prefix);
//Copy_Int8_Params_Str(rf_c3_det_conv1, retina_net, "rf_c3_det_conv1", prefix);
//Copy_Int8_Params_Str(rf_c3_det_context_conv1, retina_net, "rf_c3_det_context_conv1", prefix);
//Copy_Int8_Params_Str(rf_c3_det_context_conv2, retina_net, "rf_c3_det_context_conv2", prefix);
//Copy_Int8_Params_Str(rf_c3_det_context_conv3_1, retina_net, "rf_c3_det_context_conv3_1", prefix);
//Copy_Int8_Params_Str(rf_c3_det_context_conv3_2, retina_net, "rf_c3_det_context_conv3_2", prefix);
//Copy_Params_Str(face_rpn_cls_score_stride32_weights, retina_net, quantize_level, "face_rpn_cls_score_stride32_weights", prefix);
//Copy_Params_Str(face_rpn_cls_score_stride32_bias, retina_net, quantize_level, "face_rpn_cls_score_stride32_bias", prefix);
//Copy_Params_Str(face_rpn_bbox_pred_stride32_weights, retina_net, quantize_level, "face_rpn_bbox_pred_stride32_weights", prefix);
//Copy_Params_Str(face_rpn_bbox_pred_stride32_bias, retina_net, quantize_level, "face_rpn_bbox_pred_stride32_bias", prefix);
//Copy_Params_Str(face_rpn_landmark_pred_stride32_weights, retina_net, quantize_level, "face_rpn_landmark_pred_stride32_weights", prefix);
//Copy_Params_Str(face_rpn_landmark_pred_stride32_bias, retina_net, quantize_level, "face_rpn_landmark_pred_stride32_bias", prefix);
//Copy_Params_Str(rf_c2_lateral_weights, retina_net, quantize_level, "rf_c2_lateral_weights", prefix);
//Copy_Params_Str(rf_c2_lateral_bias, retina_net, quantize_level, "rf_c2_lateral_bias", prefix);
//Copy_Params_Str(rf_c3_upsampling_weights, retina_net, quantize_level, "rf_c3_upsampling_weights", prefix);
//Copy_Int8_Params_Str(rf_c2_aggr, retina_net, "rf_c2_aggr", prefix);
//Copy_Int8_Params_Str(rf_c2_det_conv1, retina_net, "rf_c2_det_conv1", prefix);
//Copy_Int8_Params_Str(rf_c2_det_context_conv1, retina_net, "rf_c2_det_context_conv1", prefix);
//Copy_Int8_Params_Str(rf_c2_det_context_conv2, retina_net, "rf_c2_det_context_conv2", prefix);
//Copy_Int8_Params_Str(rf_c2_det_context_conv3_1, retina_net, "rf_c2_det_context_conv3_1", prefix);
//Copy_Int8_Params_Str(rf_c2_det_context_conv3_2, retina_net, "rf_c2_det_context_conv3_2", prefix);
//Copy_Params_Str(face_rpn_cls_score_stride16_weights, retina_net, quantize_level, "face_rpn_cls_score_stride16_weights", prefix);
//Copy_Params_Str(face_rpn_cls_score_stride16_bias, retina_net, quantize_level, "face_rpn_cls_score_stride16_bias", prefix);
//Copy_Params_Str(face_rpn_bbox_pred_stride16_weights, retina_net, quantize_level, "face_rpn_bbox_pred_stride16_weights", prefix);
//Copy_Params_Str(face_rpn_bbox_pred_stride16_bias, retina_net, quantize_level, "face_rpn_bbox_pred_stride16_bias", prefix);
//Copy_Params_Str(face_rpn_landmark_pred_stride16_weights, retina_net, quantize_level, "face_rpn_landmark_pred_stride16_weights", prefix);
//Copy_Params_Str(face_rpn_landmark_pred_stride16_bias, retina_net, quantize_level, "face_rpn_landmark_pred_stride16_bias", prefix);
//Copy_Params_Str(rf_c1_red_conv_weights, retina_net, quantize_level, "rf_c1_red_conv_weights", prefix);
//Copy_Params_Str(rf_c1_red_conv_bias, retina_net, quantize_level, "rf_c1_red_conv_bias", prefix);
//Copy_Params_Str(rf_c2_upsampling_weights, retina_net, quantize_level, "rf_c2_upsampling_weights", prefix);
//Copy_Int8_Params_Str(rf_c1_aggr, retina_net, "rf_c1_aggr", prefix);
//Copy_Int8_Params_Str(rf_c1_det_conv1, retina_net, "rf_c1_det_conv1", prefix);
//Copy_Int8_Params_Str(rf_c1_det_context_conv1, retina_net, "rf_c1_det_context_conv1", prefix);
//Copy_Int8_Params_Str(rf_c1_det_context_conv2, retina_net, "rf_c1_det_context_conv2", prefix);
//Copy_Int8_Params_Str(rf_c1_det_context_conv3_1, retina_net, "rf_c1_det_context_conv3_1", prefix);
//Copy_Int8_Params_Str(rf_c1_det_context_conv3_2, retina_net, "rf_c1_det_context_conv3_2", prefix);
//Copy_Params_Str(face_rpn_cls_score_stride8_weights, retina_net, quantize_level, "face_rpn_cls_score_stride8_weights", prefix);
//Copy_Params_Str(face_rpn_cls_score_stride8_bias, retina_net, quantize_level, "face_rpn_cls_score_stride8_bias", prefix);
//Copy_Params_Str(face_rpn_bbox_pred_stride8_weights, retina_net, quantize_level, "face_rpn_bbox_pred_stride8_weights", prefix);
//Copy_Params_Str(face_rpn_bbox_pred_stride8_bias, retina_net, quantize_level, "face_rpn_bbox_pred_stride8_bias", prefix);
//Copy_Params_Str(face_rpn_landmark_pred_stride8_weights, retina_net, quantize_level, "face_rpn_landmark_pred_stride8_weights", prefix);
//Copy_Params_Str(face_rpn_landmark_pred_stride8_bias, retina_net, quantize_level, "face_rpn_landmark_pred_stride8_bias", prefix);

//writedatafileend(out);

//}

#endif // !_PRUNE_HPP_