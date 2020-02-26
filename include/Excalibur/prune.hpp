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
//string file_name = "Unicorn";
//string name_space = "cassius";
//ofstream out("D:/projects/data/unicornModel/cassius/new/unicorn_int8_data.hpp");
//int pos = 0;
//writedatafilehead(file_name, name_space, out);
//if (true)
//{
//	Copy_Int8_Params_Str(conv1a, Unicorn, "conv1a", "Unicorn");//64
//	Copy_Params_Str(relu1a_weights, Unicorn, quantize_level, "relu1a_weights", "Unicorn");//32
//	Copy_Int8_Params_Str(conv1b, Unicorn, "conv1b", "Unicorn");//18432
//	Copy_Params_Str(relu1b_weights, Unicorn, quantize_level, "relu1b_weights", "Unicorn");//64
//	Copy_Int8_Params_Str(conv2_1, Unicorn, "conv2_1", "Unicorn");//36864
//	Copy_Params_Str(relu2_1_weights, Unicorn, quantize_level, "relu2_1_weights", "Unicorn");//64
//	Copy_Int8_Params_Str(conv2_2, Unicorn, "conv2_2", "Unicorn");//36864
//	Copy_Params_Str(relu2_2_weights, Unicorn, quantize_level, "relu2_2_weights", "Unicorn");//64
//	Copy_Int8_Params_Str(conv2, Unicorn, "conv2", "Unicorn");//73728
//	Copy_Params_Str(relu2_weights, Unicorn, quantize_level, "relu2_weights", "Unicorn");//128
//	Copy_Int8_Params_Str(conv3_1, Unicorn, "conv3_1", "Unicorn");//147456
//	Copy_Params_Str(relu3_1_weights, Unicorn, quantize_level, "relu3_1_weights", "Unicorn");//128
//	Copy_Int8_Params_Str(conv3_2, Unicorn, "conv3_2", "Unicorn");//147456
//	Copy_Params_Str(relu3_2_weights, Unicorn, quantize_level, "relu3_2_weights", "Unicorn");//128
//	Copy_Int8_Params_Str(conv3_3, Unicorn, "conv3_3", "Unicorn");//147456
//	Copy_Params_Str(relu3_3_weights, Unicorn, quantize_level, "relu3_3_weights", "Unicorn");//128
//	Copy_Int8_Params_Str(conv3_4, Unicorn, "conv3_4", "Unicorn");//147456
//	Copy_Params_Str(relu3_4_weights, Unicorn, quantize_level, "relu3_4_weights", "Unicorn");//128
//	Copy_Int8_Params_Str(conv3, Unicorn, "conv3", "Unicorn");//294912
//	Copy_Params_Str(relu3_weights, Unicorn, quantize_level, "relu3_weights", "Unicorn");//256
//	Copy_Int8_Params_Str(conv4_1, Unicorn, "conv4_1", "Unicorn");//589824
//	Copy_Params_Str(relu4_1_weights, Unicorn, quantize_level, "relu4_1_weights", "Unicorn");//256
//	Copy_Int8_Params_Str(conv4_2, Unicorn, "conv4_2", "Unicorn");//589824
//	Copy_Params_Str(relu4_2_weights, Unicorn, quantize_level, "relu4_2_weights", "Unicorn");//256
//	Copy_Int8_Params_Str(conv4_3, Unicorn, "conv4_3", "Unicorn");//589824
//	Copy_Params_Str(relu4_3_weights, Unicorn, quantize_level, "relu4_3_weights", "Unicorn");//256
//	Copy_Int8_Params_Str(conv4_4, Unicorn, "conv4_4", "Unicorn");//589824
//	Copy_Params_Str(relu4_4_weights, Unicorn, quantize_level, "relu4_4_weights", "Unicorn");//256
//	Copy_Int8_Params_Str(conv4_5, Unicorn, "conv4_5", "Unicorn");//589824
//	Copy_Params_Str(relu4_5_weights, Unicorn, quantize_level, "relu4_5_weights", "Unicorn");//256
//	Copy_Int8_Params_Str(conv4_6, Unicorn, "conv4_6", "Unicorn");//589824
//	Copy_Params_Str(relu4_6_weights, Unicorn, quantize_level, "relu4_6_weights", "Unicorn");//256
//	Copy_Int8_Params_Str(conv4_7, Unicorn, "conv4_7", "Unicorn");//589824
//	Copy_Params_Str(relu4_7_weights, Unicorn, quantize_level, "relu4_7_weights", "Unicorn");//256
//	Copy_Int8_Params_Str(conv4_8, Unicorn, "conv4_8", "Unicorn");//589824
//	Copy_Params_Str(relu4_8_weights, Unicorn, quantize_level, "relu4_8_weights", "Unicorn");//256
//	Copy_Int8_Params_Str(conv4_9, Unicorn, "conv4_9", "Unicorn");//589824
//	Copy_Params_Str(relu4_9_weights, Unicorn, quantize_level, "relu4_9_weights", "Unicorn");//256
//	Copy_Int8_Params_Str(conv4_10, Unicorn, "conv4_10", "Unicorn");//589824
//	Copy_Params_Str(relu4_10_weights, Unicorn, quantize_level, "relu4_10_weights", "Unicorn");//256,512
//	Copy_Int8_Params_Str(conv4, Unicorn, "conv4", "Unicorn");//1179648
//	Copy_Params_Str(relu4_weights, Unicorn, quantize_level, "relu4_weights", "Unicorn");//512
//	Copy_Int8_Params_Str(conv5_1, Unicorn, "conv5_1", "Unicorn");//2359296
//	Copy_Params_Str(relu5_1_weights, Unicorn, quantize_level, "relu5_1_weights", "Unicorn");//512
//	Copy_Int8_Params_Str(conv5_2, Unicorn, "conv5_2", "Unicorn");//2359296
//	Copy_Params_Str(relu5_2_weights, Unicorn, quantize_level, "relu5_2_weights", "Unicorn");//512
//	Copy_Int8_Params_Str(conv5_3, Unicorn, "conv5_3", "Unicorn");//2359296
//	Copy_Params_Str(relu5_3_weights, Unicorn, quantize_level, "relu5_3_weights", "Unicorn");//512
//	Copy_Int8_Params_Str(conv5_4, Unicorn, "conv5_4", "Unicorn");//2359296
//	Copy_Params_Str(relu5_4_weights, Unicorn, quantize_level, "relu5_4_weights", "Unicorn");//512
//	Copy_Int8_Params_Str(conv5_5, Unicorn, "conv5_5", "Unicorn");//2359296
//	Copy_Params_Str(relu5_5_weights, Unicorn, quantize_level, "relu5_5_weights", "Unicorn");//512
//	Copy_Int8_Params_Str(conv5_6, Unicorn, "conv5_6", "Unicorn");//2359296
//	Copy_Params_Str(relu5_6_weights, Unicorn, quantize_level, "relu5_6_weights", "Unicorn");//512
//	Copy_Int8_Params_Str(conv5, Unicorn, "conv5", "Unicorn");//2359296
//	Copy_Params_Str(relu5_weights, Unicorn, quantize_level, "relu5_weights", "Unicorn");//512
//	Copy_Params_Str(conv5_dw_weights, Unicorn, quantize_level, "conv5_dw_weights", "Unicorn");//864
//	Copy_Params_Str(conv5_dw_bias, Unicorn, quantize_level, "conv5_dw_bias", "Unicorn");//64
//}
//writedatafileend(out);
//}

#endif // !_PRUNE_HPP_