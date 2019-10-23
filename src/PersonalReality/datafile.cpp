#include "datafile.hpp"

#define DATA_TYPE "float"
//#define DATA_TYPE "unsigned short"

datafile::datafile(std::string outpath)
{
	pos = 0;
	out = std::ofstream(outpath, std::ios::app);
}


datafile::~datafile()
{
}

// convert float to half precision floating point
unsigned short datafile::float2half(float value)
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

bool datafile::quantize_weight(float *data, size_t data_length, int quantize_level, std::vector<float> &quantize_table, std::vector<unsigned char> &quantize_index)
{
	assert(quantize_level != 0);
	assert(data != NULL);
	assert(data_length > 0);

	if (data_length < static_cast<size_t>(quantize_level)) {
		fprintf(stderr, "No need quantize,because: data_length < quantize_level");
		return false;
	}

	quantize_table.reserve(quantize_level);
	quantize_index.reserve(data_length);

	// 1. Find min and max value
	float max_value = std::numeric_limits<float>::min();
	float min_value = std::numeric_limits<float>::max();

	for (size_t i = 0; i < data_length; ++i)
	{
		if (max_value < data[i]) max_value = data[i];
		if (min_value > data[i]) min_value = data[i];
	}
	float strides = (max_value - min_value) / quantize_level;

	// 2. Generate quantize table
	for (int i = 0; i < quantize_level; ++i)
	{
		quantize_table.push_back(min_value + i * strides);
	}

	// 3. Align data to the quantized value
	for (size_t i = 0; i < data_length; ++i)
	{
		size_t table_index = int((data[i] - min_value) / strides);
		table_index = std::min<float>(table_index, quantize_level - 1);

		float low_value = quantize_table[table_index];
		float high_value = low_value + strides;

		// find a nearest value between low and high value.
		float targetValue = data[i] - low_value < high_value - data[i] ? low_value : high_value;

		table_index = int((targetValue - min_value) / strides);
		table_index = std::min<float>(table_index, quantize_level - 1);
		quantize_index.push_back(table_index);
	}

	return true;
}

void datafile::writedata(const float* data, int len, std::string datatype, float scale)
{
	if (datatype == "float")
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
	if (datatype == "unsigned short")
	{
		for (int i = 0; i < len; ++i, ++pos)
		{
			if (i == len - 1)
				out << float2half(data[i]);
			else
				out << float2half(data[i]) << ", ";
			if (pos >= 15) {
				pos = 0;
				out << std::endl;
			}
		}
	}
	if (datatype == "signed char")
	{
		for (int i = 0; i < len; ++i, ++pos)
		{
			if (i == len - 1)
				out << int(float2int8(data[i] * scale));
			else
				out << int(float2int8(data[i] * scale)) << ", ";
			if (pos >= 15) {
				pos = 0;
				out << std::endl;
			}
		}
	}
}

void datafile::writedataend()
{
	out << "};" << std::endl << std::endl << std::endl;
	pos = 0;
}

void datafile::writedatahead(std::string netname, std::string layername, std::string datatype)
{
	out << "static const " + datatype + " " + netname + "_" + layername + "[] = {" << std::endl;
}

void datafile::writedatafileend()
{
	out << "}" << std::endl;
	out << "}" << std::endl;
	out << "#endif" << std::endl;
}

void datafile::writedatafilehead(std::string filename, std::string namespace_)
{
	std::transform(filename.begin(), filename.end(), filename.begin(), toupper);
	out << "#ifndef _" + filename + "_DATA_HPP_" << std::endl;
	out << "#define _" + filename + "_DATA_HPP_" << std::endl;
	out << std::endl << std::endl;
	out << "namespace glasssix {" << std::endl;
	out << "namespace " << namespace_ << " {" << std::endl;
}

void datafile::writedatahpp(std::string netpath, std::string netname, std::string namespacename)
{
	NetParameter param;
	ReadProtoFromBinaryFile(netpath.c_str(), &param);
	writedatafilehead(netname, namespacename);
	for (int i = 1; i < param.layer_size(); i++)
	{
		LayerParameter& layer_param = *param.mutable_layer(i);
		int n = param.mutable_layer(i)->mutable_blobs()->size();
		if (layer_param.type() == "BatchNorm")
		{

			std::string layer_name = layer_param.name();
			std::transform(layer_name.begin(), layer_name.end(), layer_name.begin(), tolower);

			const float bn_scale_factor = layer_param.mutable_blobs(2)->data(0) == 0 ?
				0 : 1 / layer_param.mutable_blobs(2)->data(0);

			
			const BlobProto& mean = layer_param.blobs(0);
			const BlobProto& variance = layer_param.blobs(1);

			float *temp_weights = new float[variance.data_size()];
			std::memcpy(temp_weights, variance.data().data(), variance.data_size() * sizeof(variance.data().data()[0]));
			for (int j = 0; j < variance.data_size(); j++)
			{
				temp_weights[j] = pow(variance.data(j) * bn_scale_factor + layer_param.batch_norm_param().eps(), 0.5);
			}

			float *temp_bias = new float[mean.data_size()];
			std::memcpy(temp_bias, mean.data().data(), mean.data_size() * sizeof(mean.data().data()[0]));
			for (int j = 0; j < mean.data_size(); j++)
			{
				temp_bias[j] *= -1.0f * bn_scale_factor / temp_weights[j];
			}

			for (int j = 0; j < variance.data_size(); j++)
			{
				temp_weights[j] = 1.0f / temp_weights[j];
			}
			
			writedatahead(netname, layer_name + "_weights", DATA_TYPE);
			writedata(temp_weights, variance.data_size(), DATA_TYPE);
			delete[] temp_weights;
			writedataend();

			writedatahead(netname, layer_name + "_bias", DATA_TYPE);
			writedata(temp_bias, mean.data_size(), DATA_TYPE);
			delete[] temp_bias;
			writedataend();
		}
		else
		{
			if (n)
			{
				std::string layer_name = layer_param.name();
				std::transform(layer_name.begin(), layer_name.end(), layer_name.begin(), tolower);
				writedatahead(netname, layer_name + "_weights", DATA_TYPE);
				const BlobProto& blob = layer_param.blobs(0);
				writedata(blob.data().data(), blob.data_size(), DATA_TYPE);

				writedataend();
				if (n>1)
				{
					writedatahead(netname, layer_name + "_bias", DATA_TYPE);
					const BlobProto& bias = layer_param.blobs(1);
					writedata(bias.data().data(), bias.data_size(), DATA_TYPE);
					writedataend();
				}
			}
		}
	}
	writedatafileend();
}


signed char datafile::float2int8(float v)
{
	int int32 = round(v);
	if (int32 > 127) return 127;
	if (int32 < -128) return -128;
	return (signed char)int32;
}

void datafile::insert_scales()
{
	Unicorn_scales["conv1a"] = 168.626725724f;
	Unicorn_scales["conv1b"] = 194.307459918f;
	Unicorn_scales["conv2_1"] = 274.472378716f;
	Unicorn_scales["conv2_2"] = 176.114160843f;
	Unicorn_scales["conv2"] = 264.601509306f;
	Unicorn_scales["conv3_1"] = 274.026229762f;
	Unicorn_scales["conv3_2"] = 371.508281583f;
	Unicorn_scales["conv3_3"] = 292.500820515f;
	Unicorn_scales["conv3_4"] = 369.318496505f;
	Unicorn_scales["conv3"] = 487.45436938f;
	Unicorn_scales["conv4_1"] = 487.379385175f;
	Unicorn_scales["conv4_2"] = 699.1747656f;
	Unicorn_scales["conv4_3"] = 426.879147006f;
	Unicorn_scales["conv4_4"] = 599.783328188f;
	Unicorn_scales["conv4_5"] = 616.978328872f;
	Unicorn_scales["conv4_6"] = 521.076596219f;
	Unicorn_scales["conv4_7"] = 624.644821949f;
	Unicorn_scales["conv4_8"] = 379.590710985f;
	Unicorn_scales["conv4_9"] = 653.362954491f;
	Unicorn_scales["conv4_10"] = 560.751397762f;
	Unicorn_scales["conv4"] = 387.85863108f;
	Unicorn_scales["conv5_1"] = 442.958470508f;
	Unicorn_scales["conv5_2"] = 495.714291497f;
	Unicorn_scales["conv5_3"] = 509.101942536f;
	Unicorn_scales["conv5_4"] = 295.730290681f;
	Unicorn_scales["conv5_5"] = 429.575008314f;
	Unicorn_scales["conv5_6"] = 502.631792597f;
	Unicorn_scales["conv5"] = 745.230312878f;
}

void datafile::write_int8_datahpp(std::string netpath, std::string netname, std::string namespacename)
{
	insert_scales();

	NetParameter param;
	ReadProtoFromBinaryFile(netpath.c_str(), &param);
	writedatafilehead(netname, namespacename);
	for (int i = 1; i < param.layer_size(); i++)
	{
		LayerParameter& layer_param = *param.mutable_layer(i);
		int n = param.mutable_layer(i)->mutable_blobs()->size();
		if (n)
		{
			std::string layer_name = layer_param.name();
			std::transform(layer_name.begin(), layer_name.end(), layer_name.begin(), tolower);

			float scale;
			std::map<std::string, float>::iterator iter;
			iter = Unicorn_scales.find(layer_name);
			if (iter == Unicorn_scales.end())
			{
				scale = 1.0f;
				writedatahead(netname, layer_name + "_weights", "float");
				const BlobProto& blob = layer_param.blobs(0);
				writedata(blob.data().data(), blob.data_size(), "float");
				writedataend();
			}
			else
			{
				scale = Unicorn_scales[layer_name];
				writedatahead(netname, layer_name + "_weights", "signed char");
				const BlobProto& blob = layer_param.blobs(0);
				writedata(blob.data().data(), blob.data_size(), "signed char", scale);
				writedataend();
			}

			std::cout << "layer_name:" << layer_name << ",scale:" << scale << std::endl;

			
			if (n > 1)
			{
				writedatahead(netname, layer_name + "_bias", "float");
				const BlobProto& bias = layer_param.blobs(1);
				writedata(bias.data().data(), bias.data_size(), "float");
				writedataend();
			}
		}
	}
	writedatafileend();
}

