#include "datafile.hpp"



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

void datafile::writedata(const float* data, int len, std::string datatype)
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
	out << "#endif" << std::endl;
}

void datafile::writedatafilehead(std::string filename, std::string namespace_)
{
	std::transform(filename.begin(), filename.end(), filename.begin(), toupper);
	out << "#ifndef _" + filename + "_DATA_HPP_" << std::endl;
	out << "#define _" + filename + "_DATA_HPP_" << std::endl;
	out << std::endl << std::endl;
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
		if (n)
		{
			std::string layer_name = layer_param.name();
			std::transform(layer_name.begin(), layer_name.end(), layer_name.begin(), tolower);
			writedatahead(netname, layer_name + "_weights", "unsigned short");
			const BlobProto& blob = layer_param.blobs(0);
			writedata(blob.data().data(), blob.data_size(), "unsigned short");
			writedataend();
			if (n>1)
			{
				writedatahead(netname, layer_name + "_bias", "unsigned short");
				const BlobProto& bias = layer_param.blobs(1);
				writedata(bias.data().data(), bias.data_size(), "unsigned short");
				writedataend();
			}
		}
	}
	writedatafileend();
}
