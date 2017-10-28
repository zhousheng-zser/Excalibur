#include "datafile.hpp"



datafile::datafile(std::string outpath)
{
	pos = 0;
	out = std::ofstream(outpath, std::ios::app);
}


datafile::~datafile()
{
}


void datafile::writedata(const float* data, int len)
{
	for (int i = 0; i < len; ++i, ++pos) {
		if (i == len - 1)
		{
			if (data[i]==0.0f)
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
			if (data[i]==0.0f)
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

void datafile::writedataend()
{
	out << "};" << std::endl << std::endl << std::endl;
	pos = 0;
}

void datafile::writedatahead(std::string netname, std::string layername)
{
	out << "static const float " + netname + "_" + layername + "[] = {" << std::endl;
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
			writedatahead(netname, layer_name + "_weights");
			const BlobProto& blob = layer_param.blobs(0);
			writedata(blob.data().data(), blob.data_size());
			writedataend();
			if (n>1)
			{
				writedatahead(netname, layer_name + "_bias");
				const BlobProto& bias = layer_param.blobs(1);
				writedata(bias.data().data(), bias.data_size());
				writedataend();
			}
		}
	}
	writedatafileend();
}
