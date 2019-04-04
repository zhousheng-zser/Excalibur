#pragma once
#include <ios>
#include "readcaffemodel.hpp"
#include <map>

class datafile
{
	int pos;
	std::ofstream out;

	void writedatafilehead(std::string filename, std::string namespace_);

	void writedatafileend();

	void writedatahead(std::string netname, std::string layername, std::string datatype = "float");

	void writedataend();

	void writedata(const float* data, int len, std::string datatype = "float", float scale = 1.0f);
	
	static unsigned short float2half(float value);

	static bool quantize_weight(float *data, size_t data_length, int quantize_level, std::vector<float> &quantize_table, std::vector<unsigned char> &quantize_index);

	static signed char float2int8(float v);

	std::map<std::string, float> Unicorn_scales;

	void insert_scales();

public:
	datafile();
	explicit datafile(std::string outpath);
	~datafile();

	void writedatahpp(std::string netpath, std::string netname, std::string namespacename);

	void write_int8_datahpp(std::string netpath, std::string netname, std::string namespacename);
};

