#pragma once
#include <ios>
#include "readcaffemodel.hpp"

class datafile
{
	int pos;
	std::ofstream out;

	void writedatafilehead(std::string filename, std::string namespace_);

	void writedatafileend();

	void writedatahead(std::string netname, std::string layername);

	void writedataend();

	void writedata(const float* data, int len);
	
public:
	datafile();
	explicit datafile(std::string outpath);
	~datafile();

	void writedatahpp(std::string netpath, std::string netname, std::string namespacename);
};

