#include <fstream>
#include <string>
#include <glasssix/timer.hpp>
#include "../../include/Irisvian/IrisvianSearch.hpp"

using namespace glasssix::Irisvian;

void static loadData(const char* filename, std::vector<const float*> &baseData, unsigned baseNumItem,
	std::vector<const float*> &queryData, unsigned queryNumItem, unsigned dim) 
{
	std::ifstream in(filename, std::ios::in | std::ios::binary);
	if (!in.is_open())
	{
		std::cout << "open file error" << std::endl; exit(-1);
	}
	in.seekg(0, std::ios::beg);
	for (int i = 0; i < baseNumItem; ++i)
	{
		float *temp_data = (float*)malloc(dim * sizeof(float));
		in.read((char*)(temp_data), dim * sizeof(float));
		baseData.push_back(const_cast<const float*>(temp_data));
	}
	for (int i = 0; i < queryNumItem; ++i)
	{
		float *temp_data = (float*)malloc(dim * sizeof(float));
		in.read((char*)(temp_data), dim * sizeof(float));
		queryData.push_back(const_cast<const float*>(temp_data));
	}
	in.close();
}

void static loadData(const char* filename, std::vector<const float*> &queryData, unsigned queryNumItem, unsigned dim) 
{
	std::ifstream in(filename, std::ios::in | std::ios::binary);
	if (!in.is_open())
	{
		std::cout << "open file error" << std::endl; exit(-1);
	}
	in.seekg(0, std::ios::beg);
	for (int i = 0; i < queryNumItem; ++i)
	{
		float *temp_data = (float*)malloc(dim * sizeof(float));
		in.read((char*)(temp_data), dim * sizeof(float));
		queryData.push_back(const_cast<const float*>(temp_data));
	}
	in.close();
}

void search_first_time()
{
	std::string sourceDataPath = "../../data/map512.bin";
	std::vector<const float*> baseData;
	std::vector<const float*> queryData;
	const unsigned baseNum = 10000;
	const unsigned queryNum = 10;
	const unsigned data_dimension = 512;
	loadData(sourceDataPath.c_str(), baseData, baseNum, queryData, queryNum, data_dimension);
	IrisvianSearch irisvian(&baseData, data_dimension);
	irisvian.buildGraph();
	irisvian.optimizeGraph();
	const unsigned topK = 10;
	std::vector<std::vector<unsigned>> returnIDs;
	std::vector<std::vector<float>> returnSimilarities;
	irisvian.searchVector(&queryData, topK, returnIDs, returnSimilarities);
	irisvian.saveGraph("./search_index.graph", "./base_data.bin");
}

void search_second_time()
{
	const unsigned queryNum = 10;
	const unsigned data_dimension = 512;
	IrisvianSearch irisvian(data_dimension);
	irisvian.loadGraph("./search_index.graph", "./base_data.bin");
	irisvian.optimizeGraph();
	const unsigned topK = 10;
	const std::vector<const float*>* baseData = irisvian.getBasedata();
	std::vector<const float*> queryData;
	loadData("../../data/map512.bin", queryData, queryNum, data_dimension);
	std::vector<std::vector<unsigned>> returnIDs;
	std::vector<std::vector<float>> returnDistancesInPercentage;
	irisvian.searchVector(&queryData, topK, returnIDs, returnDistancesInPercentage);
}

int main()
{
	search_first_time();
	search_second_time();
	return 0;
}