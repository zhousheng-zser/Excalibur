#include <fstream>
#include <string>
#include "../../include/Irisvian/IrisvianSearch.hpp"
#include "../../include/Irisvian/distance.hpp"
#include <glasssix/profiler.hpp>
#include <glasssix/timer.hpp>

using namespace glasssix;
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

float inner(const float *benchmark, const float *m, int size)
{
	float res = 0.0f;
	for (int q = 0; q < size; q++)
	{
		res += benchmark[q] * m[q];
	}

	return res;
}

float calc_diff(const float *benchmark, const float *m, int size)
{
	return (inner(benchmark, m, size) / sqrt(inner(benchmark, benchmark, size) * inner(m, m, size)));
}

void get_accurate_similarity(std::vector<const float*> &baseData, unsigned baseNumItem,
	std::vector<const float*> &queryData, unsigned queryNumItem,
	unsigned dim, unsigned topK, std::vector<std::vector<float>> &accurateSimilarities)
{
	accurateSimilarities.clear();
	accurateSimilarities.resize(queryNumItem);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < queryNumItem; i++)
	{
		std::vector<float> &similarity = accurateSimilarities[i];
		similarity.resize(topK);
		for (int j = 0; j < topK; j++)
		{
			similarity[j] = FLT_MIN;
		}

		for (int j = 0; j < baseNumItem; j++)
		{
			float distance = calc_diff(queryData[i], baseData[j], dim);

			if (j == 0)
			{
				similarity[0] = distance;
			}
			else
			{
				if (distance <= similarity[topK - 1])
				{
					continue;
				}

				int pos = 0;
				while ((pos < topK) && (distance <= similarity[pos]))
				{
					pos++;
				}

				for (int k = topK - 1; k > pos; k--)
				{
					similarity[k] = similarity[k - 1];
				}

				similarity[pos] = distance;
			}
		}
	}

	//for (int i = 0; i < queryNumItem; i++)
	//{
	//	for (int j = 0; j < topK; j++)
	//	{
	//		std::cout << accurateSimilarities[i][j] << " ";
	//	}
	//	std::cout << std::endl;
	//}
}

float get_accuracy(std::vector<std::vector<float>> &accurateSimilarities, std::vector<std::vector<float>> &searchSimilarities,
	int topK, unsigned queryNumItem)
{
	float accuracy = 0;

	for (int i = 0; i < queryNumItem; i++)
	{
		unsigned foundInKnn = 0;
		unsigned n_search = 0;
		unsigned n_accurate = 0;
		while (n_search < topK && n_accurate < topK) {
			if (abs(accurateSimilarities[i][n_accurate] - searchSimilarities[i][n_search]) < 1e-5) {
				++foundInKnn;
				++n_accurate;
				++n_search;
			}
			else if (accurateSimilarities[i][n_accurate] < searchSimilarities[i][n_search]) {
				++n_accurate;
			}
			else {
				std::cout << "distance error" << std::endl;
				break;
			}
		}

		accuracy += (float)foundInKnn / topK;
	}

	accuracy /= queryNumItem;

	return accuracy;
}

void search_first_time()
{
	std::string sourceDataPath = "D:/projects/nsg/map512.bin";
	std::vector<const float*> baseData;
	std::vector<const float*> queryData;
	const unsigned baseNum = 2000;
	const unsigned queryNum = baseNum / 10;
	const unsigned data_dimension = 512;
	loadData(sourceDataPath.c_str(), baseData, baseNum, queryData, queryNum, data_dimension);

	double build_time, search_time;
	glasssix::Timer calcTime;
	calcTime.Start();
	IrisvianSearch irisvian(&baseData, data_dimension);
	irisvian.buildGraph();
	calcTime.Stop();
	build_time = calcTime.GetElapsedMilliseconds() / 1000;

	irisvian.optimizeGraph();
	const unsigned topK = 10;
	std::vector<std::vector<unsigned>> returnIDs;
	std::vector<std::vector<float>> returnSimilarities;

	calcTime.Start();
	irisvian.searchVector(&queryData, topK, returnIDs, returnSimilarities);
	calcTime.Stop();
	search_time = calcTime.GetElapsedMilliseconds() / queryNum;

	irisvian.saveGraph("D:/projects/nsg/search_index.graph");

	std::vector<std::vector<float>> accurateSimilarities;
	get_accurate_similarity(baseData, baseNum, queryData, queryNum, data_dimension, topK, accurateSimilarities);
	float accuracy = get_accuracy(accurateSimilarities, returnSimilarities, topK, queryNum);

	std::cout << "build_time:" << build_time << ", search_time:" << search_time << ", accuracy: " << accuracy << std::endl;

	std::cout << "result:" << std::endl;
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			std::cout << returnIDs[i][j] << " ," << returnSimilarities[i][j] << "   ";
		}
		std::cout << std::endl;
	}
}

void search_second_time()
{
	const unsigned queryNum = 10;
	const unsigned data_dimension = 512;
	IrisvianSearch irisvian(data_dimension);
	irisvian.loadGraph("D:/projects/nsg/search_index.graph", "D:/projects/nsg/map512.bin");
	irisvian.optimizeGraph();
	const unsigned topK = 10;
	const std::vector<const float*>* baseData = irisvian.getBasedata();
	std::vector<const float*> queryData;
	loadData("D:/projects/nsg/map512.bin", queryData, queryNum, data_dimension);
	std::vector<std::vector<unsigned>> returnIDs;
	std::vector<std::vector<float>> returnDistancesInPercentage;
	irisvian.searchVector(&queryData, topK, returnIDs, returnDistancesInPercentage);
}




int main()
{
	search_first_time();
	//search_second_time();

	//system("pause");
	return 0;
}