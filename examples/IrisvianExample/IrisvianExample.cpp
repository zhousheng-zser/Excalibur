#include <fstream>
#include <string>
#include <cfloat>
#include <cmath>
#include <climits>
#include "../../include/Irisvian/IrisvianSearch.hpp"
#include "../../include/Julius/simd_helper.hpp"
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

float calc_distance(const float *a, const float *b, int size)
{
#define AVX_DOT(addr1, addr2, dest, tmp1, tmp2) \
		  tmp1 = _mm256_loadu_ps(addr1);\
          tmp2 = _mm256_loadu_ps(addr2);\
		  dest = mm_fmadd_ps(tmp1, tmp2, dest);

	__m256 sum_aa = mm_setzero_ps();
	__m256 sum_bb = mm_setzero_ps();
	__m256 sum_ab = mm_setzero_ps();
	__m256 l0, l1, l2, l3;
	__m256 r0, r1, r2, r3;
	unsigned D = (size + 7) & ~7U; // # dim aligned up to 256 bits, or 8 floats
	unsigned DR = D % 32;
	unsigned DD = D - DR;
	const float *l = a;
	const float *r = b;
	const float *e_l = l + DD;
	const float *e_r = r + DD;
	switch (DR)
	{
	case 24:
		AVX_DOT(e_l + 16, e_l + 16, sum_aa, l2, l2);
		AVX_DOT(e_r + 16, e_r + 16, sum_bb, r2, r2);
		AVX_DOT(e_l + 16, e_r + 16, sum_ab, l2, r2);
	case 16:
		AVX_DOT(e_l + 8, e_l + 8, sum_aa, l1, l1);
		AVX_DOT(e_r + 8, e_r + 8, sum_bb, r1, r1);
		AVX_DOT(e_l + 8, e_r + 8, sum_ab, l1, r1);
	case 8:
		AVX_DOT(e_l, e_l, sum_aa, l0, l0);
		AVX_DOT(e_r, e_r, sum_bb, r0, r0);
		AVX_DOT(e_l, e_r, sum_ab, l0, r0);
	}
	for (unsigned i = 0; i < DD; i += 32, l += 32, r += 32)
	{
		AVX_DOT(l, l, sum_aa, l0, l0);
		AVX_DOT(l + 8, l + 8, sum_aa, l1, l1);
		AVX_DOT(l + 16, l + 16, sum_aa, l2, l2);
		AVX_DOT(l + 24, l + 24, sum_aa, l3, l3);

		AVX_DOT(r, r, sum_bb, r0, r0);
		AVX_DOT(r + 8, r + 8, sum_bb, r1, r1);
		AVX_DOT(r + 16, r + 16, sum_bb, r2, r2);
		AVX_DOT(r + 24, r + 24, sum_bb, r3, r3);

		AVX_DOT(l, r, sum_ab, l0, r0);
		AVX_DOT(l + 8, r + 8, sum_ab, l1, r1);
		AVX_DOT(l + 16, r + 16, sum_ab, l2, r2);
		AVX_DOT(l + 24, r + 24, sum_ab, l3, r3);
	}

	float res_aa = _mm256_sumall_ps(sum_aa);
	float res_bb = _mm256_sumall_ps(sum_bb);
	float res_ab = _mm256_sumall_ps(sum_ab);

	float result = res_ab / (sqrt(res_aa * res_bb));
	return result;
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
			float distance = calc_distance(queryData[i], baseData[j], dim);

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
			else if (accurateSimilarities[i][n_accurate] > searchSimilarities[i][n_search]) {
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

void nsg_build_search()
{
	unsigned dim[18] = { 128,128,512,128,512,128,128,512,512,128,512,128,128,512,512,128,512,512 };
	unsigned numItem[18] = { 1000,2000,1000,5000,2000,10000,20000,5000,10000,50000,20000,100000,200000,50000,100000,500000,200000,500000 };

	glasssix::Timer calcTime;
	double indexTime = 0, searchTime = 0;

	std::ofstream out("./record.txt", std::fstream::out | std::fstream::trunc);

	std::vector<const float*> baseData;
	std::vector<const float*> queryData;

	for (int i = 0; i < 18; ++i)
	{
		std::string sourceDataPath = "./map512.bin";
		std::string nsgPath = "./nsg_" + std::to_string(dim[i]) + "_" + std::to_string(numItem[i]) + ".bin";
		std::string searchResultPath = "./search_nsg_" + std::to_string(dim[i]) + "_" + std::to_string(numItem[i]) + ".bin";

		baseData.clear();
		queryData.clear();
		unsigned baseNum = numItem[i];
		unsigned queryNum = numItem[i] / 10;
		unsigned dimension = dim[i];
		loadData(sourceDataPath.c_str(), baseData, baseNum, queryData, queryNum, dimension);

		calcTime.Start();
		IrisvianSearch oIndex(&baseData, dimension);
		oIndex.buildGraph();
		calcTime.Stop();
		indexTime = calcTime.GetElapsedMilliseconds() / 1000;

		//oIndex.saveGraph(nsgPath.c_str());
		//oIndex.loadGraph(nsgPath.c_str());

		oIndex.optimizeGraph();
		unsigned topK = 10;
		std::vector<std::vector<unsigned>> returnIDs;
		std::vector<std::vector<float>> returnSimilarities;
		std::vector<std::vector<float>> accurateSimilarities;

		calcTime.Start();
		oIndex.searchVector(&queryData, topK, returnIDs, returnSimilarities);
		calcTime.Stop();
		searchTime = calcTime.GetElapsedMilliseconds() / queryNum;

		get_accurate_similarity(baseData, baseNum, queryData, queryNum, dimension, topK, accurateSimilarities);
		float accuracy = get_accuracy(accurateSimilarities, returnSimilarities, topK, queryNum);

		out << "baseNum: " << baseNum << ", dimension: " << dimension << ", index: " << indexTime << ", search: " << searchTime << ", accuracy: " << accuracy << std::endl;
		std::cout << "baseNum: " << baseNum << ", dimension: " << dimension << ", index: " << indexTime << ", search: " << searchTime << ", accuracy: " << accuracy << std::endl;
		out.flush();

		//oIndex.saveResult(searchResultPath.c_str(), returnIDs);
	}
}


int main()
{
	nsg_build_search();

	//search_first_time();
	//search_second_time();

	system("pause");
	return 0;
}