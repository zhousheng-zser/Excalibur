#ifndef _SEARCH_HPP_
#define _SEARCH_HPP_

#include <cstring>
#include <iostream>
#include <fstream>
#include "nGraph.hpp"
#include "kGraph.hpp"
#include "distance.hpp"
#include "baseSearch.hpp"

namespace glasssix
{
	namespace Irisvian 
	{

			class Search : public BaseSearch
			{
			public:
				Search(const std::vector<const float*> *baseData, int dimension);

				Search(int dimension);

				virtual ~Search();

				void loadGraph(const char* graphPath) override;

				void loadGraph(const char* graphPath, const char *basedataPath) override;

				void optimizeGraph() override;

#ifndef PROFILER
				void searchVector(const std::vector<const float*>* queryData, unsigned topK, std::vector<std::vector<unsigned>> &returnIDs, std::vector<std::vector<float>> &returnDistancesInPercentage) override;
#else
				void searchVector(const std::vector<const float*>* queryData, unsigned topK, std::vector<std::vector<unsigned>> &returnIDs, std::vector<std::vector<Neighbor>> &returnNeighbors) override;
#endif // !PROFILER

				
				void saveResult(const char* resultPath, std::vector<std::vector<unsigned> > &returnIDs) override;

#ifdef PROFILER
				const std::vector<const float*>* baseData_;
				std::vector<const float*> baseDataPtr;
				const std::vector<const float*>* queryData_;
#endif // !PROFILER

			private:

#ifndef PROFILER
				std::vector<std::vector<float> > baseDataVector;
				std::vector<const float*> baseDataPtr;
				const std::vector<const float*>* baseData_;
				const std::vector<const float*>* queryData_;
#endif // !PROFILER
				unsigned baseNum_;
				unsigned queryNum_;
				unsigned dimension_;

				char* optGraph_;
				size_t nodeSize;
				size_t dataLen;
				size_t neighborLen;

				unsigned neighborsMaxLength = 0;
				typedef std::vector<std::vector<unsigned > > CompactGraph;

#ifndef PROFILER
				void searchWithOptGraph(const float *singleQueryData, unsigned topK,
					std::vector<unsigned> &returnIDs, std::vector<float> &returnDistancesInPercentage);
#else
				void searchWithOptGraph(
					const float *singleQueryData,
					unsigned topK,
					std::vector<unsigned> &returnIDs,
					std::vector<Neighbor> &returnNeighbors);
#endif // PROFILER

			private:
				int nIds_;
				/*std::bitset<32> f_1_ECX_;
				std::bitset<32> f_1_EDX_;
				std::array<int, 4> cpui;
				bool AVX();
				bool SSE();*/
				//instructionFlag_ = 0, NO_SIMD
				//instructionFlag_ = 1, SSE
				//instructionFlag_ = 2, AVX
				//unsigned instructionFlag_ = 0;
			};
	}
}

#endif // !_SEARCH_HPP_