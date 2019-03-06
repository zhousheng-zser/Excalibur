#ifndef _BASE_SEARCH_HPP_
#define _BASE_SEARCH_HPP_
#include <iostream>
#include <vector>

namespace glasssix 
{
	namespace Irisvian 
	{

		class BaseSearch
		{
		public:
			BaseSearch() = default;

			BaseSearch(const std::vector<const float*> *baseData, int dimension) {}

			virtual ~BaseSearch() {}

			virtual void loadGraph(const char* graphPath) = 0;

			virtual void loadGraph(const char* graphPath, const char *basedataPath) = 0;

			virtual void optimizeGraph() = 0;

#ifdef PROFILER
			virtual void searchVector(const std::vector<const float*>* queryData, unsigned topK,
				std::vector<std::vector<unsigned>> &returnIDs, std::vector<std::vector<Neighbor>> &returnNeighbors) = 0;
#else
			virtual void searchVector(const std::vector<const float*>* queryData, unsigned topK,
				std::vector<std::vector<unsigned>> &returnIDs, std::vector<std::vector<float>> &returnDistancesInPercentage) = 0;
#endif // !PROFILER
			virtual void saveResult(const char* resultPath, std::vector<std::vector<unsigned> > &returnIDs) = 0;

			unsigned navigateNode = 0;
			unsigned width = 0;
			bool isNormalized = false;
			std::vector<std::vector<unsigned > > ngraph;
		};
	}
}

#endif // !_BASE_SEARCH_HPP_