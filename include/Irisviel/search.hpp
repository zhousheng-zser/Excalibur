#ifndef _SEARCH_HPP_
#define _SEARCH_HPP_

#include <cstring>
#include <iostream>
#include <fstream>
#include "nGraph.hpp"
#include "kGraph.hpp"
#include "distance.hpp"
#include <glasssix/tensor.hpp>

namespace glasssix
{
	namespace irisviel 
	{
			class Search
			{
			public:
				Search(const std::vector<const float*> *baseData, int dimension);

				Search(int dimension);

				virtual ~Search();

				bool loadGraph(const char* graphPath);

				bool loadGraph(const char* graphPath, const char *basedataPath);

				const std::vector<const float*>* getBasedata();

				void optimizeGraph();

				void searchVector(const std::vector<const float*>* queryData, unsigned topK,
					std::vector<std::vector<unsigned>> &returnIDs, std::vector<std::vector<float>> &returnSimilarities);

				void saveResult(const char* resultPath, std::vector<std::vector<unsigned> > &returnIDs);

				unsigned navigateNode = 0;
				unsigned width = 0;
				bool isNormalized = false;
				std::vector<std::vector<unsigned > > ngraph;
				const std::vector<const float*>* baseData_;
				unsigned baseNum_;

			private:
				unsigned dimension_;
				const std::vector<const float*>* queryData_;
				unsigned queryNum_;
				std::vector<const float*> baseDataPtr;

				std::shared_ptr<glasssix::excalibur::tensor<char>> optGraph_tensor_;
				char* optGraph_ = nullptr;

				size_t nodeSize;
				size_t dataLen;
				size_t neighborLen;

				unsigned neighborsMaxLength = 0;
				typedef std::vector<std::vector<unsigned > > CompactGraph;

				void searchWithOptGraph(const float *singleQueryData, unsigned topK,
					std::vector<unsigned> &returnIDs, std::vector<float> &returnSimilarities);

			};
	}
}

#endif // !_SEARCH_HPP_