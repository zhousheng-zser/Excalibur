#ifndef _SEARCH_HPP_
#define _SEARCH_HPP_

#include <cstring>
#include <iostream>
#include <fstream>
#include "nGraph.hpp"
#include "kGraph.hpp"
#include "distance.hpp"
#include "baseSearch.hpp"
#include <glasssix/tensor.hpp>

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

				const std::vector<const float*>* getBasedata() override;

				void optimizeGraph() override;

				void searchVector(const std::vector<const float*>* queryData, unsigned topK,
					std::vector<std::vector<unsigned>> &returnIDs, std::vector<std::vector<float>> &returnSimilarities) override;

				void saveResult(const char* resultPath, std::vector<std::vector<unsigned> > &returnIDs) override;

			private:
				unsigned dimension_;
				const std::vector<const float*>* queryData_;
				unsigned queryNum_;
				std::vector<const float*> baseDataPtr;

				std::shared_ptr<glasssix::excalibur::tensor<char>> optGraph_tensor_;
				char* optGraph_;

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