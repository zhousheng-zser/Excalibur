#ifndef _SearchWrapper_HPP_
#define _SearchWrapper_HPP_

#include "search.hpp"
#include <vector>

namespace glasssix
{
	namespace Irisvian
	{
		class SearchWrapper
		{
		public:
			SearchWrapper() = delete;
			
			SearchWrapper(std::vector<const float *> *baseDataPtr, int dimension)
			{
				for(size_t i = 0; i < baseDataPtr->size(); i++)
				{
					baseData.push_back((*baseDataPtr)[i]);
				}
				search = new Search(baseDataPtr, dimension);
			}
			
			SearchWrapper(int dimension)
			{
				search = new Search(dimension);
			}
			
			~SearchWrapper()
			{
				delete search;
				
				for(size_t i = 0; i < baseData.size(); i++)
				{
					delete[] baseData[i];
				}
			}
			
			void loadGraph(const char* graphPath)
			{
				search->loadGraph(graphPath);
			}

			void loadGraph(const char* graphPath, const char *basedataPath)
			{
				search->loadGraph(graphPath, basedataPath);
			}
			
			const std::vector<const float*>* getBasedata()
			{
				return search->getBasedata();
			}
			
			void optimizeGraph()
			{
				search->optimizeGraph();
			}
			
			void searchVector(const std::vector<const float*>* queryData, unsigned topK,
					std::vector<std::vector<unsigned>> &returnIDs, std::vector<std::vector<float>> &returnSimilarities)
			{
				search->searchVector(queryData, topK, returnIDs, returnSimilarities);
			}

			void saveResult(const char* resultPath, std::vector<std::vector<unsigned> > &returnIDs)
			{
				search->saveResult(resultPath, returnIDs);
			}
			
		private:
			std::vector<const float *> baseData;
			Search *search;
		};
	}
}

#endif