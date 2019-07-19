#ifndef _IRISVIANSEARCHWRAPPER_HPP_
#define _IRISVIANSEARCHWRAPPER_HPP_

#include "IrisvianSearch.hpp"
#include <vector>

namespace glasssix
{
	namespace Irisvian
	{
		class IrisvianSearchWrapper
		{
		public:
			IrisvianSearchWrapper() = delete;
			
			IrisvianSearchWrapper(std::vector<const float *> *baseDataPtr, int dimension)
			{
				for(size_t i = 0; i < baseDataPtr->size(); i++)
				{
					baseData.push_back((*baseDataPtr)[i]);
				}
				search = new IrisvianSearch(baseDataPtr, dimension);
				dimension_ = dimension;
			}
			
			IrisvianSearchWrapper(int dimension)
			{				
				search = new IrisvianSearch(dimension);
				dimension_ = dimension;
			}
			
			~IrisvianSearchWrapper()
			{
				delete search;
				
				for(size_t i = 0; i < baseData.size(); i++)
				{
					delete[] baseData[i];
				}
			}
			
			int buildGraph()
			{
				search->buildGraph();
			}
			
			int buildGraph(const std::vector<const float*> *baseDataPtr)
			{
				for(size_t i = 0; i < baseData.size(); i++)
				{
					delete[] baseData[i];
				}
				
				baseData.resize(0);
				
				for(size_t i = 0; i < baseDataPtr->size(); i++)
				{
					baseData.push_back((*baseDataPtr)[i]);
				}
				
				search->buildGraph(baseDataPtr);
			}
			
			void saveGraph(const char *graphPath)
			{
				search->saveGraph(graphPath);
			}
			
			void saveGraph(const char *graphPath, const char *basedataPath)
			{
				search->saveGraph(graphPath, basedataPath);
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
			
			const int getDimension()
			{
				return dimension_;
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
			IrisvianSearch *search;
			int dimension_;
		};
	}
}

#endif