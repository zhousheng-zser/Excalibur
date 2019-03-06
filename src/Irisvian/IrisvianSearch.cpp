#include "index.hpp"
#include "search.hpp"
#include "IrisvianSearch.hpp"


namespace glasssix
{
	namespace Irisvian
	{
		IrisvianSearch::IrisvianSearch(const std::vector<const float*> *baseData, int dimension)
		{
			index_ = new Index(baseData, dimension);
			search_ = new Search(baseData, dimension);
		}

		IrisvianSearch::IrisvianSearch(int dimension)
		{
			index_ = new Index(dimension);
			search_ = new Search(dimension);
		}

		IrisvianSearch::~IrisvianSearch()
		{
			delete index_;
			delete search_;
		}

#if defined(PROFILER) && defined(_MSC_VER)
		void IrisvianSearch::buildGraph(unsigned &maxMemoryUsage)
#else
		void IrisvianSearch::buildGraph()
#endif
		{
#if defined(PROFILER) && defined(_MSC_VER)
			index_->buildGraph(maxMemoryUsage);
#else
			index_->buildGraph();
#endif
			search_->navigateNode = index_->navigateNode;
			search_->width = index_->width;
			search_->isNormalized = index_->isNormalized;
			search_->ngraph = index_->finalGraph;

			search_->ngraph.resize(index_->finalGraph.size());
			for (size_t i = 0; i < index_->finalGraph.size(); i++)
			{
				search_->ngraph[i].resize(index_->finalGraph[i].size());
				for (size_t j = 0; j < index_->finalGraph[i].size(); j++)
				{
					search_->ngraph[i][j] = index_->finalGraph[i][j];
				}
			}
		}

#if defined(PROFILER) && defined(_MSC_VER)
		void IrisvianSearch::buildGraph(const std::vector<const float*> *baseData, unsigned &maxMemoryUsage)
#else
		void IrisvianSearch::buildGraph(const std::vector<const float*> *baseData)
#endif
		{
#if defined(PROFILER) && defined(_MSC_VER)
			index_->buildGraph(baseData, maxMemoryUsage);
#else
			index_->buildGraph(baseData);
#endif

			search_->navigateNode = index_->navigateNode;
			search_->width = index_->width;
			search_->isNormalized = index_->isNormalized;
			search_->ngraph = index_->finalGraph;

			search_->ngraph.resize(index_->finalGraph.size());
			for (size_t i = 0; i < index_->finalGraph.size(); i++)
			{
				search_->ngraph[i].resize(index_->finalGraph[i].size());
				for (size_t j = 0; j < index_->finalGraph[i].size(); j++)
				{
					search_->ngraph[i][j] = index_->finalGraph[i][j];
				}
			}
		} 


		void IrisvianSearch::saveGraph(const char *graphPath)
		{
			index_->saveGraph(graphPath);
		}


		void IrisvianSearch::saveGraph(const char *graphPath, const char *basedataPath)
		{
			index_->saveGraph(graphPath, basedataPath);
		}


		void IrisvianSearch::loadGraph(const char* graphPath)
		{
			search_->loadGraph(graphPath);
		}

		void IrisvianSearch::loadGraph(const char* graphPath, const char *basedataPath)
		{
			search_->loadGraph(graphPath, basedataPath);
		}

		void IrisvianSearch::optimizeGraph()
		{
			search_->optimizeGraph();
		}

#ifndef PROFILER
		void IrisvianSearch::searchVector(const std::vector<const float*>* queryData, unsigned topK, 
			std::vector<std::vector<unsigned>> &returnIDs, std::vector<std::vector<float>> &returnDistancesInPercentage)
		{
			search_->searchVector(queryData, topK, returnIDs, returnDistancesInPercentage);
		}
#else
		void IrisvianSearch::searchVector(const std::vector<const float*>* queryData, unsigned topK, 
			std::vector<std::vector<unsigned>> &returnIDs, std::vector<std::vector<Neighbor>> &returnNeighbors)
		{
			search_->searchVector(queryData, topK, returnIDs, returnNeighbors);
		}
#endif // !PROFILER
		

		void IrisvianSearch::saveResult(const char* resultPath, std::vector<std::vector<unsigned> > &returnIDs)
		{
			search_->saveResult(resultPath, returnIDs);
		}
	}
}