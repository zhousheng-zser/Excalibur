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

		int IrisvianSearch::buildGraph() const
		{
			int maxMemoryUsage = index_->buildGraph();
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

			return maxMemoryUsage;
		}


		int IrisvianSearch::buildGraph(const std::vector<const float*> *baseData) const
		{
			int maxMemoryUsage = index_->buildGraph(baseData);
			search_->navigateNode = index_->navigateNode;
			search_->width = index_->width;
			search_->isNormalized = index_->isNormalized;
			search_->ngraph = index_->finalGraph;
			search_->baseNum_ = index_->baseNum_;
			search_->baseData_ = index_->baseData_;

			search_->ngraph.resize(index_->finalGraph.size());
			for (size_t i = 0; i < index_->finalGraph.size(); i++)
			{
				search_->ngraph[i].resize(index_->finalGraph[i].size());
				for (size_t j = 0; j < index_->finalGraph[i].size(); j++)
				{
					search_->ngraph[i][j] = index_->finalGraph[i][j];
				}
			}

			return maxMemoryUsage;
		} 


		void IrisvianSearch::saveGraph(const char *graphPath) const
		{
			index_->saveGraph(graphPath);
		}


		void IrisvianSearch::saveGraph(const char *graphPath, const char *basedataPath) const
		{
			index_->saveGraph(graphPath, basedataPath);
		}


		void IrisvianSearch::loadGraph(const char* graphPath) const
		{
			search_->loadGraph(graphPath);
		}

		void IrisvianSearch::loadGraph(const char* graphPath, const char *basedataPath) const
		{
			search_->loadGraph(graphPath, basedataPath);
		}

		const std::vector<const float*>* IrisvianSearch::getBasedata() const 
		{
			return search_->getBasedata();
		}

		void IrisvianSearch::optimizeGraph() const
		{
			search_->optimizeGraph();
		}

		void IrisvianSearch::searchVector(const std::vector<const float*>* queryData, unsigned topK,
			std::vector<std::vector<unsigned>> &returnIDs, std::vector<std::vector<float>> &returnSimilarities) const
		{
			search_->searchVector(queryData, topK, returnIDs, returnSimilarities);
		}

		void IrisvianSearch::saveResult(const char* resultPath, std::vector<std::vector<unsigned> > &returnIDs) const
		{
			search_->saveResult(resultPath, returnIDs);
		}

		std::string IrisvianSearch::getVersion()
		{
#ifdef TRIAL
			return std::string("Glasssix Trial FaceSDK");
#else
			return std::string("Glasssix");
#endif // TRIAL	
		}
	}
}