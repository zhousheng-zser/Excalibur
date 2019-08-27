#include "index.hpp"
#include "search.hpp"
#include "IrisvielSearch.hpp"
#include <glasssix/mutex_wrapper.hpp>

namespace glasssix
{
	namespace irisviel
	{
		IrisvielSearch::IrisvielSearch(const std::vector<const float*> *baseData, int dimension)
		{
			index_.reset(new Index(baseData, dimension));
			search_.reset(new Search(baseData, dimension));
			mutex_wrapper_.reset(new mutex_wrapper());
		}

		IrisvielSearch::IrisvielSearch(int dimension)
		{
			index_.reset(new Index(dimension));
			search_.reset(new Search(dimension));
			mutex_wrapper_.reset(new mutex_wrapper());
		}

		IrisvielSearch::IrisvielSearch(const std::vector<const float*> *baseData, int dimension, const std::shared_ptr<mutex_wrapper> &lock)
			:IrisvielSearch(baseData, dimension)
		{
			mutex_wrapper_ = lock;
		}

		IrisvielSearch::IrisvielSearch(int dimension, const std::shared_ptr<mutex_wrapper> &lock)
			: IrisvielSearch(dimension)
		{
			mutex_wrapper_ = lock;
		}

		IrisvielSearch::~IrisvielSearch()
		{
		}

		int IrisvielSearch::buildGraph() const
		{
			auto lock = mutex_wrapper_->guard();
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


		int IrisvielSearch::buildGraph(const std::vector<const float*> *baseData) const
		{
			auto lock = mutex_wrapper_->guard();
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


		void IrisvielSearch::saveGraph(const char *graphPath) const
		{
			auto lock = mutex_wrapper_->guard();
			index_->saveGraph(graphPath);
		}

		void IrisvielSearch::saveGraph(std::string graphPath) const
		{
			saveGraph(graphPath.c_str());
		}


		void IrisvielSearch::saveGraph(const char *graphPath, const char *basedataPath) const
		{
			auto lock = mutex_wrapper_->guard();
			index_->saveGraph(graphPath, basedataPath);
		}

		void IrisvielSearch::saveGraph(std::string graphPath, std::string basedataPath) const
		{
			saveGraph(graphPath.c_str(), basedataPath.c_str());
		}


		void IrisvielSearch::loadGraph(const char* graphPath) const
		{
			auto lock = mutex_wrapper_->guard();
			search_->loadGraph(graphPath);
		}

		// For C++/CLI, implementations shuold not be done in header files.
		// Or it will report C2001 bugs for the un-support functions.

		void IrisvielSearch::loadGraph(std::string graphPath) const
		{
			loadGraph(graphPath.c_str());
		}

		void IrisvielSearch::loadGraph(const char* graphPath, const char *basedataPath) const
		{
			auto lock = mutex_wrapper_->guard();
			search_->loadGraph(graphPath, basedataPath);
		}

		void IrisvielSearch::loadGraph(std::string graphPath, std::string basedataPath) const
		{
			loadGraph(graphPath.c_str(), basedataPath.c_str());
		}

		const std::vector<const float*>* IrisvielSearch::getBasedata() const 
		{
			auto lock = mutex_wrapper_->guard();
			return search_->getBasedata();
		}

		void IrisvielSearch::optimizeGraph() const
		{
			auto lock = mutex_wrapper_->guard();
			search_->optimizeGraph();
		}

		void IrisvielSearch::searchVector(const std::vector<const float*>* queryData, unsigned topK,
			std::vector<std::vector<unsigned>> &returnIDs, std::vector<std::vector<float>> &returnSimilarities) const
		{
			auto lock = mutex_wrapper_->guard();
			search_->searchVector(queryData, topK, returnIDs, returnSimilarities);
		}

		void IrisvielSearch::saveResult(const char* resultPath, std::vector<std::vector<unsigned> > &returnIDs) const
		{
			auto lock = mutex_wrapper_->guard();
			search_->saveResult(resultPath, returnIDs);
		}

		void IrisvielSearch::saveResult(std::string resultPath, std::vector<std::vector<unsigned>>& returnIDs) const
		{
			saveResult(resultPath.c_str(), returnIDs);
		}

		std::string IrisvielSearch::getVersion()
		{
#ifdef TRIAL
			return std::string("Glasssix Trial FaceSDK");
#else
			return std::string("Glasssix");
#endif // TRIAL	
		}
	}
}