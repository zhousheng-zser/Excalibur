#ifndef _IRISVIAN_SEARACH_HPP_
#define _IRISVIAN_SEARACH_HPP_

#include <iostream>
#include <vector>
#include "baseIndex.hpp"
#include "baseSearch.hpp"


#ifdef EXPORT_IRISVIAN
#undef EXPORT_IRISVIAN
#ifdef _MSC_VER
#define EXPORT_IRISVIAN __declspec(dllexport)
#else
#define EXPORT_IRISVIAN
#endif
#else
#ifdef _MSC_VER
#define EXPORT_IRISVIAN __declspec(dllimport)
#else
#define EXPORT_IRISVIAN
#endif
#endif

namespace glasssix
{
	namespace Irisvian
	{
		class EXPORT_IRISVIAN IrisvianSearch
		{
		public:

			IrisvianSearch(const std::vector<const float*> *baseData, int dimension);

			IrisvianSearch(int dimension);

			~IrisvianSearch();


#if defined(PROFILER) && defined(_MSC_VER)
			void buildGraph(unsigned &maxMemoryUsage);

			void buildGraph(const std::vector<const float*> *baseData, unsigned &maxMemoryUsage);
#else
			void buildGraph();

			void buildGraph(const std::vector<const float*> *baseData);
#endif 


			void saveGraph(const char *graphPath);


			void saveGraph(std::string graphPath)
			{
				saveGraph(graphPath.c_str());
			}


			void saveGraph(const char *graphPath, const char *basedataPath);


			void saveGraph(std::string graphPath, std::string basedataPath)
			{
				saveGraph(graphPath.c_str(), basedataPath.c_str());
			}


			void loadGraph(const char* graphPath);


			void loadGraph(std::string graphPath)
			{
				loadGraph(graphPath.c_str());
			}


			void loadGraph(const char* graphPath, const char *basedataPath);


			void loadGraph(std::string graphPath, std::string basedataPath)
			{
				loadGraph(graphPath.c_str(), basedataPath.c_str());
			}


			void optimizeGraph();

#ifndef PROFILER
			void searchVector(const std::vector<const float*>* queryData, unsigned topK,
				std::vector<std::vector<unsigned>> &returnIDs, std::vector<std::vector<float>> &returnDistancesInPercentage);
#else
			void searchVector(const std::vector<const float*>* queryData, unsigned topK,
				std::vector<std::vector<unsigned>> &returnIDs, std::vector<std::vector<Neighbor>> &returnNeighbors);
#endif // !PROFILER


			void saveResult(const char* resultPath, std::vector<std::vector<unsigned> > &returnIDs);

			void saveResult(std::string resultPath, std::vector<std::vector<unsigned> > &returnIDs)
			{
				saveResult(resultPath.c_str(), returnIDs);
			}

		private:
			BaseIndex *index_;
			BaseSearch *search_;
		};
	}
}

#endif // !_IRISVIAN_SEARACH_HPP_