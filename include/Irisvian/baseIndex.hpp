#ifndef _BASE_INDEX_HPP_
#define _BASE_INDEX_HPP_
#include <iostream>
#include <vector>

namespace glasssix
{
	namespace Irisvian
	{
		class BaseIndex
		{
		public:

			BaseIndex() = default;

			BaseIndex(const std::vector<const float*> *baseData, int dimension) {}

			BaseIndex(int dimension) {}

			virtual ~BaseIndex() {}

#if defined(PROFILER) && defined(_MSC_VER)

			virtual void buildGraph(unsigned &maxMemoryUsage) = 0;

			virtual void buildGraph(const std::vector<const float*> *baseData, unsigned &maxMemoryUsage) = 0;
#else
			virtual void buildGraph() = 0;

			virtual void buildGraph(const std::vector<const float*> *baseData) = 0;
#endif 

			virtual void saveGraph(const char *nGraphPath) = 0;

			virtual void saveGraph(const char *nGraphPath, const char *basedataPath) = 0;

			unsigned navigateNode = 0;
			unsigned width = 0;
			bool isNormalized = false;
			std::vector<std::vector<unsigned > > finalGraph;
		};
	}
}

#endif // !_BASE_INDEX_HPP_