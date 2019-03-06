#ifndef _INDEX_HPP_
#define _INDEX_HPP_
#include <iostream>
#include <fstream>
#include "nGraph.hpp"
#include "kGraph.hpp"
#include "distance.hpp"
#include "baseIndex.hpp"

namespace glasssix 
{
	namespace Irisvian
	{
			class Index : public BaseIndex
			{
			public:

				Index();

				Index(const std::vector<const float*> *baseData, int dimension);

				Index(int dimension);

				virtual ~Index();

#if defined(PROFILER) && defined(_MSC_VER)

				void buildGraph(unsigned &maxMemoryUsage) override;

				void buildGraph(const std::vector<const float*> *baseData, unsigned &maxMemoryUsage) override;
#else
				void buildGraph() override;

				void buildGraph(const std::vector<const float*> *baseData) override;
#endif 

				void saveGraph(const char *nGraphPath) override;

				void saveGraph(const char *nGraphPath, const char *basedataPath) override;

			private:
				const std::vector<const float*> *baseData_;
				unsigned baseNum_;
				unsigned dimension_;
				KGraph kgraph_;
				NGraph ngraph_;
				float *normArray_;
			};
	}
}
#endif // !_INDEX_HPP_
