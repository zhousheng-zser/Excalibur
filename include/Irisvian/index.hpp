#ifndef _INDEX_HPP_
#define _INDEX_HPP_
#include <iostream>
#include <fstream>
#include "nGraph.hpp"
#include "kGraph.hpp"
#include "distance.hpp"
#include "baseIndex.hpp"
#include <glasssix/tensor.hpp>

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

				int buildGraph() override;

				int buildGraph(const std::vector<const float*> *baseData) override;

				void saveGraph(const char *nGraphPath) override;

				void saveGraph(const char *nGraphPath, const char *basedataPath) override;

			private:
				unsigned dimension_;
				KGraph kgraph_;
				NGraph ngraph_;
				std::shared_ptr<glasssix::excalibur::tensor<float>> normArray_tensor_;
				float *normArray_;
			};
	}
}
#endif // !_INDEX_HPP_
