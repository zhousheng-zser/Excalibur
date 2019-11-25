#ifndef _INDEX_HPP_
#define _INDEX_HPP_
#include <iostream>
#include <fstream>
#include "nGraph.hpp"
#include "kGraph.hpp"
#include "distance.hpp"
#include <glasssix/tensor.hpp>

namespace glasssix 
{
	namespace irisviel
	{
		class Index
		{
		public:

			Index();

			Index(const std::vector<const float*> *baseData, int dimension);

			Index(int dimension);

			virtual ~Index();

			int buildGraph();

			int buildGraph(const std::vector<const float*> *baseData);

			void saveGraph(const char *nGraphPath);

			void saveGraph(const char *nGraphPath, const char *basedataPath);

			unsigned navigateNode = 0;
			unsigned width = 0;
			bool isNormalized = false;
			std::vector<std::vector<unsigned > > finalGraph;
			const std::vector<const float*> *baseData_;
			unsigned baseNum_;

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
