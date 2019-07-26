#ifndef _KGRAPH_HPP_
#define _KGRAPH_HPP_

#include "neighbor.hpp"

namespace glasssix 
{
	namespace irisviel 
	{
		static unsigned const defaultK = 40;
		static unsigned const defaultPoolSize = 90;//结果kgraph文件中，每个点实际返回的neighbors的数量
		static unsigned const defaultReversePoolSize = 90;//reverse neighbors的数量上限
		static unsigned const defaultIterations = 30;
		static float const defaultDelta = 0.0002;
		static float const defaultRecall = 0.99;

		class KGraph {
		public:
			/// Indexing parameters.
			struct IndexParams {
				unsigned K;
				unsigned poolSize;
				unsigned reversePoolSize;
				unsigned iterations;
				float delta;
				float recall;

				/// Construct with default values.
				IndexParams() : iterations(defaultIterations), poolSize(defaultPoolSize), K(defaultK), reversePoolSize(defaultReversePoolSize), delta(defaultDelta), recall(defaultRecall) {
				}
			};

			/// Information and statistics of the indexing algorithm.
			struct IndexInfo 
			{
				enum StopCondition 
				{
					ITERATION = 0,
					DELTA,
					RECALL
				} stopCondition;
				unsigned iterations;
				float recall;
				float delta;
			};

			const std::vector<const float*> *baseData_;
			unsigned baseNum_;
			unsigned dimension_;
			const float *normArray_;
			IndexParams params;
			IndexInfo info;
			Nhoods nhoods;
			std::vector<std::vector<Neighbor>> kgraph;

			KGraph() : baseData_(nullptr), baseNum_(0), dimension_(0) {}
			~KGraph() {}

			void init();

			void join();

			void update();

			int build();

			void linearSearch(unsigned queryID, unsigned K, std::vector<Neighbor> *pnns);

			void generateControl(unsigned K, unsigned numControls, std::vector<Control> *pcontrols);
		};
	}
}

#endif // _KGRAPH_HPP_