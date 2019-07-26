#ifndef _NGRAPH_HPP_
#define _NGRAPH_HPP_

#include "kGraph.hpp"
#include "neighbor.hpp"
#include <boost/dynamic_bitset.hpp>
#include <stack>
#include <fstream>

namespace glasssix 
{
	namespace irisviel 
	{
		class NGraph 
		{
		public:

			NGraph() : baseData_(nullptr), baseNum_(0), dimension_(0) {}
			~NGraph() {}

			void initGraph();

			void getNavigateNode(const float *approximateCenter, std::vector<Neighbor> *pnns);

			void getNeighbors(unsigned queryID, std::vector<Neighbor> &pool, std::vector<Neighbor> &fullset);

			void addToGraph(unsigned destinationID, Neighbor newcomer, LockGraph& cutGraph_);

			void edgePrune(unsigned queryID, std::vector<Neighbor>& fullset, LockGraph& cutGraph_);

			void build();

			void link(LockGraph& cutGraph_);

			void treeGrow();

			void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);

			void findRoot(boost::dynamic_bitset<> &flag, unsigned &root);


			const std::vector<const float*> *baseData_;
			unsigned baseNum_;
			unsigned dimension_;
			const float *normArray_;
			typedef std::vector<std::vector<unsigned > > CompactGraph;
			CompactGraph finalGraph_;
			unsigned width;
			unsigned navigateNode;
			unsigned neighborsMaxLength;//max number of neighbors
			unsigned range;//max number of edges followed by MRNG strategy
		};
	}
}

#endif // !_NGRAPH_HPP_