#include "NGraph.hpp"
#include "distance.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace boost;

namespace glasssix
{
	namespace Irisvian
	{
		void NGraph::getNavigateNode(const float *approximateCenter, vector<Neighbor> *pnns)
		{
			int testNum = std::min(10, (int)baseNum_);
			Neighbors neighbors(testNum + 1);
			unsigned tempK = 0;
			for (int i = 0; i < baseNum_; ++i)
			{
				float dist = DistanceL2::compare((*baseData_).at(i), approximateCenter, (unsigned)dimension_);
				Neighbor nn = Neighbor(i, dist, true);
				insertIntoPool(&neighbors[0], tempK, nn);
				if (tempK < testNum) ++tempK;
			}
			neighbors.resize(testNum);
			pnns->swap(neighbors);
		}

		void NGraph::getNeighbors(
			unsigned queryID,
			std::vector <Neighbor> &pool, std::vector <Neighbor> &fullset)
		{
			pool.resize(neighborsMaxLength + 1);
			std::vector<unsigned> initIds(neighborsMaxLength);
			boost::dynamic_bitset<> flags{ baseNum_, 0 };
			unsigned count = 0;

			for (unsigned i = 0; i < initIds.size() && i < finalGraph_[navigateNode].size(); i++)
			{
				initIds[i] = finalGraph_[navigateNode][i];
				flags[initIds[i]] = true;
				count++;
			}
			while (count < initIds.size())
			{
				unsigned id = rand() % baseNum_;
				if (flags[id])continue;
				initIds[count] = id;
				count++;
				flags[id] = true;
			}

			float normQuery = normArray_[queryID];
			for (unsigned i = 0; i < initIds.size(); i++)
			{
				unsigned id = initIds[i];
				float normID = normArray_[id];

#ifdef COSINE_DISTANCE
				float dist = DistanceCosine::compare((*baseData_).at(id), normID, (*baseData_).at(queryID), normQuery, dimension_);
#else
				float dist = DistanceFastL2::compare((*baseData_).at(id), normID, (*baseData_).at(queryID), normQuery, dimension_);
#endif // COSINE_DISTANCE

				pool[i] = Neighbor(id, dist, true);
			}

			std::sort(pool.begin(), pool.begin() + neighborsMaxLength);
			int i = 0;
			while (i < (int)neighborsMaxLength)
			{
				int minPos = neighborsMaxLength;

				if (pool[i].flag)
				{
					pool[i].flag = false;
					unsigned n = pool[i].id;

					for (unsigned m = 0; m < finalGraph_[n].size(); ++m)
					{
						unsigned id = finalGraph_[n][m];
						if (flags[id])continue;
						flags[id] = 1;

						float normID = normArray_[id];

#ifdef COSINE_DISTANCE
						float dist = DistanceCosine::compare((*baseData_).at(id), normID, (*baseData_).at(queryID), normQuery, dimension_);
#else
						float dist = DistanceFastL2::compare((*baseData_).at(id), normID, (*baseData_).at(queryID), normQuery, dimension_);
#endif // COSINE_DISTANCE

						Neighbor nn(id, dist, true);
						fullset.push_back(nn);

						if (dist >= pool[neighborsMaxLength - 1].distance)continue;
						int insertPos = insertIntoPool(&pool[0], neighborsMaxLength, nn);
						if (pool.size() > neighborsMaxLength + 1) {
							pool.pop_back();
						}

						if (insertPos < minPos)
						{
							minPos = insertPos;
						}
					}
				}
				if (minPos <= i)
				{
					i = minPos;
				}
				else
				{
					++i;
				}
			}
		}

		void NGraph::initGraph()
		{

#ifdef COSINE_DISTANCE

			if (dimension_ <= 128)
			{
				if (baseNum_ <= 50000) {
					neighborsMaxLength = 50;
					range = 30;
				}
				else if (baseNum_ <= 100000) {
					neighborsMaxLength = 50;
					range = 30;
				}
				else if (baseNum_ <= 200000) {
					neighborsMaxLength = 50;
					range = 40;
				}
				else if (baseNum_ <= 500000) {
					neighborsMaxLength = 60;
					range = 60;
				}
				else {
					neighborsMaxLength = 60;
					range = 60;
				}
			}
			else
			{
				if (baseNum_ <= 50000) {
					neighborsMaxLength = 20;
					range = 20;
				}
				else if (baseNum_ <= 100000) {
					neighborsMaxLength = 40;
					range = 40;
				}
				else if (baseNum_ <= 200000) {
					neighborsMaxLength = 70;
					range = 70;
				}
				else if (baseNum_ <= 500000) {
					neighborsMaxLength = 40;
					range = 40;
				}
				else {
					neighborsMaxLength = 40;
					range = 40;
				}
			}

#else

			if (dimension_ <= 128)
			{
				if (baseNum_ <= 50000) {
					neighborsMaxLength = 50;
					range = 30;
				}
				else if (baseNum_ <= 100000) {
					neighborsMaxLength = 50;
					range = 40;
				}
				else if (baseNum_ <= 200000) {
					neighborsMaxLength = 80;
					range = 80;
				}
				else if (baseNum_ <= 500000) {
					neighborsMaxLength = 60;
					range = 60;
				}
				else {
					neighborsMaxLength = 60;
					range = 60;
				}
			}
			else
			{
				if (baseNum_ <= 50000) {
					neighborsMaxLength = 20;
					range = 20;
				}
				else if (baseNum_ <= 100000) {
					neighborsMaxLength = 40;
					range = 40;
				}
				else if (baseNum_ <= 200000) {
					neighborsMaxLength = 70;
					range = 70;
				}
				else if (baseNum_ <= 500000) {
					neighborsMaxLength = 40;
					range = 40;
				}
				else {
					neighborsMaxLength = 40;
					range = 40;
				}
			}

#endif // COSINE_DISTANCE

			if (baseNum_ <= neighborsMaxLength)
			{
				cerr << "Warning: small dataset, shrinking neighborsMaxLength to " << baseNum_ << "." << endl;
				neighborsMaxLength = baseNum_;
			}

			if (baseNum_ <= range)
			{
				cerr << "Warning: small dataset, shrinking range to " << baseNum_ << "." << endl;
				range = baseNum_;
			}

			float* center = (float*)malloc(dimension_ * sizeof(float));
			for (unsigned j = 0; j < dimension_; j++)
			{
				center[j] = 0;
			}
			for (unsigned i = 0; i < baseNum_; i++)
			{
				for (unsigned j = 0; j < dimension_; j++)
				{
					center[j] += (*baseData_).at(i)[j];
				}
			}
			for (unsigned j = 0; j < dimension_; j++) {
				center[j] /= baseNum_;
			}
			std::vector <Neighbor> pool;
			getNavigateNode(center, &pool);
			navigateNode = pool[0].id;
		}

		void NGraph::addToGraph(unsigned destinationID, Neighbor newcomer, LockGraph &cutGraph_) 
		{
			LockGuard guard(cutGraph_[destinationID].lock);
			for (unsigned i = 0; i < cutGraph_[destinationID].pool.size(); i++) 
			{
				if (newcomer.id == cutGraph_[destinationID].pool[i].id)
					return;
			}
			cutGraph_[destinationID].pool.push_back(newcomer);
			if (cutGraph_[destinationID].pool.size() > range) 
			{
				std::vector <Neighbor> result;
				std::vector <Neighbor> &pool = cutGraph_[destinationID].pool;
				unsigned start = 0;
				std::sort(pool.begin(), pool.end());
				result.push_back(pool[start]);

				while (result.size() < range && (++start) < pool.size()) 
				{
					auto &candidate = pool[start];
					bool signal = false;
					for (unsigned i = 0; i < result.size(); i++) 
					{
						if (candidate.id == result[i].id) 
						{
							signal = true;
							break;
						}

						//three points(candidate,queryID and result[i].id) construct a triangle,
						//MRNG strategy will be satisfied(for all result[i]) when edge cr(edge between candidata and result[i].id) is the longest
						float normResultID = normArray_[result[i].id];
						float normCandidateID = normArray_[candidate.id];

#ifdef COSINE_DISTANCE
						float distance = DistanceCosine::compare((*baseData_).at(result[i].id), normResultID,
							(*baseData_).at(candidate.id), normCandidateID, dimension_);
#else
						float distance = DistanceFastL2::compare((*baseData_).at(result[i].id), normResultID,
							(*baseData_).at(candidate.id), normCandidateID, dimension_);
#endif // COSINE_DISTANCE

						if (distance < candidate.distance) 
						{
							signal = true;
							break;
						}

					}
					if (!signal)
						result.push_back(candidate);
				}
				pool.swap(result);
			}

		}

		void NGraph::edgePrune(unsigned queryID, std::vector <Neighbor> &fullset, LockGraph &cutGraph_)
		{
			width = range;
			unsigned start = 0;

			boost::dynamic_bitset<> flags{ baseNum_, 0 };
			for (unsigned i = 0; i < fullset.size(); i++)
			{
				flags[fullset[i].id] = 1;
			}

			float normQuery = normArray_[queryID];
			for (unsigned nn = 0; nn < finalGraph_[queryID].size(); nn++)
			{
				unsigned id = finalGraph_[queryID][nn];
				if (flags[id])continue;

				float normID = normArray_[id];

#ifdef COSINE_DISTANCE
				float dist = DistanceCosine::compare((*baseData_).at(queryID), normQuery,
					(*baseData_).at(id), normID, dimension_);
#else

				float dist = DistanceFastL2::compare((*baseData_).at(queryID), normQuery,
					(*baseData_).at(id), normID, dimension_);
#endif // COSINE_DISTANCE

				fullset.push_back(Neighbor(id, dist, true));
			}

			std::sort(fullset.begin(), fullset.end());
			std::vector <Neighbor> result;
			if (fullset[start].id == queryID)start++;
			result.push_back(fullset[start]);

			while (result.size() < range && (++start) < fullset.size())
			{
				auto &candidate = fullset[start];
				bool signal = false;
				for (unsigned i = 0; i < result.size(); i++)
				{
					if (candidate.id == result[i].id)
					{
						signal = true;
						break;
					}
					//three points(candidate,queryID and result[i].id) construct a triangle,
					//MRNG strategy will be satisfied(for all result[i]) when edge cr(edge between candidata and result[i].id) is the longest
					float normResultID = normArray_[result[i].id];
					float normCandidateID = normArray_[candidate.id];

#ifdef COSINE_DISTANCE
					float distance = DistanceCosine::compare((*baseData_).at(result[i].id), normResultID,
						(*baseData_).at(candidate.id), normCandidateID, dimension_);
#else
					float distance = DistanceFastL2::compare((*baseData_).at(result[i].id), normResultID,
						(*baseData_).at(candidate.id), normCandidateID, dimension_);
#endif // COSINE_DISTANCE

					if (distance < candidate.distance)
					{
						signal = true;
						break;
					}
				}
				if (!signal)result.push_back(candidate);
			}

			for (unsigned i = 0; i < result.size(); i++)
			{
				addToGraph(queryID, result[i], cutGraph_);
				addToGraph(result[i].id, Neighbor(queryID, result[i].distance, true), cutGraph_);
			}
		}

		void NGraph::link(LockGraph &cutGraph_)
		{
			//std::cout << "indexing nsg..." << std::endl;
#pragma omp parallel
			{
#pragma omp for
				for (int n = 0; n < baseNum_; ++n) {
					std::vector <Neighbor> pool, fullset;
					getNeighbors(n, pool, fullset);
					edgePrune(n, fullset, cutGraph_);
				}
			}
		}

		void NGraph::build()
		{
			initGraph();//find out navigateNode
			LockGraph cutGraph_(baseNum_);
			link(cutGraph_);//for each point, find nearest neighbors to navigateNode, then execute prune strategy
			finalGraph_.resize(baseNum_);

			unsigned max = 0, min = 1e6, avg = 0, cnt = 0;

			for (unsigned i = 0; i < baseNum_; i++)
			{
				auto &pool = cutGraph_[i].pool;
				finalGraph_[i].resize(pool.size());
				for (unsigned j = 0; j < pool.size(); j++)
				{
					finalGraph_[i][j] = pool[j].id;
				}

				max = max < pool.size() ? pool.size() : max;
				min = min > pool.size() ? pool.size() : min;
				avg += pool.size();
				if (pool.size() < 2) cnt++;
			}

			treeGrow();

			avg /= baseNum_;
			max = 0;
			for (unsigned i = 0; i < baseNum_; i++)
			{
				max = max < finalGraph_[i].size() ? finalGraph_[i].size() : max;
			}
			if (max > width)
				width = max;
			/*std::cout << "max is:" << max
				<< ", avg is:" << avg
				<< ", min is:" << min
				<< ", unconnected num is:" << cnt
				<< ", width is:" << width << std::endl;*/
		}

		void NGraph::DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt)
		{
			unsigned tmp = root;
			std::stack<unsigned> s;
			s.push(root);
			if (!flag[root])cnt++;
			flag[root] = true;
			while (!s.empty())
			{
				unsigned next = baseNum_ + 1;
				for (unsigned i = 0; i<finalGraph_[tmp].size(); i++)
				{
					if (flag[finalGraph_[tmp][i]] == false)
					{
						next = finalGraph_[tmp][i];
						break;
					}
				}
				if (next == (baseNum_ + 1))
				{
					s.pop();
					if (s.empty())
						break;
					tmp = s.top();
					continue;
				}
				tmp = next;
				flag[tmp] = true; s.push(tmp); cnt++;
			}
		}

		void NGraph::findRoot(boost::dynamic_bitset<> &flag, unsigned &root) 
		{
			unsigned id;
			for (unsigned i = 0; i<baseNum_; i++)
			{
				if (flag[i] == false)
				{
					id = i;
					break;
				}
			}
			std::vector <Neighbor> pool, fullset;
			getNeighbors(id, pool, fullset);
			std::sort(pool.begin(), pool.end());

			unsigned found = 0;
			for (unsigned i = 0; i<pool.size(); i++)
			{
				if (flag[pool[i].id])
				{
					root = pool[i].id;
					found = 1;
					break;
				}
			}
			if (found == 0)
			{
				while (true)
				{
					unsigned rid = rand() % baseNum_;
					if (flag[rid])
					{
						root = rid;
						break;
					}
				}
			}
			finalGraph_[root].push_back(id);
		}

		void NGraph::treeGrow()
		{
			unsigned root = navigateNode;
			boost::dynamic_bitset<> flags{ baseNum_, 0 };
			unsigned unlinkedCnt = 0;
			while (unlinkedCnt < baseNum_)
			{
				DFS(flags, root, unlinkedCnt);
				if (unlinkedCnt >= baseNum_)break;
				findRoot(flags, root);
			}
		}
	}
}