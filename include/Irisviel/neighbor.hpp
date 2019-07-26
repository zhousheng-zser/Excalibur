#ifndef _NEIGHBOR_HPP_
#define _NEIGHBOR_HPP_

#include <vector>
#include <mutex>
#include <glasssix/accelerator.hpp>
#include "boost/smart_ptr/detail/spinlock.hpp"

namespace glasssix 
{
	namespace irisviel 
	{
		typedef boost::detail::spinlock Lock;
		typedef std::lock_guard<Lock> LockGuard;

		struct Neighbor 
		{
			uint32_t id = 0;
			float distance;
			bool flag;

			Neighbor() = default;
			Neighbor(unsigned id, float distance, bool f = true) : id{ id }, distance{ distance }, flag(f) {}

			inline bool operator<(const Neighbor &other) const 
			{
				return distance < other.distance;
			}

			inline bool operator == (const Neighbor &other) 
			{
				return id == other.id;
			}
		};

		struct LockNeighbor 
		{
			Lock lock;
			std::vector<Neighbor> pool;
		};

		typedef std::vector<LockNeighbor > LockGraph;

		struct Control 
		{
			unsigned id;
			std::vector<Neighbor> pool;
		};

		static inline int insertIntoPool(Neighbor *addr, unsigned topK, Neighbor nn)
		{
			if (topK == 0)
			{
				addr[0] = nn;
				return 0;
			}

			int start = 0;
			int end = topK - 1;
			int mid;

			// binary search
			while (start <= end)
			{
				mid = (start + end) / 2;
				if (nn.distance < addr[mid].distance)
				{
					end = mid - 1;
				}
				else
				{
					start = mid + 1;
				}
			}

		    // check for equal ID
			int pos = start;
			while (pos-- > 0)
			{
				if (addr[pos].distance < nn.distance)
				{
					break;
				}
					
				if (addr[pos].id == nn.id)
				{
					return topK + 1;
				}
			}

			for (int j = topK; j > start; j--) 
			{
				addr[j] = addr[j - 1];
			}

			addr[start] = nn;
			return start;
		}

		struct Nhood 
		{ // neighborhood
			Lock lock;
			std::vector<Neighbor> pool;
			unsigned neighborsLength;     // # valid items in the pool,  L + 1 <= pool.size()
			unsigned M;     // we only join items in pool[0..M)
			float radius;   // distance of interesting range
			float radiusM;
			bool found;     // helped found new NN in this round
			std::vector<unsigned> nnOld;
			std::vector<unsigned> nnNew;
			std::vector<unsigned> rnnOld;
			std::vector<unsigned> rnnNew;

			// only non-readonly method which is supposed to be called in parallel
			unsigned parallelTryInsert(unsigned id, float dist) 
			{
				if (dist > radius) 
					return pool.size();
				LockGuard guard(lock);
				unsigned insertPos = insertIntoPool(&pool[0], neighborsLength, Neighbor(id, dist, true));
				if (insertPos <= neighborsLength) 
				{ // inserted
					if (neighborsLength + 1 < pool.size()) 
					{ // if insertPos == neighborsLength + 1, there's a duplicate
						++neighborsLength;
					}
					else 
					{
						radius = pool[neighborsLength - 1].distance;
					}
				}
				return insertPos;
			}

			// join should not be conflict with insert
			template <typename C>
			void join(C callback) const 
			{
				for (unsigned const i : nnNew) 
				{
					for (unsigned const j : nnNew) 
					{
						if (i < j) 
						{
							callback(i, j);
						}
					}
					for (unsigned j : nnOld) 
					{
						callback(i, j);
					}
				}
			}
		};

		typedef std::vector<std::shared_ptr<Nhood> > Nhoods;
	}
}

#endif // !_NEIGHBOR_HPP_