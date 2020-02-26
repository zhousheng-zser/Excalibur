#include "kGraph.hpp"
#include <bitset>
#include <boost/dynamic_bitset.hpp>
#include <iostream>
#include <fstream>
#include <random>
#include "distance.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(PROFILER) && defined(_MSC_VER)
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <Psapi.h>
#endif 

using namespace std;

namespace glasssix
{
	namespace irisviel
	{
		// both pool and knn should be sorted in ascending order
		static float evaluateRecall(std::vector<Neighbor> const &approximateResults, std::vector<Neighbor> const &accurateResults)
		{
			if (accurateResults.empty()) return 1.0;
			unsigned found = 0;
			unsigned nApproximate = 0;
			unsigned nAccurate = 0;
			while (nApproximate < approximateResults.size() && nAccurate < accurateResults.size())
			{
				if (abs(accurateResults[nAccurate].distance - approximateResults[nApproximate].distance) < 1e-5)
				{
					++found;
					++nAccurate;
					++nApproximate;
				}
				else if (accurateResults[nAccurate].distance < approximateResults[nApproximate].distance)
				{
					++nAccurate;
				}
				else {
					cerr << "Distance is unstable." << endl;
					cerr << "Exact";
					for (auto const &p : accurateResults)
					{
						cerr << ' ' << p.id << ':' << p.distance;
					}
					cerr << endl;
					cerr << "Approx";
					for (auto const &p : approximateResults)
					{
						cerr << ' ' << p.id << ':' << p.distance;
					}
					cerr << endl;
					throw nsg_calculate_error("distance is unstable");
				}
			}
			return float(found) / accurateResults.size();
		}

		static float evaluateDelta(std::vector<Neighbor> const &pool, unsigned K)
		{
			unsigned count = 0;
			unsigned num = K;
			if (pool.size() < num) num = pool.size();
			for (unsigned i = 0; i < num; ++i) {
				if (pool[i].flag) ++count;
			}
			return float(count) / K;
		}

		void KGraph::linearSearch(unsigned queryID, unsigned K, std::vector<Neighbor> *pnns) {
			std::vector<Neighbor> neighbors(K + 1);
			Neighbor nn;
			nn.id = 0;
			nn.flag = true; // we don't really use this
			float normQuery = normArray_[queryID];
			unsigned tempK = 0;
			while (nn.id < baseNum_) {
				if (nn.id != queryID) {
					float normNN = normArray_[nn.id];
#ifdef COSINE_DISTANCE
					nn.distance = DistanceCosine::compare((*baseData_).at(nn.id), normNN, (*baseData_).at(queryID), normQuery, dimension_);
#else
					nn.distance = DistanceFastL2::compare((*baseData_).at(nn.id), normNN, (*baseData_).at(queryID), normQuery, dimension_);
#endif // COSINE_DISTANCE

					insertIntoPool(&neighbors[0], tempK, nn);
					if (tempK < K) ++tempK;
				}
				++nn.id;
			}
			neighbors.resize(K);
			pnns->swap(neighbors);
		}

		void KGraph::generateControl(unsigned K, unsigned numControls, vector<Control> *pcontrols) {
			vector<Control> controls(numControls);
			{
				vector<unsigned> index(baseNum_);
				int i = 0;
				for (unsigned &v : index) {
					v = i++;
				}
				std::mt19937 random_device{ std::random_device{}() };
				shuffle(index.begin(), index.end(), random_device);

#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (int i = 0; i < numControls; ++i) {
					controls[i].id = index[i];
					linearSearch(index[i], K, &controls[i].pool);
				}
			}
			pcontrols->swap(controls);
		}

		// generate size distinct random numbers (< numItem) to fill in addr
		template <typename RNG>
		static void genRandom(RNG &rng, unsigned *addr, unsigned size, unsigned numItem)
		{
			if (numItem == size)
			{
				for (unsigned i = 0; i < size; ++i)
				{
					addr[i] = i;
				}
				return;
			}
			for (unsigned i = 0; i < size; ++i)
			{
				addr[i] = rng() % (numItem - size);
			}
			std::sort(addr, addr + size);
			for (unsigned i = 1; i < size; ++i)
			{
				if (addr[i] <= addr[i - 1]) {
					addr[i] = addr[i - 1] + 1;
				}
			}
			unsigned off = rng() % numItem;
			for (unsigned i = 0; i < size; ++i)
			{
				addr[i] = (addr[i] + off) % numItem;
			}
		}

		// The neighborhood structure maintains a pool of near neighbors of an object.
		// The neighbors are stored in the pool.  "n" (<="params.poolSize") is the number of valid entries
		// in the pool, with the beginning "k" (<="n") entries sorted.

		void KGraph::init()
		{

#ifdef COSINE_DISTANCE

			if (dimension_ <= 128)
			{
				if (baseNum_ <= 50000)
				{
					params.K = 50;
					params.poolSize = 100;
					params.reversePoolSize = 100;
				}
				else if (baseNum_ <= 100000)
				{
					params.K = 60;
					params.poolSize = 110;
					params.reversePoolSize = 110;
				}
				else if (baseNum_ <= 200000)
				{
					params.K = 80;
					params.poolSize = 130;
					params.reversePoolSize = 130;
				}
				else if (baseNum_ <= 500000)
				{
					params.K = 80;
					params.poolSize = 130;
					params.reversePoolSize = 130;
				}
				else
				{
					params.K = 80;
					params.poolSize = 130;
					params.reversePoolSize = 130;
				}
			}
			else {
				if (baseNum_ <= 50000)
				{
					params.K = 70;
					params.poolSize = 120;
					params.reversePoolSize = 120;
				}
				else if (baseNum_ <= 100000)
				{
					params.K = 80;
					params.poolSize = 130;
					params.reversePoolSize = 130;
				}
				else if (baseNum_ <= 200000)
				{
					params.K = 70;
					params.poolSize = 120;
					params.reversePoolSize = 120;
				}
				else if (baseNum_ <= 500000)
				{
					params.K = 100;
					params.poolSize = 150;
					params.reversePoolSize = 150;
				}
				else
				{
					params.K = 100;
					params.poolSize = 150;
					params.reversePoolSize = 150;
				}
			}

#else

			if (dimension_ <= 128)
			{
				if (baseNum_ <= 50000)
				{
					params.K = 50;
					params.poolSize = 100;
					params.reversePoolSize = 100;
				}
				else if (baseNum_ <= 100000)
				{
					params.K = 60;
					params.poolSize = 110;
					params.reversePoolSize = 110;
				}
				else if (baseNum_ <= 200000)
				{
					params.K = 80;
					params.poolSize = 130;
					params.reversePoolSize = 130;
				}
				else if (baseNum_ <= 500000)
				{
					params.K = 80;
					params.poolSize = 130;
					params.reversePoolSize = 130;
				}
				else
				{
					params.K = 80;
					params.poolSize = 130;
					params.reversePoolSize = 130;
				}
			}
			else {
				if (baseNum_ <= 50000)
				{
					params.K = 70;
					params.poolSize = 120;
					params.reversePoolSize = 120;
				}
				else if (baseNum_ <= 100000)
				{
					params.K = 80;
					params.poolSize = 130;
					params.reversePoolSize = 130;
				}
				else if (baseNum_ <= 200000)
				{
					params.K = 70;
					params.poolSize = 120;
					params.reversePoolSize = 120;
				}
				else if (baseNum_ <= 500000)
				{
					params.K = 100;
					params.poolSize = 150;
					params.reversePoolSize = 150;
				}
				else
				{
					params.K = 100;
					params.poolSize = 150;
					params.reversePoolSize = 150;
				}
			}

#endif // COSINE_DISTANCE

			if (baseNum_ <= params.K)
			{
				LOG(WARNING) << "Warning: small dataset, shrinking params.K to " << baseNum_ - 1 << ".";
				params.K = baseNum_ - 1;
			}
			if (baseNum_ <= params.poolSize)
			{
				LOG(WARNING) << "Warning: small dataset, shrinking poolSize to " << baseNum_ - 1 << ".";
				params.poolSize = baseNum_ - 1;
			}
			if (baseNum_ <= params.reversePoolSize)
			{
				LOG(WARNING) << "Warning: small dataset, shrinking reversePoolSize to " << baseNum_ - 1 << ".";
				params.reversePoolSize = baseNum_ - 1;
			}

			unsigned N = baseNum_;
			nhoods.resize(baseNum_);
			unsigned seed = 1998;
			mt19937 rng(seed);

#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int n = 0; n < N; n++)
			{
				auto &nhood = nhoods[n];
				nhood.reset(new Nhood());
				(nhood->nnNew).resize(params.K * 2);
				(nhood->pool).resize(params.poolSize + 1);
				nhood->radius = numeric_limits<float>::max();
			}

#ifdef _OPENMP
#pragma omp parallel
#endif
			{
#ifdef _OPENMP
				mt19937 rng(seed ^ omp_get_thread_num());
#else
				mt19937 rng(seed);
#endif

				vector<unsigned> random(params.K + 1);
#ifdef _OPENMP
#pragma omp for
#endif
				for (int n = 0; n < N; ++n)
				{
					auto &nhood = nhoods[n];
					std::vector<Neighbor> &pool = nhood->pool;
					genRandom(rng, &nhood->nnNew[0], nhood->nnNew.size(), N);
					genRandom(rng, &random[0], random.size(), N);
					nhood->neighborsLength = params.K;
					nhood->M = params.K;
					unsigned i = 0;
					for (unsigned j = 0; j < nhood->neighborsLength; ++j)
					{
						if (random[i] == n) ++i;
						auto &nn = nhood->pool[j];
						nn.id = random[i++];
						float normN = normArray_[n];
						float normNN = normArray_[nn.id];
#ifdef COSINE_DISTANCE
						nn.distance = DistanceCosine::compare((*baseData_).at(nn.id), normNN, (*baseData_).at(n), normN, dimension_);
#else
						nn.distance = DistanceFastL2::compare((*baseData_).at(nn.id), normNN, (*baseData_).at(n), normN, dimension_);
#endif // COSINE_DISTANCE

						nn.flag = true;
					}
					sort(pool.begin(), pool.begin() + nhood->neighborsLength);
				}
			}
		}

		void KGraph::join()
		{
#ifdef _OPENMP
#pragma omp parallel for default(shared) schedule(dynamic, 100)
#endif
			for (int n = 0; n < baseNum_; ++n)
			{
				size_t signal = 0;
				nhoods[n]->found = false;
				nhoods[n]->join([&](unsigned i, unsigned j)
				{
					float norm_i = normArray_[i];
					float norm_j = normArray_[j];
#ifdef COSINE_DISTANCE
					float dist = DistanceCosine::compare((*baseData_).at(i), norm_i, (*baseData_).at(j), norm_j, dimension_);
#else
					float dist = DistanceFastL2::compare((*baseData_).at(i), norm_i, (*baseData_).at(j), norm_j, dimension_);
#endif // COSINE_DISTANCE	
					unsigned insertPos;
					insertPos = nhoods[i]->parallelTryInsert(j, dist);
					if (insertPos < params.K) ++signal;
					insertPos = nhoods[j]->parallelTryInsert(i, dist);
					if (insertPos < params.K) ++signal;
				});
				nhoods[n]->found = signal > 0;
			}
		}

		void KGraph::update()
		{
			std::random_device rd;
			std::mt19937 rng(rd());
			unsigned N = baseNum_;

#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int n = 0; n < N; ++n)
			{
				auto &nhood = nhoods[n];
				nhood->nnNew.clear();
				nhood->nnOld.clear();
				nhood->rnnNew.clear();
				nhood->rnnOld.clear();
				nhood->radius = nhoods[n]->pool.back().distance;
				
				if (nhood->found)
				{
					unsigned maxPos = std::min(nhood->M + params.K, nhood->neighborsLength);
					unsigned count = 0;
					unsigned pos = 0;
					while ((pos < maxPos) && (count < params.K))
					{
						if (nhood->pool[pos].flag) ++count;
						++pos;
					}
					//nhood->M is position represented by nhood.old+params.K
					nhood->M = pos;
				}
				BOOST_VERIFY(nhood->M > 0);
				nhood->radiusM = nhood->pool[nhood->M - 1].distance;
			}
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int n = 0; n < N; ++n)
			{
				auto &nhood = nhoods[n];
				auto &nnNew = nhood->nnNew;
				auto &nnOld = nhood->nnOld;
				for (unsigned i = 0; i < nhood->M; ++i)
				{
					auto &nn = nhood->pool[i];
					auto &nhoodO = nhoods[nn.id];  // nhood on the other side of the edge
					if (nn.flag)
					{
						nnNew.push_back(nn.id);
						//nn belongs to nhoods[n]->pool, but nn is not a knn point of n(we can judge from a large dist),
						//so nn is a reverse-KNN point of n, that is to say n is a knn point of nn, so we add n to nhoodO.rnn
						if (nn.distance > nhoodO->radiusM)
						{
							LockGuard guard(nhoodO->lock);
							nhoodO->rnnNew.push_back(n);
						}
						nn.flag = false;
					}
					else
					{
						nnOld.push_back(nn.id);
						if (nn.distance > nhoodO->radiusM)
						{
							LockGuard guard(nhoodO->lock);
							nhoodO->rnnOld.push_back(n);
						}
					}
				}
			}

#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int n = 0; n < N; ++n)
			{
				auto &nhood = nhoods[n];
				auto &nnNew = nhood->nnNew;
				auto &nnOld = nhood->nnOld;
				auto &rnnNew = nhood->rnnNew;
				auto &rnnOld = nhood->rnnOld;
				if (rnnNew.size() > params.reversePoolSize)
				{
					std::shuffle(rnnNew.begin(), rnnNew.end(), rng);
					rnnNew.resize(params.reversePoolSize);
				}
				nnNew.insert(nnNew.end(), rnnNew.begin(), rnnNew.end());
				if (rnnOld.size() > params.reversePoolSize)
				{
					std::shuffle(rnnOld.begin(), rnnOld.end(), rng);
					rnnOld.resize(params.reversePoolSize);
				}
				nnOld.insert(nnOld.end(), rnnOld.begin(), rnnOld.end());
			}
		}

		int KGraph::build()
		{
			init();
			int maxMemoryUsage = 0;
			unsigned N = baseNum_;
			vector<Control> controls;
			
			int controlNum = 100 < N ? 100 : N;
			generateControl(params.K, controlNum, &controls);
			info.stopCondition = IndexInfo::ITERATION;
			info.recall = 0;
			info.iterations = 0;
			info.delta = 1.0;

#ifdef PROFILER
#ifdef _MSC_VER
			HANDLE handle = GetCurrentProcess();
			PROCESS_MEMORY_COUNTERS_EX pmc = { 0 };
			if (!GetProcessMemoryInfo(handle, (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc)))
			{
				DWORD errCode = GetLastError();
				return maxMemoryUsage;
			}
#endif // _MSC_VER
#endif // PROFILER

			for (unsigned it = 0; it < params.iterations; ++it)
			{		
				++info.iterations;

				join();
				{
					float recall2 = 0, delta2 = 0;
					for (auto const &nhood : nhoods)
					{
						delta2 += evaluateDelta(nhood->pool, params.K);
					}
					for (auto const &c : controls)
					{
						recall2 += evaluateRecall(nhoods[c.id]->pool, c.pool);
					}
					info.delta = delta2 / nhoods.size();
					info.recall = recall2 / controls.size();


					if (info.delta <= params.delta)
					{
						info.stopCondition = IndexInfo::DELTA;
						//LOG(INFO) << "recall: " << info.recall << " delta: " << info.delta;
						break;
					}
					if (info.recall >= params.recall)
					{
						info.stopCondition = IndexInfo::RECALL;
						//LOG(INFO) << "recall: " << info.recall << " delta: " << info.delta;
						break;
					}
					update();
				}
			}

			kgraph.resize(N);

			for (unsigned n = 0; n < N; ++n)
			{
				auto &knn = kgraph[n];
				auto const &pool = nhoods[n]->pool;
				unsigned size = params.poolSize;
				knn.resize(size);
				for (unsigned k = 0; k < size; ++k)
				{
					knn[k].id = pool[k].id;
					knn[k].distance = pool[k].distance;
				}
			}
#if defined(PROFILER) && defined(_MSC_VER)
			maxMemoryUsage = pmc.WorkingSetSize / 1048576;
#endif 
			nhoods.clear();
			return maxMemoryUsage;
		}
	}
}