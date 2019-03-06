#include "index.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace boost;

namespace glasssix
{
	namespace Irisvian
	{
		Index::Index() :baseData_(nullptr), baseNum_(0), dimension_(0) {};

		Index::Index(const vector<const float*> *baseData, int dimension)
			:baseData_(baseData), baseNum_((*baseData).size()), dimension_(dimension)
		{			
			normArray_ = (float*)malloc(baseNum_ * sizeof(float));

			//use 10 data to judge if normalized
			float sum = 0;
			int calcNum = std::min(10, (int)baseNum_);
			for (int i = 0; i < calcNum; ++i)
			{
#ifdef COSINE_DISTANCE
				sum += DistanceCosine::norm((*baseData_).at(i), dimension_);
#else
				sum += DistanceFastL2::norm((*baseData_).at(i), dimension_);
#endif // COSINE_DISTANCE
			}

			if (abs(sum / calcNum - 1) <= 1e-5)
			{
				isNormalized = true;
#pragma omp parallel for
				for (int i = 0; i < baseNum_; ++i)
				{
					normArray_[i] = 1;
				}
			}
			else
			{
#pragma omp parallel for
				for (int i = 0; i < baseNum_; ++i)
				{
#ifdef COSINE_DISTANCE
					normArray_[i] = DistanceCosine::norm((*baseData_).at(i), dimension_);
#else
					normArray_[i] = DistanceFastL2::norm((*baseData_).at(i), dimension_);
#endif // COSINE_DISTANCE
				}
			}

			kgraph_.baseData_ = baseData_;
			kgraph_.baseNum_ = baseNum_;
			kgraph_.dimension_ = dimension_;
			kgraph_.normArray_ = normArray_;

			ngraph_.baseData_ = baseData_;
			ngraph_.baseNum_ = baseNum_;
			ngraph_.dimension_ = dimension_;
			ngraph_.normArray_ = normArray_;
		};

		Index::Index(int dimension)
			:dimension_(dimension)
		{
			kgraph_.dimension_ = dimension_;
			ngraph_.dimension_ = dimension_;
		}

		Index::~Index() {}

#if defined(PROFILER) && defined(_MSC_VER)
			void Index::buildGraph(unsigned &maxMemoryUsage)
#else
			void Index::buildGraph()
#endif 
		{
			//index kgraph
			{
#if defined(PROFILER) && defined(_MSC_VER)
				kgraph_.build(maxMemoryUsage);
#else
				kgraph_.build();
#endif 
			}

			//transfer kgraph to nGraph's finalGraph_
			{
				std::vector<std::vector<Neighbor>> &kgraphTemp = kgraph_.kgraph;
				ngraph_.finalGraph_.resize(kgraphTemp.size());
				for (unsigned i = 0; i < kgraphTemp.size(); ++i)
				{
					auto const &neighbors = kgraphTemp[i];
					uint32_t size = neighbors.size();
					ngraph_.finalGraph_[i].resize(size);

					for (unsigned j = 0; j < size; ++j)
					{
						ngraph_.finalGraph_[i][j] = neighbors[j].id;
					}
				}
			}

			//index nGraph
			{
				ngraph_.build();
			}

			width = ngraph_.width;
			navigateNode = ngraph_.navigateNode;

			finalGraph.resize(ngraph_.finalGraph_.size());
			for (size_t i = 0; i < ngraph_.finalGraph_.size(); i++)
			{
				finalGraph[i].resize(ngraph_.finalGraph_[i].size());
				for (size_t j = 0; j < ngraph_.finalGraph_[i].size(); j++)
				{
					finalGraph[i][j] = ngraph_.finalGraph_[i][j];
				}
			}
		}


#if defined(PROFILER) && defined(_MSC_VER)
			void Index::buildGraph(const std::vector<const float*> *baseData, unsigned &maxMemoryUsage)
#else
			void Index::buildGraph(const std::vector<const float*> *baseData)
#endif
			{
				baseData_ = baseData;
				baseNum_ = ((*baseData).size());
				normArray_ = (float*)malloc(baseNum_ * sizeof(float));

				//use 10 data to judge if normalized
				float sum = 0;
				int calcNum = std::min(10, (int)baseNum_);
				for (int i = 0; i < calcNum; ++i)
				{
#ifdef COSINE_DISTANCE
					sum += DistanceCosine::norm((*baseData_).at(i), dimension_);
#else
					sum += DistanceFastL2::norm((*baseData_).at(i), dimension_);
#endif // COSINE_DISTANCE
				}

				if (abs(sum / calcNum - 1) <= 1e-5)
				{
					isNormalized = true;
#pragma omp parallel for
					for (int i = 0; i < baseNum_; ++i)
					{
						normArray_[i] = 1;
					}
				}
				else
				{
#pragma omp parallel for
					for (int i = 0; i < baseNum_; ++i)
					{
#ifdef COSINE_DISTANCE
						normArray_[i] = DistanceCosine::norm((*baseData_).at(i), dimension_);
#else
						normArray_[i] = DistanceFastL2::norm((*baseData_).at(i), dimension_);
#endif // COSINE_DISTANCE
					}
				}

				kgraph_.baseData_ = baseData_;
				kgraph_.baseNum_ = baseNum_;
				kgraph_.normArray_ = normArray_;

				ngraph_.baseData_ = baseData_;
				ngraph_.baseNum_ = baseNum_;
				ngraph_.normArray_ = normArray_;
				double elapsedTime = 0;

				//index kgraph
				{
#if defined(PROFILER) && defined(_MSC_VER)
					kgraph_.build(maxMemoryUsage);
#else
					kgraph_.build();
#endif 
				}
				//transfer kgraph to nGraph's finalGraph_
				{
					std::vector<std::vector<Neighbor>> &kgraphTemp = kgraph_.kgraph;
					ngraph_.finalGraph_.resize(kgraphTemp.size());
					for (unsigned i = 0; i < kgraphTemp.size(); ++i)
					{
						auto const &neighbors = kgraphTemp[i];
						uint32_t size = neighbors.size();
						ngraph_.finalGraph_[i].resize(size);

						for (unsigned j = 0; j < size; ++j)
						{
							ngraph_.finalGraph_[i][j] = neighbors[j].id;
						}
					}
				}

				//index nGraph
				{
					ngraph_.build();
				}

				width = ngraph_.width;
				navigateNode = ngraph_.navigateNode;

				finalGraph.resize(ngraph_.finalGraph_.size());
				for (size_t i = 0; i < ngraph_.finalGraph_.size(); i++)
				{
					finalGraph[i].resize(ngraph_.finalGraph_[i].size());
					for (size_t j = 0; j < ngraph_.finalGraph_[i].size(); j++)
					{
						finalGraph[i][j] = ngraph_.finalGraph_[i][j];
					}
				}
			}


		void Index::saveGraph(const char *nGraphPath) {
			std::ofstream out(nGraphPath, std::ios::binary);
			std::vector<std::vector<unsigned>> &tempGraph = ngraph_.finalGraph_;
			assert(tempGraph.size() == baseNum_);

			out.write(reinterpret_cast<char const *>(&isNormalized), sizeof(bool));
			out.write(reinterpret_cast<char const *>(&(ngraph_.width)), sizeof(unsigned));
			out.write(reinterpret_cast<char const *>(&(ngraph_.navigateNode)), sizeof(unsigned));
			for (unsigned i = 0; i < baseNum_; i++) {
				unsigned GK = (unsigned)tempGraph[i].size();
				out.write(reinterpret_cast<char const *>(&GK), sizeof(unsigned));
				out.write(reinterpret_cast<char const *>(&tempGraph[i][0]), GK * sizeof(unsigned));
				if (i % 100 == 0)
				{
					out.flush();
				}
			}
			out.close();
		}

		void Index::saveGraph(const char *nGraphPath, const char *basedataPath) {
			//save ngraph
			std::ofstream outGraph(nGraphPath, std::ios::binary);
			std::vector<std::vector<unsigned>> &tempGraph = ngraph_.finalGraph_;
			assert(tempGraph.size() == baseNum_);

			outGraph.write(reinterpret_cast<char const *>(&isNormalized), sizeof(bool));
			outGraph.write(reinterpret_cast<char const *>(&(ngraph_.width)), sizeof(unsigned));
			outGraph.write(reinterpret_cast<char const *>(&(ngraph_.navigateNode)), sizeof(unsigned));
			for (unsigned n = 0; n < baseNum_; n++) {
				unsigned GK = (unsigned)tempGraph[n].size();
				outGraph.write(reinterpret_cast<char const *>(&GK), sizeof(unsigned));
				outGraph.write(reinterpret_cast<char const *>(&tempGraph[n][0]), GK * sizeof(unsigned));
				if (n % 100 == 0)
				{
					outGraph.flush();
				}
			}
			outGraph.close();

			//save basedata
			std::ofstream outBaseData(basedataPath, std::ios::binary);
			for (unsigned n = 0; n < baseNum_; n++) {
				outBaseData.write(reinterpret_cast<char const *>(&(*baseData_)[n][0]), dimension_ * sizeof(float));
				if (n % 100 == 0)
				{
					outBaseData.flush();
				}
			}
			outBaseData.close();
		}
	}
}