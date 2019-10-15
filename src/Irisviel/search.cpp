#include "search.hpp"
#include <glasssix/accelerator.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace boost;
using namespace glasssix::excalibur;

namespace glasssix
{
	namespace irisviel
	{
		Search::Search(const std::vector<const float*> *baseData, int dimension)
			: dimension_(dimension)
		{
			baseData_ = baseData;
			baseNum_ = (*baseData).size();
		}

		Search::Search(int dimension)
			: dimension_(dimension){}

		Search::~Search(){}

		bool Search::loadGraph(const char *graphPath)
		{
			ngraph.clear();
			std::ifstream in(graphPath, std::ios::binary);
			if (!in.is_open())
			{
				std::cout << "open file error" << std::endl; exit(-1);
			}

			index_header header;
			in.read((char *)&header, sizeof(index_header));

			isNormalized = header.normalized;
			width = header.graph_width;
			navigateNode = header.navigate_node;
			uint64_t file_size = header.file_size;
			
			in.seekg(0, std::ios::end);
			auto pos = in.tellg();
			if (pos != file_size)
			{
				return false;
			}

			in.seekg(sizeof(index_header), std::ios::beg);
			while (!in.eof()) {
				unsigned k;
				in.read((char *)&k, sizeof(unsigned));
				if (in.eof())break;
				std::vector<unsigned> tmp(k);
				in.read((char *)tmp.data(), k * sizeof(unsigned));
				ngraph.push_back(tmp);
			}

			return true;
		}

		bool Search::loadGraph(const char *graphPath, const char *basedataPath)
		{
			//load graph
			ngraph.clear();
			std::ifstream in(graphPath, std::ios::binary);
			if (!in.is_open())
			{
				std::cout << "open file error" << std::endl; exit(-1);
			}

			index_header header;
			in.read((char *)&header, sizeof(index_header));

			isNormalized = header.normalized;
			width = header.graph_width;
			navigateNode = header.navigate_node;
			uint64_t file_size = header.file_size;

			in.seekg(0, std::ios::end);
			auto pos = in.tellg();
			if (pos != file_size)
			{
				return false;
			}

			//in.read((char *)&isNormalized, sizeof(bool));
			//in.read((char *)&width, sizeof(unsigned));
			//in.read((char *)&navigateNode, sizeof(unsigned));
			in.seekg(sizeof(index_header), std::ios::beg);
			while (!in.eof())
			{
				unsigned k;
				in.read((char *)&k, sizeof(unsigned));
				if (in.eof())break;
				std::vector<unsigned> tmp(k);
				in.read((char *)tmp.data(), k * sizeof(unsigned));
				ngraph.push_back(tmp);
			}

			//load basedata
			baseDataPtr.clear();
			std::ifstream inBaseData(basedataPath, std::ios::binary);
			if (!inBaseData.is_open())
			{
				std::cout << "open file error" << std::endl; exit(-1);
			}

			while (!inBaseData.eof()) 
			{
				float *temp_data = (float*)malloc(dimension_ * sizeof(float));
				inBaseData.read((char*)(temp_data), dimension_ * sizeof(float));
				baseDataPtr.push_back(const_cast<const float*>(temp_data));
			}
			
			baseData_ = &baseDataPtr;
			baseNum_ = baseDataPtr.size() - 1;

			return true;
		}

		const std::vector<const float*>* Search::getBasedata()
		{
			return baseData_;
		}

		void Search::optimizeGraph() 
		{
			//use after build or load
			if (baseNum_ <= 50000) {
				neighborsMaxLength = 200;
			}
			else if (baseNum_ <= 100000) {
				neighborsMaxLength = 200;
			}
			else if (baseNum_ <= 200000) {
				neighborsMaxLength = 240;
			}
			else if (baseNum_ <= 500000) {
				neighborsMaxLength = 280;
			}
			else {
				neighborsMaxLength = 300;
			}

			if (baseNum_ <= neighborsMaxLength)
			{
				cerr << "Warning: small dataset, shrinking neighborsMaxLength to " << baseNum_ << "." << endl;
				neighborsMaxLength = baseNum_;
			}

			dataLen = (dimension_ + 1) * sizeof(float);
			neighborLen = (width + 1) * sizeof(unsigned);
			nodeSize = dataLen + neighborLen;
			optGraph_tensor_.reset(new tensor<char>(nodeSize * baseNum_));
			optGraph_ = optGraph_tensor_->mutable_cpu_data();
			for (unsigned i = 0; i<baseNum_; i++) {
				char* curNodeOffset = optGraph_ + i * nodeSize;

				float cur_norm = 0;
				if (isNormalized)
				{
					cur_norm = 1;
				}
				else
				{
#ifdef COSINE_DISTANCE
					cur_norm = DistanceCosine::norm((*baseData_).at(i), dimension_);
#else
					cur_norm = DistanceFastL2::norm((*baseData_).at(i), dimension_);
#endif // COSINE_DISTANCE
				}

				memcpy(curNodeOffset, &cur_norm, sizeof(float));
				memcpy(curNodeOffset + sizeof(float), (*baseData_).at(i), dataLen - sizeof(float));

				curNodeOffset += dataLen;
				unsigned k = ngraph[i].size();
				memcpy(curNodeOffset, &k, sizeof(unsigned));
				memcpy(curNodeOffset + sizeof(unsigned), ngraph[i].data(), k * sizeof(unsigned));
				std::vector<unsigned>().swap(ngraph[i]);
			}
			CompactGraph().swap(ngraph);
		}

		void Search::searchVector(const vector<const float*>* queryData, unsigned topK, std::vector<std::vector<unsigned>> &returnIDs,
			std::vector<std::vector<float>> &returnSimilarities)
		{
			queryData_ = queryData;
			queryNum_ = (*queryData).size();

			returnIDs.resize(queryNum_);
			returnSimilarities.resize(queryNum_);

			if (baseNum_ != 1)
			{
#ifdef _OPENMP
#pragma omp parallel for
#endif
				for (int i = 0; i < queryNum_; ++i)
				{
					searchWithOptGraph((*queryData_).at(i), topK, returnIDs[i], returnSimilarities[i]);
				}
			}
			else
			{
				for (int i = 0; i < queryNum_; ++i)
				{
					returnIDs[i].resize(1);
					returnSimilarities[i].resize(1);

					//return 0 when there is only one picture in database
					returnIDs[i][0] = 0;

#ifdef COSINE_DISTANCE
					float normBase = DistanceCosine::norm((*baseData_).at(0), dimension_);
					float normQuery = DistanceCosine::norm((*queryData_).at(i), dimension_);
					float dist = DistanceCosine::compare((*baseData_).at(0), normBase, (*queryData_).at(i), normQuery, dimension_);
					returnSimilarities[i][0] = 1.0f - dist;
#else
					float normQuery = DistanceFastL2::norm((*queryData_).at(i), dimension_);
					float dist = DistanceL2::compare((*baseData_).at(0), (*queryData_).at(i), dimension_);
					returnSimilarities[i][0] = 1.0f - 1.0f * dist / normQuery;
#endif // COSINE_DISTANCE
				}
			}
		}

		void Search::searchWithOptGraph(const float *singleQueryData, unsigned topK,
			std::vector<unsigned> &returnIDs, std::vector<float> &returnSimilarities)
		{
			if (topK > neighborsMaxLength || topK > baseNum_)
			{
				cerr << "error, topK is bigger than " << neighborsMaxLength << ", or bigger than " << baseNum_;
				return;
			}

			std::vector <Neighbor> returnNeighbors;
			returnNeighbors.resize(neighborsMaxLength + 1);
			returnIDs.resize(topK);
			returnSimilarities.resize(topK);
			std::vector<unsigned> initIds(neighborsMaxLength);

			boost::dynamic_bitset<> flags{ baseNum_, 0 };
			unsigned count = 0;
			unsigned *neighbors = (unsigned*)(optGraph_ + nodeSize * navigateNode + dataLen);
			unsigned MaxM_ep = *neighbors;
			neighbors++;

			for (; count < neighborsMaxLength && count < MaxM_ep; count++)
			{
				initIds[count] = neighbors[count];
				flags[initIds[count]] = true;
			}

			while (count < neighborsMaxLength)
			{
				unsigned id = rand() % baseNum_;
				if (flags[id])continue;
				flags[id] = true;
				initIds[count] = id;
				count++;
			}

#if SIMD_TYPE >= SIMDTYPE_SSE
			for (unsigned i = 0; i < initIds.size(); i++)
			{
				unsigned id = initIds[i];
				if (id >= baseNum_)continue;
				_mm_prefetch(optGraph_ + nodeSize * id, _MM_HINT_T0);
			}
#endif

			float normQuery = 0.0f;
			if (isNormalized)
			{
				normQuery = 1.0f;
			}
			else
			{
#ifdef COSINE_DISTANCE
				normQuery = DistanceCosine::norm(singleQueryData, dimension_);
#else
				normQuery = DistanceFastL2::norm(singleQueryData, dimension_);
#endif // COSINE_DISTANCE
			}

			for (unsigned i = 0; i < initIds.size(); i++)
			{
				unsigned id = initIds[i];
				float *x = (float*)(optGraph_ + nodeSize * id);
				float norm_x = *x; x++;

#ifdef COSINE_DISTANCE
				float dist = DistanceCosine::compare(x, norm_x, singleQueryData, normQuery, dimension_);
#else
				float dist = DistanceFastL2::compare(x, norm_x, singleQueryData, normQuery, dimension_);
#endif // COSINE_DISTANCE

				returnNeighbors[i] = Neighbor(id, dist, true);
				flags[id] = true;

			}

			std::sort(returnNeighbors.begin(), returnNeighbors.begin() + neighborsMaxLength);
			int i = 0;
			while (i < (int)neighborsMaxLength)
			{
				int minPos = neighborsMaxLength;
				if (returnNeighbors[i].flag)
				{
					returnNeighbors[i].flag = false;
					unsigned n = returnNeighbors[i].id;

#if SIMD_TYPE >= SIMDTYPE_SSE
					_mm_prefetch(optGraph_ + nodeSize * n + dataLen, _MM_HINT_T0);
#endif

					unsigned *neighbors = (unsigned*)(optGraph_ + nodeSize * n + dataLen);
					unsigned MaxM = *neighbors;
					neighbors++;

#if SIMD_TYPE >= SIMDTYPE_SSE
					for (unsigned m = 0; m < MaxM; ++m)
						_mm_prefetch(optGraph_ + nodeSize * neighbors[m], _MM_HINT_T0);
#endif

					for (unsigned m = 0; m < MaxM; ++m)
					{
						unsigned id = neighbors[m];
						if (flags[id])continue;
						flags[id] = 1;
						float *data = (float*)(optGraph_ + nodeSize * id);
						float normID = *data; data++;

#ifdef COSINE_DISTANCE
						float dist = DistanceCosine::compare(data, normID, singleQueryData, normQuery, dimension_);
#else
						float dist = DistanceFastL2::compare(data, normID, singleQueryData, normQuery, dimension_);
#endif // COSINE_DISTANCE

						if (dist >= returnNeighbors[neighborsMaxLength - 1].distance)continue;
						Neighbor nn(id, dist, true);

						int insertPos = insertIntoPool(&returnNeighbors[0], neighborsMaxLength, nn);
						if (returnNeighbors.size() > neighborsMaxLength + 1)
						{
							returnNeighbors.pop_back();
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

			for (size_t i = 0; i < topK; i++)
			{
				returnIDs[i] = returnNeighbors[i].id;

#ifdef COSINE_DISTANCE
				returnSimilarities[i] = 1.0f - returnNeighbors[i].distance;

#else
				returnSimilarities[i] = 1.0f - 1.0f * DistanceL2::compare(singleQueryData, (*baseData_).at(returnIDs[i]),
					(unsigned)dimension_) / normQuery;
#endif // COSINE_DISTANCE

			}
		}

		void Search::saveResult(const char* resultPath, std::vector<std::vector<unsigned> > &returnIDs) {
			std::ofstream out(resultPath, std::ios::binary | std::ios::out);

			for (unsigned i = 0; i < returnIDs.size(); i++) {
				unsigned GK = (unsigned)returnIDs[i].size();
				out.write((char *)&GK, sizeof(unsigned));
				out.write((char *)returnIDs[i].data(), GK * sizeof(unsigned));
				if (i % 100 == 0)
				{
					out.flush();
				}
			}
			out.close();
		}

	}
}
