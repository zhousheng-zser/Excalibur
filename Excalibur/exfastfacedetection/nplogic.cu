#include "nplogic.hpp"
#ifdef USE_CUDA
#include <algorithm>
#include <iostream>
namespace glasssix
{
	struct faceInfo {
		float x;
		float y;
		float size;
		float score;
	};

	int compare(const void *a, const void *b)
	{
		struct faceInfo *pa = (struct faceInfo *)a;
		struct faceInfo *pb = (struct faceInfo *)b;
		float sub = pa->score - pb->score;
		return sub > 0 ? 1 : (sub < 0 ? -1 : 0);
	}

	__global__ void collect_gpu_kernel(int *m_xs, int *m_ys, int *m_sizes, float *m_scores, int *resultSubWin, float *result)
	{
		int id = blockIdx.x*blockDim.x + threadIdx.x;

		int num = resultSubWin[id];
		int offset = 0;
		for (int i = 0; i < id; i++)
		{
			offset += resultSubWin[i];
		}

		offset *= 4;

		int i = 0;
		for (int j = 0; j < num; i++)
		{
			if (m_xs[id + blockDim.x * gridDim.x * i] != -1)
			{
				result[offset + 4 * j] = (float)m_xs[id + blockDim.x * gridDim.x * i];
				result[offset + 4 * j + 1] = (float)m_ys[id + blockDim.x * gridDim.x * i];
				result[offset + 4 * j + 2] = (float)m_sizes[id + blockDim.x * gridDim.x * i];
				result[offset + 4 * j + 3] = m_scores[id + blockDim.x * gridDim.x * i];
				j++;
			}
		}
	}

	__global__ void scan_gpu_kernel(unsigned char *I, int width, int height, int *resultSubWin, int m_numScales, int m_numStages,
		int m_numBranchNodes, int *param, const int *m_pixelx, const int *m_pixely, const int *m_treeRoot,
		const unsigned char *features, const unsigned char *m_cutpoint, const int *m_rightChild, const int *m_leftChild, const float *m_fit,
		const float *m_stageThreshold, int *m_xs, int *m_ys, int *m_sizes, float *m_scores)
	{
		int id = blockIdx.x*blockDim.x + threadIdx.x;
		int numScales = m_numScales;
		int numStages = m_numStages;
		int numBranchNodes = m_numBranchNodes;
		//int objSize = m_objSize;
		int Height = height;
		int Width = width;

		int avalibleScales = param[6 * numScales];
		if (avalibleScales == 0)
			return;

		int numTotalSubWin = param[5 * numScales + avalibleScales - 1];

		for (int j = id; j < numTotalSubWin; j += gridDim.x * blockDim.x)
		{
			int winSize, winStep, colSteps, rowSteps, OriScaleIndex, rowStep, colStep;
			for (int i = 0; i < avalibleScales; i++)
			{
				int scaleIndex = j / param[5 * numScales + i];
				if (scaleIndex == 0)
				{
					winSize = param[i];
					winStep = param[numScales + i];
					colSteps = param[2 * numScales + i];
					rowSteps = param[3 * numScales + i];
					OriScaleIndex = param[4 * numScales + i];

					if (i > 0)
					{
						rowStep = j % param[5 * numScales + i - 1] / colSteps;
						colStep = j % param[5 * numScales + i - 1] - rowStep * colSteps;
					}
					else if (i == 0)
					{
						rowStep = j / colSteps;
						colStep = j - rowStep * colSteps;
					}

					break;
				}
			}

			unsigned char *pPixel = I + rowStep * winStep * Width + colStep * winStep;

			float _score = 0;
			int treeIndex = 0;
			int s;

			for (s = 0; s < numStages; s++)
			{
				int node = m_treeRoot[treeIndex];

				while (node > -1)
				{
					int offsetx = m_pixelx[OriScaleIndex * numBranchNodes + node] / winSize + m_pixelx[OriScaleIndex * numBranchNodes + node] % winSize * Width;
					int offsety = m_pixely[OriScaleIndex * numBranchNodes + node] / winSize + m_pixely[OriScaleIndex * numBranchNodes + node] % winSize * Width;
					unsigned char p1 = pPixel[offsetx];
					unsigned char p2 = pPixel[offsety];

					unsigned char fea = features[p2 * 256 + p1];
					if (fea < m_cutpoint[node] || fea > m_cutpoint[numBranchNodes + node])
						node = m_leftChild[node];
					else
						node = m_rightChild[node];
				}

				node = -node - 1;
				_score = _score + m_fit[node];
				treeIndex++;

				if (_score < m_stageThreshold[s])
					break; // negative samples
			}

			if (s == numStages) // a face detected
			{
				m_xs[j] = colStep * winStep;
				m_ys[j] = rowStep * winStep;
				m_sizes[j] = winSize;
				m_scores[j] = _score;
				resultSubWin[id] += 1;
			}
			else
			{
				m_xs[j] = -1;
			}
		}
	}

	
	int nplogic::scan_gpu(const unsigned char* I, int width, int height, int min_size)
	{
		if (cuda_level < 40)
		{
			return scan_gpu_template<40>(I, width, height, min_size);
		}
		else if (cuda_level >= 40 && cuda_level < 80)
		{
			return scan_gpu_template<80>(I, width, height, min_size);
		}
		else if (cuda_level >= 80 && cuda_level < 120)
		{
			return scan_gpu_template<120>(I, width, height, min_size);
		}
		else if (cuda_level >= 120 && cuda_level < 160)
		{
			return scan_gpu_template<160>(I, width, height, min_size);
		}
		else if (cuda_level >= 160 && cuda_level < 200)
		{
			return scan_gpu_template<200>(I, width, height, min_size);
		}
		else if (cuda_level >= 240 && cuda_level < 280)
		{
			return scan_gpu_template<280>(I, width, height, min_size);
		}
		else if (cuda_level >= 280 && cuda_level < 320)
		{
			return scan_gpu_template<320>(I, width, height, min_size);
		}
		else if (cuda_level >= 320 && cuda_level < 360)
		{
			return scan_gpu_template<360>(I, width, height, min_size);
		}
		else if (cuda_level >= 360 && cuda_level < 400)
		{
			return scan_gpu_template<400>(I, width, height, min_size);
		}
		else // cuda_level > 400
		{
			return scan_gpu_template<440>(I, width, height, min_size);
		}
	}

	template<int gridDimension>
	int nplogic::scan_gpu_template(const unsigned char* I, int width, int height, int min_size)
	{
		const int gridDim = gridDimension;
		const int blockDim = 128;

		int *param_g = 0;
		const int* treeRoot_data = model_->treeRoot->gpu_data();
		const int * pixelx_data = model_->pixelx->gpu_data();
		const int * pixely_data = model_->pixely->gpu_data();
		const unsigned char* cutpoint_data = model_->cutpoint->gpu_data();
		const int* leftchild_data = model_->leftChild->gpu_data();
		const int* rightchild_data = model_->rightChild->gpu_data();
		const float* fit_data = model_->fit->gpu_data();
		const float* stage_thres_data = model_->stageThreshold->gpu_data();
		const unsigned char* fea_data = model_->fea->gpu_data();
		const int nsc = model_->numScales;
		const int nst = model_->numStages;
		const int nbn = model_->numBranchNodes;

		struct faceInfo * faceVec = 0;
		std::vector<int> index;
		this->minFace = min_size;
		int minFace = std::max(this->minFace, model_->objSize);
		int maxFace = std::min(height, width);

		if (maxFace < minFace)
		{
			fprintf(stderr, "maxFace(%d) < minFace(%d)!", maxFace, minFace);
			return 0;
		}

		int paramIntSize = 7 * model_->numScales;
		int *param = (int *)malloc(paramIntSize * sizeof(int));

		int numTotalSubWin = 0;
		int j = 0;
		param[6 * model_->numScales] = 0;
		const int* winSize_data = model_->winSize->cpu_data();
		for (int i = 0; i < model_->numScales; i++)
		{
			int winSize = winSize_data[i];
			if (winSize < minFace) continue;
			if (winSize > maxFace) break;
			int winStep = (int)floor(winSize * 0.1);
			if (winSize > 40)
				winStep = (int)floor(winSize * 0.05);
			int colMax = width - winSize;
			int rowMax = height - winSize;
			int colSteps = colMax / winStep + 1;
			int rowSteps = rowMax / winStep + 1;
			int numSubWin = colSteps * rowSteps;
			numTotalSubWin += numSubWin;
			param[j] = winSize;
			param[model_->numScales + j] = winStep;
			param[2 * model_->numScales + j] = colSteps;
			param[3 * model_->numScales + j] = rowSteps;
			param[4 * model_->numScales + j] = i;
			param[5 * model_->numScales + j] = numTotalSubWin;
			j++;
		}
		param[6 * model_->numScales] = j;

		int numThreads = gridDim * blockDim;
		int numImageUChar = width * height;
		int memSize = (paramIntSize + numThreads) * sizeof(int) + numImageUChar * sizeof(unsigned char);
		param = (int *)realloc(param, memSize);
		memset(param + paramIntSize, 0, numThreads * sizeof(int));
		memcpy(param + paramIntSize + numThreads, I, numImageUChar * sizeof(unsigned char));
		CUDA_CHECK(cudaMalloc((void**)&param_g, 8 * numTotalSubWin * sizeof(int) + memSize));
		CUDA_CHECK(cudaMemcpy(param_g + 8 * numTotalSubWin, param, memSize, cudaMemcpyHostToDevice));

		scan_gpu_kernel << <gridDim, blockDim >> >(
			(unsigned char *)(param_g + 8 * numTotalSubWin + paramIntSize + numThreads),
			width, height,
			param_g + 8 * numTotalSubWin + paramIntSize,
			nsc,
			nst,
			nbn,
			param_g + 8 * numTotalSubWin,
			pixelx_data,
			pixely_data,
			treeRoot_data,
			fea_data,
			cutpoint_data,
			rightchild_data,
			leftchild_data,
			fit_data,
			stage_thres_data,
			param_g,
			param_g + numTotalSubWin,
			param_g + 2 * numTotalSubWin,
			(float *)(param_g + 3 * numTotalSubWin));

		collect_gpu_kernel << <gridDim, blockDim >> > (
			param_g,
			param_g + numTotalSubWin,
			param_g + 2 * numTotalSubWin,
			(float *)(param_g + 3 * numTotalSubWin),
			param_g + 8 * numTotalSubWin + paramIntSize,
			(float *)(param_g + 4 * numTotalSubWin));

		int resultNum[gridDim * blockDim] = { 0 };
		CUDA_CHECK(cudaMemcpy(resultNum, param_g + 8 * numTotalSubWin + paramIntSize, numThreads * sizeof(int), cudaMemcpyDeviceToHost));
		int sum = 0;
		for (int i = 0; i < gridDim * blockDim; i++)
		{
			sum += resultNum[i];
		}

		faceVec = (struct faceInfo *)malloc(sum * sizeof(struct faceInfo));
		CUDA_CHECK(cudaMemcpy(faceVec, (float *)(param_g + 4 * numTotalSubWin), 4 * sum * sizeof(float), cudaMemcpyDeviceToHost));


		std::qsort(faceVec, sum, sizeof(struct faceInfo), compare);

		for (int j = 0; j < sum; j++)
		{
			xs.push_back(faceVec[j].x);
			ys.push_back(faceVec[j].y);
			sizes.push_back(faceVec[j].size);
			scores.push_back(faceVec[j].score);
		}

		free(faceVec);
		free(param);

		cudaFree(param_g);

		return sum;
	}

}
#endif // USE_CUDA