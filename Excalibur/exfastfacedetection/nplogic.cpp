#include "nplogic.hpp"
#include <algorithm>

namespace glasssix
{
	nplogic::nplogic(int device)
	{
		device_ = device;
		if (device_>=0)
		{
#ifdef USE_CUDA
			int cudacores = GetDeviceCUDACoreNum(device_);
			cuda_level = cudacores / 64 * 5;
#else
			NO_GPU;
#endif
		}
	}

	nplogic::nplogic(int minFace, int maxFace, int device)
	{
		device_ = device;
		if (device_ >= 0)
		{
#ifdef USE_CUDA
			int cudacores = GetDeviceCUDACoreNum(device_);
			cuda_level = cudacores / 64 * 5;
#else
			NO_GPU;
#endif
		}
		init(minFace, maxFace);
	}

	void nplogic::load()
	{
		model_.reset(new npcalculator(device_));
		init(model_->objSize, 4096);
	}


	void nplogic::load(const char* modelpath)
	{
		model_.reset(new npcalculator(modelpath, device_));
		init(model_->objSize, 4096);
	}

	void nplogic::init(int minFace, int maxFace)
	{
		this->minFace = minFace;
		this->maxFace = maxFace;
		overlappingThreshold = 0.5;
		maxScanNum = 0;
		maxDetectNum = 0;

		Tpredicate = nullptr;
		Troot = nullptr;
		Tlogweight = nullptr;
		Tparent = nullptr;
		Trank = nullptr;
		mallocsacnspace(500);

		Tneighbors = nullptr;
		Tweight = nullptr;
		Txs = nullptr;
		Tys = nullptr;
		Tss = nullptr;
		mallocdetectspace(40);
	}

	int nplogic::detect(const unsigned char* I, int width, int height, int min_size)
	{
		// Clear former data.
		reset();
		// Scan with model.
		if (device_<0)
		{
			numScan = scan_cpu(I, width, height, min_size);
		}
		else
		{
#ifdef USE_CUDA
			numScan = scan_gpu(I, width, height, min_size);
#else
			NO_GPU;
#endif
		}
		if (numScan > maxScanNum)
		{
			if (2 * maxScanNum > numScan)
				mallocsacnspace(maxScanNum * 2);
			else
				mallocsacnspace(numScan + maxScanNum);
		}

		// Merge rect.
		numDetect = nms();

		return numDetect;
	}
	
	int nplogic::floodScoreMat(std::shared_ptr<excalibur::tensor<float>> mat, int mat_height, int mat_width, int rowMax, int colMax, int winStep)
	{
		int rows = mat_height;
		int cols = mat_width;

		int yGridNum = rowMax / winStep;
		int xGridNum = colMax / winStep;

		float* mat_data = mat->mutable_cpu_data();
		//for cols
		for (int c = 0; c < colMax; c += winStep)
		{
			for (int g = 0; g < yGridNum; g++)
			{
				int begin = g*winStep;
				int end = begin + winStep;
				float beginVal = mat_data[c * mat_width + begin];//mat.at<float>(begin, c);
				float endVal = mat_data[c * mat_width + end]; //mat.at<float>(end, c);

				for (int r = begin + 1; r < end; r++)
				{
					mat_data[c * mat_width + r] = ((r - begin)*endVal + (end - r)*beginVal) / winStep;
				}
			}
			int begin = yGridNum *winStep;
			int end = rows;
			int step = rows - begin;
			float beginVal = mat_data[c * mat_width + begin];//mat.at<float>(begin, c);
			float endVal = mat_data[c * mat_width + 0];//mat.at<float>(0, c);
			for (int r = begin + 1; r < end; r++)
			{
				mat_data[c * mat_width + r] = ((r - begin)*endVal + (end - r)*beginVal) / step;
			}
		}

		//for rows
		for (int r = 0; r < rows; r++)
		{
			float* rowHead = mat_data + r * mat_width;//mat.ptr<float>(r);
			for (int g = 0; g < xGridNum; g++)
			{
				int begin = g*winStep;
				int end = begin + winStep;
				float beginVal = rowHead[begin];
				float endVal = rowHead[end];
				for (int c = begin + 1; c < end; c++)
				{
					rowHead[c] = ((c - begin) * endVal + (end - c) * beginVal) / winStep;
				}
			}
			int begin = xGridNum *winStep;
			int end = cols;
			int step = end - begin;
			float beginVal = rowHead[begin];
			float endVal = rowHead[0];
			for (int c = begin + 1; c < end; c++)
			{
				rowHead[c] = ((c - begin) * endVal + (end - c) * beginVal) / step;
			}
		}
		return 0;
	}

	void nplogic::mallocdetectspace(int n)
	{
		Tneighbors.reset(new excalibur::tensor<int>(n, device_));
		Tweight.reset(new excalibur::tensor<float>(n, device_));
		Txs.reset(new excalibur::tensor<float>(n, device_));
		Tys.reset(new excalibur::tensor<float>(n, device_));
		Tss.reset(new excalibur::tensor<float>(n, device_));
		maxDetectNum = n;
	}

	void nplogic::mallocsacnspace(int s)
	{
		Tpredicate.reset(new excalibur::tensor<char>(s*s, device_));
		Troot.reset(new excalibur::tensor<int>(s, device_));
		Tlogweight.reset(new excalibur::tensor<float>(s, device_));
		Tparent.reset(new excalibur::tensor<int>(s, device_));
		Trank.reset(new excalibur::tensor<int>(s, device_));
		maxScanNum = s;
	}

	int nplogic::nms()
	{
		if (numScan <= 0)
			return 0;

		int i, j, ni, nj;
		float h, w, s, si, sj;
		char* Tpredicate_data = Tpredicate->mutable_cpu_data();
		memset(Tpredicate_data, 0, sizeof(char) * numScan * numScan);

		// mark nearby detections
		for (i = 0; i < numScan; i++)
		{
			ni = i * numScan;
			for (j = 0; j < numScan; j++)
			{
				nj = j * numScan;
				h = std::min(ys[i] + sizes[i], ys[j] + sizes[j]) -
					std::max(ys[i], ys[j]);
				w = std::min(xs[i] + sizes[i], xs[j] + sizes[j]) -
					std::max(xs[i], xs[j]);
				s = std::max(double(h), 0.0) * std::max(double(w), 0.0);
				si = sizes[i] * sizes[i];
				sj = sizes[j] * sizes[j];

				// 1. Overlap 50%
				if ((s / (si + sj - s)) >=
					overlappingThreshold)
				{
					Tpredicate_data[ni + j] = 1;
					Tpredicate_data[nj + i] = 1;
				}
			}
		}
		int* Troot_data = Troot->mutable_cpu_data();
		for (i = 0; i < numScan; i++)
			Troot_data[i] = -1;
		int n = partition(Tpredicate->cpu_data(), Troot_data);
		if (n > maxDetectNum)
			mallocdetectspace(n + 40);

		float* Tlogweight_data = Tlogweight->mutable_cpu_data();
		for (i = 0; i < numScan; i++)
		{
			Tlogweight_data[i] = logistic(scores[i]);
		}
		float* Tweight_data = Tweight->mutable_cpu_data();
		int* Tneighbors_data = Tneighbors->mutable_cpu_data();
		float* Txs_data = Txs->mutable_cpu_data();
		float* Tys_data = Tys->mutable_cpu_data();
		float* Tss_data = Tss->mutable_cpu_data();
		memset(Tweight_data, 0, sizeof(float) * n);
		memset(Tneighbors_data, 0, sizeof(int) * n);
		memset(Txs_data, 0, sizeof(float) * n);
		memset(Tys_data, 0, sizeof(float) * n);
		memset(Tss_data, 0, sizeof(float) * n);
		for (i = 0; i < numScan; i++)
		{
			Tweight_data[Troot_data[i]] += Tlogweight_data[i];
			Tneighbors_data[Troot_data[i]] += 1;
		}

		for (i = 0; i < numScan; i++)
		{
			if (Tweight_data[Troot_data[i]] != 0)
				Tlogweight_data[i] /= Tweight_data[Troot_data[i]];
			else
				Tlogweight_data[i] = 1 / Tneighbors_data[Troot_data[i]];
			Txs_data[Troot_data[i]] += xs[i] * Tlogweight_data[i];
			Tys_data[Troot_data[i]] += ys[i] * Tlogweight_data[i];
			Tss_data[Troot_data[i]] += sizes[i] * Tlogweight_data[i];
		}

		for (i = 0; i < n; i++)
		{
			Xs.push_back(int(Txs_data[i]));
			Ys.push_back(int(Tys_data[i]));
			Ss.push_back(int(Tss_data[i]));
			Scores.push_back((Tweight_data[i]));
		}
		return n;
	}

	int nplogic::partition(const char* predicate, int* root)
	{
		int i, j, ni;
		int root_i, root_j;
		int* trank_data = Trank->mutable_cpu_data();
		int* tparent_data = Tparent->mutable_cpu_data();
		for (i = 0; i < numScan; i++)
		{
			tparent_data[i] = i;
			trank_data[i] = 0;
		}


		ni = 0;
		for (i = 0; i < numScan; i++)
		{
			for (j = 0; j < numScan; j++, ni++)
			{
				if (predicate[ni] == 0)
					continue;

				root_i = findRoot(Tparent->cpu_data(), i);
				root_j = findRoot(Tparent->cpu_data(), j);

				if (root_i != root_j)
				{
					if (trank_data[i] > trank_data[j])
						tparent_data[root_j] = root_i;
					else if (trank_data[i] < trank_data[j])
						tparent_data[root_i] = root_j;
					else
					{
						tparent_data[root_j] = root_i;
						trank_data[root_i] ++;
					}
				}
			}
		}

		int n = 0;
		for (i = 0; i < numScan; i++)
		{
			if (tparent_data[i] == i)
			{
				if (root[i] == -1)
					root[i] = n++;
				continue;
			}

			root_i = findRoot(Tparent->cpu_data(), i);
			if (root[root_i] == -1)
				root[root_i] = n++;
			root[i] = root[root_i];
		}

		return n;
	}

	void nplogic::reset()
	{
		xs.clear();
		ys.clear();
		sizes.clear();
		scores.clear();
		Xs.clear();
		Ys.clear();
		Ss.clear();
		Scores.clear();
		numScan = 0;
	}

	int nplogic::scan_cpu(const unsigned char* I, int width, int height, int min_size)
	{
		this->minFace = min_size;
		int minFace = std::max(this->minFace, model_->objSize);
		int maxFace = std::min(this->maxFace, std::min(height, width));
		const unsigned char* fea_data = model_->fea->cpu_data();
		if (std::min(height, width) < minFace)
			return 0;

		const int* treeRoot_data = model_->treeRoot->cpu_data();
		const int * pixelx_data = model_->pixelx->cpu_data();
		const int * pixely_data = model_->pixely->cpu_data();
		const unsigned char* cutpoint_data = model_->cutpoint->cpu_data();
		const int* leftchild_data = model_->leftChild->cpu_data();
		const int* rightchild_data = model_->rightChild->cpu_data();
		const float* fit_data = model_->fit->cpu_data();
		const float* stage_thres_data = model_->stageThreshold->cpu_data();

		int k;
		for (k = 0; k < model_->numScales; k++) // process each scale
		{
			const int* winSize_data = model_->winSize->cpu_data();
			if (winSize_data[k] < minFace) continue;
			else if (winSize_data[k] > maxFace) break;
			// determine the step of the sliding subwindow
			int winStep = (int)floor(winSize_data[k] * 0.1);
			if (winSize_data[k] > 40) winStep = (int)floor(winSize_data[k] * 0.05);

			// calculate the offset values of each pixel in a subwindow
			// pre-determined offset of pixels in a subwindow
			//std::vector<int> offset(winSize_data[k] * winSize_data[k]);
			int* offset = new int[winSize_data[k] * winSize_data[k]];
			int p1 = 0, p2 = 0, gap = width;

			for (int j = 0; j < winSize_data[k]; j++) // column coordinate
			{
				p2 = j;
				for (int i = 0; i < winSize_data[k]; i++) // row coordinate
				{
					offset[p1++] = p2;
					p2 += gap;
				}
			}
			//t = ((double)cvGetTickCount() - t) / ((double)cvGetTickFrequency()*1000.) ;
			//tcnt += t;

			int colMax = width - winSize_data[k] + 1;
			int rowMax = height - winSize_data[k] + 1;

			// process each subwindow
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int r = 0; r < rowMax; r += winStep) // slide in row
			{
				const unsigned char *pPixel = I + r * width;
				for (int c = 0; c < colMax; c += winStep, pPixel += winStep) // slide in column

				{
					int treeIndex = 0;
					float _score = 0;
					int s;

					// test each tree classifier
					for (s = 0; s < model_->numStages; s++)
					{
						int node = treeRoot_data[treeIndex];

						// test the current tree classifier
						int BranchNodes = model_->numBranchNodes;
						while (node > -1) // branch node
						{
							unsigned char p1 = pPixel[
								offset[pixelx_data[k * BranchNodes + node]]];
							unsigned char p2 = pPixel[
								offset[pixely_data[k * BranchNodes + node]]];
							unsigned char fea = fea_data[p2 * 256 + p1];

							if (fea < cutpoint_data[node]
								|| fea > cutpoint_data[BranchNodes + node])
								node = leftchild_data[node];
							else
								node = rightchild_data[node];
						}

						// leaf node
						node = -node - 1;
						_score = _score + fit_data[node];
						treeIndex++;

						//printf("stage = %d, score = %f\n", s, _score);
						if (_score < stage_thres_data[s])
							break; // negative samples
					}

					if (s == model_->numStages) // a face detected
					{
#ifdef _OPENMP
#pragma omp critical
						{
#endif
							ys.push_back(r + 1);
							xs.push_back(c + 1);
							sizes.push_back(winSize_data[k]);
							scores.push_back(_score);
#ifdef _OPENMP
						}
#endif
					}
				} // Cols.
			} // Row.

			delete offset;
		} // Scale.
		return (int)ys.size();
	}

}