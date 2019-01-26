#include "InternalLonginusCascade.hpp"
#include "ImageOperation.hpp"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define INTEGRAL_CALC_SUM(p0, p1, p2, p3, offset_) \
((p0)[offset_] - (p1)[offset_] - (p2)[offset_] + (p3)[offset_])

using namespace glasssix::longinus;

extern const int map_lut_length[];
extern const unsigned char LBPMAP[][256];

extern void extendWin(int ix, int iy, int xstep, int ystep, int xmax, int ymax, std::vector<glasssix::longinus::Point> &points);
extern void UpdateCascade(const int *p_fea_info, int numStages, const int *p_weak_num_in_stages, const int **pNode, const int *pSum, int sum_width);

static inline __device__ double DetectAt(const int * const *pNode, int offset, const double *regression_value, int lut_len, const double *stage_threshold, const int * weak_num_in_stages,
	const double *weak_threshold, const unsigned char *LBPMAP, int numStages, bool doEarlyReject)
{
	double stage_sum = 0;

	const int* const *p = pNode;

	for (int i = 0; i < numStages; i++)
	{
		stage_sum = 0;
		int weakCount = weak_num_in_stages[i];
		for (int j = 0; j < weakCount; j++, p += 36, regression_value += lut_len, weak_threshold++)
		{
			int cval = INTEGRAL_CALC_SUM(p[14], p[15], p[20], p[21], offset);

			int code = ((INTEGRAL_CALC_SUM(p[0], p[1], p[6], p[7], offset) >= cval) << 7) |
				((INTEGRAL_CALC_SUM(p[2], p[3], p[8], p[9], offset) >= cval) << 6) |
				((INTEGRAL_CALC_SUM(p[4], p[5], p[10], p[11], offset) >= cval) << 5) |
				((INTEGRAL_CALC_SUM(p[16], p[17], p[22], p[23], offset) >= cval) << 4) |
				((INTEGRAL_CALC_SUM(p[28], p[29], p[34], p[35], offset) >= cval) << 3) |
				((INTEGRAL_CALC_SUM(p[26], p[27], p[32], p[33], offset) >= cval) << 2) |
				((INTEGRAL_CALC_SUM(p[24], p[25], p[30], p[31], offset) >= cval) << 1) |
				((INTEGRAL_CALC_SUM(p[12], p[13], p[18], p[19], offset) >= cval));

			stage_sum += regression_value[LBPMAP[code]];

			//doEarlyReject dose not work on gpu. It will not take any benifit on speed.
			/*if (doEarlyReject && (stage_sum < *weak_threshold))
				return -i;*/
		}

		if (stage_sum < stage_threshold[i])
			return -i;
		else
			stage_sum = stage_sum - stage_threshold[i] + 1;
	}

	return stage_sum;
}

__global__ void scan_kernel(int I_width, const int * const *pNode, const double *regression_value, int lut_len, const double *stage_threshold, 
	const int * weak_num_in_stages, const double *weak_threshold, const unsigned char *LBPMAP, double * result, int numStages, int xstep, int ystep, int numSubWin, int colSteps, bool doEarlyReject)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = threadId; i < numSubWin; i += gridDim.x * blockDim.x)
	{
		int iy = i / colSteps * ystep;
		int ix = i % colSteps * xstep;
		int w_offset = iy * I_width + ix;

		result[i] = DetectAt(pNode, w_offset, regression_value, lut_len, stage_threshold, weak_num_in_stages, weak_threshold, LBPMAP, numStages, doEarlyReject);
	}
}

static inline double DetectAt_CPU(const int * const *pNode, int offset, const double *regression_value, int lut_len, const double *stage_threshold, const int * weak_num_in_stages,
	const double *weak_threshold, const unsigned char *pLBPMAP, int numStages, bool doEarlyReject)
{
	double stage_sum = 0;

	const int* const *p = pNode;
	for (int i = 0; i < numStages; i++)
	{
		stage_sum = 0;
		int weakCount = weak_num_in_stages[i];
		for (int j = 0; j < weakCount; j++, p += 36, regression_value += lut_len, weak_threshold++)
		{
			int cval = INTEGRAL_CALC_SUM(p[14], p[15], p[20], p[21], offset);

			int code = ((INTEGRAL_CALC_SUM(p[0], p[1], p[6], p[7], offset) >= cval) << 7) |
				((INTEGRAL_CALC_SUM(p[2], p[3], p[8], p[9], offset) >= cval) << 6) |
				((INTEGRAL_CALC_SUM(p[4], p[5], p[10], p[11], offset) >= cval) << 5) |
				((INTEGRAL_CALC_SUM(p[16], p[17], p[22], p[23], offset) >= cval) << 4) |
				((INTEGRAL_CALC_SUM(p[28], p[29], p[34], p[35], offset) >= cval) << 3) |
				((INTEGRAL_CALC_SUM(p[26], p[27], p[32], p[33], offset) >= cval) << 2) |
				((INTEGRAL_CALC_SUM(p[24], p[25], p[30], p[31], offset) >= cval) << 1) |
				((INTEGRAL_CALC_SUM(p[12], p[13], p[18], p[19], offset) >= cval));

			stage_sum += regression_value[pLBPMAP[code]];

			if (doEarlyReject && (stage_sum < *weak_threshold))
				return -i;
		}

		if (stage_sum < stage_threshold[i])
			return -i;
		else
			stage_sum = stage_sum - stage_threshold[i] + 1;
	}

	return stage_sum;
}

void glasssix::longinus::InternalLonginusCascade::scan_gpu(excalibur::tensor<int>& I, std::vector<CandidateRect>& rects, int xstep, int ystep, int xmax, int ymax, int sum_width,
	int factor1024x, int lut_len, bool doEarlyReject)
{
	UpdateCascade(tensor_fea_info.cpu_data(), numStages, tensor_weak_num_in_stages.cpu_data(), tensor_pNode.mutable_cpu_data(), I.gpu_data(), I.width());

	int rowSteps = ymax / ystep + 1;
	int colSteps = xmax / xstep + 1;
	int numSubWin = colSteps * rowSteps;
	excalibur::tensor<double> tensor_result(numSubWin, device_);

	unsigned int dimX = numSubWin / 128 + ((numSubWin % 128) == 0 ? 0 : 1);
	dim3 gridDim(dimX > 60 ? 60 : dimX);
	dim3 blockDim(128);

	scan_kernel << <gridDim, blockDim >> > (sum_width, tensor_pNode.gpu_data(), tensor_regression_value.gpu_data(), lut_len,
		tensor_stage_threshold.gpu_data(), tensor_weak_num_in_stages.gpu_data(), tensor_weak_threshold.gpu_data(), tensor_FeaMap.gpu_data(), tensor_result.mutable_gpu_data(), 
		numStages, xstep, ystep, numSubWin, colSteps, doEarlyReject);

	UpdateCascade(tensor_fea_info.cpu_data(), numStages, tensor_weak_num_in_stages.cpu_data(), tensor_pNode.mutable_cpu_data(), I.cpu_data(), I.width());

	const double *result = tensor_result.cpu_data();
	for (int i = 0; i < tensor_result.count(); i++)
	{
		if (result[i] > 0)
		{
			CandidateRect fr;

			fr.ix = i % colSteps * xstep;
			fr.iy = i / colSteps * ystep;

			fr.x = (fr.ix * factor1024x + 512) >> 10;
			fr.y = (fr.iy * factor1024x + 512) >> 10;
			fr.width = (win_width * factor1024x + 512) >> 10;
			fr.height = (win_height * factor1024x + 512) >> 10;

			fr.confidence = result[i];

			fr.xstep = xstep;
			fr.ystep = ystep;
			fr.xmax = xmax;
			fr.ymax = ymax;

			rects.push_back(fr);
		}
		else if (result[i] < -4)
		{
			int ix = i % colSteps * xstep;
			int iy = i / colSteps * ystep;
			std::vector<Point> points;
			extendWin(ix, iy, xstep, ystep, xmax, ymax, points);
			for (size_t i = 0; i < points.size(); i++)
			{
				int w_offset = points[i].y * sum_width + points[i].x;
				double confidence = DetectAt_CPU(tensor_pNode.cpu_data(), w_offset, tensor_regression_value.cpu_data(), lut_len, tensor_stage_threshold.cpu_data(),
					tensor_weak_num_in_stages.cpu_data(), tensor_weak_threshold.cpu_data(), tensor_FeaMap.cpu_data(), numStages, doEarlyReject);
				if (confidence > 0)
				{
					CandidateRect fr;

					fr.ix = points[i].x;
					fr.iy = points[i].y;

					fr.x = (fr.ix * factor1024x + 512) >> 10;
					fr.y = (fr.iy * factor1024x + 512) >> 10;
					fr.width = (win_width * factor1024x + 512) >> 10;
					fr.height = (win_height * factor1024x + 512) >> 10;

					fr.confidence = confidence;

					fr.xstep = xstep;
					fr.ystep = ystep;
					fr.xmax = xmax;
					fr.ymax = ymax;
					rects.push_back(fr);
				}
			}
		}
	}
}
#endif