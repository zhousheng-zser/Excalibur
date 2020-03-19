#ifdef TRIAL
#include "tinyxml2.h"
#endif
#include "common.hpp"
#include "InternalLonginusCascade.hpp"
#include "ImageOperation.hpp"
#include "HardCode.hpp"

#include "FrontalCascade.hpp"
#include "LeftProfileCascade.hpp"
#include "RightProfileCascade.hpp"

#include "FrontalReinforceCascade.hpp"
#include "LeftProfileReinforceCascade.hpp"
#include "RightProfileReinforceCascade.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
//#include <opencv2/opencv.hpp>

#define INTEGRAL_CALC_SUM(p0, p1, p2, p3, offset_) \
((p0)[offset_] - (p1)[offset_] - (p2)[offset_] + (p3)[offset_])

using namespace glasssix::longinus;
using namespace glasssix::excalibur;

//move definition of LBPMAP[5][256] to tensor_operation_cpu.cpp
//extern const unsigned char LBPMAP[5][256] = {}

extern const unsigned char LBPMAP[][256];
extern const int map_lut_length[5] = { 59, 63, 63, 51, 53 };

std::vector<std::string> split(const std::string &s, const std::string &seperator)
{
	std::vector<std::string> result;
	typedef std::string::size_type string_size;
	string_size i = 0;

	while (i != s.size())
	{
		int flag = 0;
		while (i != s.size() && flag == 0)
		{
			flag = 1;
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[i] == seperator[x])
				{
					++i;
					flag = 0;
					break;
				}
		}

		flag = 0;
		string_size j = i;
		while (j != s.size() && flag == 0)
		{
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[j] == seperator[x])
				{
					flag = 1;
					break;
				}
			if (flag == 0)
				++j;
		}
		if (i != j)
		{
			result.push_back(s.substr(i, j - i));
			i = j;
		}
	}
	return result;
}


extern void extendWin(int ix, int iy, int xstep, int ystep, int xmax, int ymax, std::vector<glasssix::longinus::Point> &points)
{
	points.clear();
	int half_xstep = xstep / 2;
	int half_ystep = ystep / 2;
	while (half_xstep >= 1 || half_ystep >= 1)
	{
		int ix_extend = ix - half_xstep;
		int iy_extend = iy - half_ystep;

		if (half_xstep && (ix_extend >= 0) && half_ystep && (iy_extend >= 0))
		{
			points.emplace_back(ix_extend, iy_extend);
		}

		if (half_xstep && (ix_extend >= 0))
		{
			points.emplace_back(ix_extend, iy);
		}

		if (half_ystep && (iy_extend >= 0))
		{
			points.emplace_back(ix, iy_extend);
		}

		ix_extend = ix + half_xstep;
		if (half_xstep && (ix_extend <= xmax) && half_ystep && (iy_extend >= 0))
		{
			points.emplace_back(ix_extend, iy_extend);
		}

		if (half_xstep && (ix_extend <= xmax))
		{
			points.emplace_back(ix_extend, iy);
		}

		iy_extend = iy + half_ystep;

		if (half_xstep && (ix_extend <= xmax) && half_ystep && (iy_extend <= ymax))
		{
			points.emplace_back(ix_extend, iy_extend);
		}

		if (half_ystep && (iy_extend <= ymax))
		{
			points.emplace_back(ix, iy_extend);
		}

		ix_extend = ix - half_xstep;
		if (half_xstep && (ix_extend >= 0) && half_ystep && (iy_extend <= ymax))
		{
			points.emplace_back(ix_extend, iy_extend);
		}

		half_xstep = half_xstep - 1;
		half_ystep = half_ystep - 1;
	}
}

InternalLonginusCascade::InternalLonginusCascade()
	: numStages(0), numWeaks(0), win_width(0), win_height(0), map_mode(0), fea_mode(0), device_(-1)
{
}

#ifdef TRIAL
void InternalLonginusCascade::LoadCascade(const std::string & filename, int device)
{
	if (device >= 0)
	{
#ifndef USE_CUDA
		NO_GPU;
#endif // CPU only
	}

	device_ = device;

	tinyxml2::XMLDocument doc;
	if (doc.LoadFile(filename.c_str()) != tinyxml2::XML_SUCCESS)
		return;

	const char *version = doc.FirstChildElement("opencv_storage")->FirstChildElement("cascade")->FirstChildElement("version")->GetText();
	win_width = atoi(doc.FirstChildElement("opencv_storage")->FirstChildElement("cascade")->FirstChildElement("width")->GetText());
	win_height = atoi(doc.FirstChildElement("opencv_storage")->FirstChildElement("cascade")->FirstChildElement("height")->GetText());
	face_width = atoi(doc.FirstChildElement("opencv_storage")->FirstChildElement("cascade")->FirstChildElement("fwidth")->GetText());
	face_height = atoi(doc.FirstChildElement("opencv_storage")->FirstChildElement("cascade")->FirstChildElement("fheight")->GetText());
	numStages = atoi(doc.FirstChildElement("opencv_storage")->FirstChildElement("cascade")->FirstChildElement("stageNum")->GetText());
	numWeaks = atoi(doc.FirstChildElement("opencv_storage")->FirstChildElement("cascade")->FirstChildElement("weakNum")->GetText());
	map_mode = atoi(doc.FirstChildElement("opencv_storage")->FirstChildElement("cascade")->FirstChildElement("LBPMode")->GetText());
	fea_mode = atoi(doc.FirstChildElement("opencv_storage")->FirstChildElement("cascade")->FirstChildElement("feaMode")->GetText());

	tensor_weak_num_in_stages = tensor<int>(numStages, device_);
	tensor_stage_threshold = tensor<double>(numStages, device_);
	tensor_stage_far = tensor<double>(numStages, device_);
	tensor_fea_index = tensor<int>(numWeaks, device_);
	tensor_fea_info = tensor<int>({ numWeaks, 8 }, device_);
	tensor_weak_threshold = tensor<double>(numWeaks, device_);
	tensor_regression_value = tensor<double>({ numWeaks, map_lut_length[map_mode] }, device_);

	tensor_pNode = tensor<const int *>({ numWeaks, 36 }, device_);
	tensor_FeaMap = tensor<unsigned char>(sizeof(LBPMAP[map_mode]), device_);

	unsigned char *p_lbpmap = tensor_FeaMap.mutable_cpu_data();
	for (size_t i = 0; i < sizeof(LBPMAP[map_mode]); i++)
		p_lbpmap[i] = LBPMAP[map_mode][i];

	tinyxml2::XMLElement *root = doc.FirstChildElement("opencv_storage")->FirstChildElement("cascade")->FirstChildElement("stages");
	int weakIndex = 0;
	int tmp_i = 0;
	for (tinyxml2::XMLElement *stage = root->FirstChildElement("_"); stage != NULL; stage = stage->NextSiblingElement("_"))
	{
		// iterate Stage
		tensor_weak_num_in_stages.mutable_cpu_data()[tmp_i] = atoi(stage->FirstChildElement("weakCount")->GetText());
		tensor_stage_threshold.mutable_cpu_data()[tmp_i] = atof(stage->FirstChildElement("stageThreshold")->GetText());
		tensor_stage_far.mutable_cpu_data()[tmp_i] = atof(stage->FirstChildElement("falseAlarm")->GetText());
		
		for (tinyxml2::XMLElement *weak_classifier = stage->FirstChildElement("weakClassifiers")->FirstChildElement("_"); weak_classifier != NULL; weak_classifier = weak_classifier->NextSiblingElement("_"))
		{
			// iterate Weak Classifier
			tensor_fea_index.mutable_cpu_data()[weakIndex] = atoi(weak_classifier->FirstChildElement("featureindex")->GetText());

			int *p_fea_info = tensor_fea_info.mutable_cpu_data();
			p_fea_info[weakIndex * 8 + 0] = atoi(weak_classifier->FirstChildElement("x")->GetText());
			p_fea_info[weakIndex * 8 + 1] = atoi(weak_classifier->FirstChildElement("y")->GetText());
			p_fea_info[weakIndex * 8 + 2] = atoi(weak_classifier->FirstChildElement("width")->GetText());
			p_fea_info[weakIndex * 8 + 3] = atoi(weak_classifier->FirstChildElement("height")->GetText());
			p_fea_info[weakIndex * 8 + 4] = atoi(weak_classifier->FirstChildElement("cellwidth")->GetText());
			p_fea_info[weakIndex * 8 + 5] = atoi(weak_classifier->FirstChildElement("cellheight")->GetText());
			p_fea_info[weakIndex * 8 + 6] = atoi(weak_classifier->FirstChildElement("cellstepX")->GetText());
			p_fea_info[weakIndex * 8 + 7] = atoi(weak_classifier->FirstChildElement("cellstepY")->GetText());

			tensor_weak_threshold.mutable_cpu_data()[weakIndex] = atof(weak_classifier->FirstChildElement("weakThreshold")->GetText());

			std::string look_up_table_string = weak_classifier->FirstChildElement("lut")->GetText();
			std::vector<std::string> lutv = split(look_up_table_string, " ");

			double *p_regression_value = tensor_regression_value.mutable_cpu_data();
			for (int tmp_k = 0; tmp_k <lutv.size(); tmp_k++)
			{
				// iterate LUT 
				p_regression_value[weakIndex * tensor_regression_value.channels() + tmp_k] = atof(lutv[tmp_k].c_str());
			}
			weakIndex++;
		}
		tmp_i++;
	}
}


void InternalLonginusCascade::LoadCascade(LonginusCascadeType cascadeType, int device)
{
	if (device >= 0)
	{
#ifndef USE_CUDA
		NO_GPU;
#endif
	}

	const HardCodeRomanciaCascade *cascade = nullptr;
	switch (cascadeType)
	{
	case FRONTAL:
		cascade = &FrontalCascade;
		break;
	case LEFT_PROFILE:
		cascade = &LeftProfileCascade;
		break;
	case RIGHT_PROFILE:
		cascade = &RightProfileCascade;
		break;
	case FRONTAL_REINFORCE:
		cascade = &FrontalReinforceCascade;
		break;
	case LEFT_PROFILE_REINFORCE:
		cascade = &LeftProfileReinforceCascade;
		break;
	case RIGHT_PROFILE_REINFORCE:
		cascade = &RightProfileReinforceCascade;
		break;
	default:
		break;
	}
	device_ = device;
	win_width = cascade->win_width;
	win_height = cascade->win_height;
	face_width = cascade->face_width;
	face_height = cascade->face_height;
	numStages = cascade->numStages;
	numWeaks = cascade->numWeaks;
	map_mode = cascade->lbp_mode;
	fea_mode = cascade->fea_mode;

	tensor_weak_num_in_stages = tensor<int>(numStages, device_);
	tensor_stage_threshold = tensor<double>(numStages, device_);
	tensor_stage_far = tensor<double>(numStages, device_);
	tensor_fea_index = tensor<int>(numWeaks, device_);
	tensor_fea_info = tensor<int>({ numWeaks, 8 }, device_);
	tensor_weak_threshold = tensor<double>(numWeaks, device_);
	tensor_regression_value = tensor<double>({ numWeaks, map_lut_length[map_mode] }, device_);

	tensor_pNode = tensor<const int *>({ numWeaks, 36 }, device_);
	tensor_FeaMap = tensor<unsigned char>(sizeof(LBPMAP[map_mode]), device_);

	unsigned char *p_lbpmap = tensor_FeaMap.mutable_cpu_data();
	for (size_t i = 0; i < sizeof(LBPMAP[map_mode]); i++)
		p_lbpmap[i] = LBPMAP[map_mode][i];

	for (size_t i = 0; i < numStages; i++)
	{
		tensor_weak_num_in_stages.mutable_cpu_data()[i] = cascade->weaksPerStage[i];
		tensor_stage_threshold.mutable_cpu_data()[i] = cascade->stageThresholdPerStage[i];
		tensor_stage_far.mutable_cpu_data()[i] = cascade->falseAlarmPerStage[i];
	}

	double *p_weak_threshold = tensor_weak_threshold.mutable_cpu_data();
	int *p_fea_index = tensor_fea_index.mutable_cpu_data();
	int *p_fea_info = tensor_fea_info.mutable_cpu_data();
	double *p_regression_value = tensor_regression_value.mutable_cpu_data();
	int lut_len = map_lut_length[map_mode];
	for (size_t i = 0; i < numWeaks; i++)
	{
		p_weak_threshold[i] = cascade->p_weaks[i].weakThreshold;
		p_fea_index[i] = cascade->p_weaks[i].featureIndex;

		p_fea_info[i * 8 + 0] = cascade->p_weaks[i].feaInfo[0];
		p_fea_info[i * 8 + 1] = cascade->p_weaks[i].feaInfo[1];
		p_fea_info[i * 8 + 2] = cascade->p_weaks[i].feaInfo[2];
		p_fea_info[i * 8 + 3] = cascade->p_weaks[i].feaInfo[3];
		p_fea_info[i * 8 + 4] = cascade->p_weaks[i].feaInfo[4];
		p_fea_info[i * 8 + 5] = cascade->p_weaks[i].feaInfo[5];
		p_fea_info[i * 8 + 6] = cascade->p_weaks[i].feaInfo[6];
		p_fea_info[i * 8 + 7] = cascade->p_weaks[i].feaInfo[7];

		for (size_t j = 0; j < lut_len; j++)
			p_regression_value[j] = cascade->p_weaks[i].regression_value[j];

		p_regression_value += lut_len;
	}
}

#endif

// pCascade: the classifier
// pSum: the integral image
// sum_width: width of pSum
// sum_height: height of pSum
extern void UpdateCascade(const int *p_fea_info, int numStages, const int *p_weak_num_in_stages, const int **pNode, const int *pSum, int sum_width)
{
	int weakIndex = 0;
	for (int i = 0; i < numStages; i++)
	{
		for (int j = 0; j < p_weak_num_in_stages[i]; j++)
		{
			int x = p_fea_info[weakIndex * 8 + 0];
			int y = p_fea_info[weakIndex * 8 + 1];
			int cw = p_fea_info[weakIndex * 8 + 4];
			int ch = p_fea_info[weakIndex * 8 + 5];
			int cstepx = p_fea_info[weakIndex * 8 + 6];
			int cstepy = p_fea_info[weakIndex * 8 + 7];

			int offset = weakIndex * 36;

			pNode[offset + 0] = pSum + y * sum_width + (x);
			pNode[offset + 1] = pSum + y * sum_width + (x + cw);
			pNode[offset + 2] = pSum + y * sum_width + (x + cstepx);
			pNode[offset + 3] = pSum + y * sum_width + (x + cstepx + cw);
			pNode[offset + 4] = pSum + y * sum_width + (x + 2 * cstepx);
			pNode[offset + 5] = pSum + y * sum_width + (x + 2 * cstepx + cw);

			pNode[offset + 6] = pSum + (y + ch) * sum_width + (x);
			pNode[offset + 7] = pSum + (y + ch) * sum_width + (x + cw);
			pNode[offset + 8] = pSum + (y + ch) * sum_width + (x + cstepx);
			pNode[offset + 9] = pSum + (y + ch) * sum_width + (x + cstepx + cw);
			pNode[offset + 10] = pSum + (y + ch) * sum_width + (x + 2 * cstepx);
			pNode[offset + 11] = pSum + (y + ch) * sum_width + (x + 2 * cstepx + cw);

			pNode[offset + 12] = pSum + (y + cstepy) * sum_width + (x);
			pNode[offset + 13] = pSum + (y + cstepy) * sum_width + (x + cw);
			pNode[offset + 14] = pSum + (y + cstepy) * sum_width + (x + cstepx);
			pNode[offset + 15] = pSum + (y + cstepy) * sum_width + (x + cstepx + cw);
			pNode[offset + 16] = pSum + (y + cstepy) * sum_width + (x + 2 * cstepx);
			pNode[offset + 17] = pSum + (y + cstepy) * sum_width + (x + 2 * cstepx + cw);

			pNode[offset + 18] = pSum + (y + cstepy + ch) * sum_width + (x);
			pNode[offset + 19] = pSum + (y + cstepy + ch) * sum_width + (x + cw);
			pNode[offset + 20] = pSum + (y + cstepy + ch) * sum_width + (x + cstepx);
			pNode[offset + 21] = pSum + (y + cstepy + ch) * sum_width + (x + cstepx + cw);
			pNode[offset + 22] = pSum + (y + cstepy + ch) * sum_width + (x + 2 * cstepx);
			pNode[offset + 23] = pSum + (y + cstepy + ch) * sum_width + (x + 2 * cstepx + cw);

			pNode[offset + 24] = pSum + (y + 2 * cstepy) * sum_width + (x);
			pNode[offset + 25] = pSum + (y + 2 * cstepy) * sum_width + (x + cw);
			pNode[offset + 26] = pSum + (y + 2 * cstepy) * sum_width + (x + cstepx);
			pNode[offset + 27] = pSum + (y + 2 * cstepy) * sum_width + (x + cstepx + cw);
			pNode[offset + 28] = pSum + (y + 2 * cstepy) * sum_width + (x + 2 * cstepx);
			pNode[offset + 29] = pSum + (y + 2 * cstepy) * sum_width + (x + 2 * cstepx + cw);

			pNode[offset + 30] = pSum + (y + 2 * cstepy + ch) * sum_width + (x);
			pNode[offset + 31] = pSum + (y + 2 * cstepy + ch) * sum_width + (x + cw);
			pNode[offset + 32] = pSum + (y + 2 * cstepy + ch) * sum_width + (x + cstepx);
			pNode[offset + 33] = pSum + (y + 2 * cstepy + ch) * sum_width + (x + cstepx + cw);
			pNode[offset + 34] = pSum + (y + 2 * cstepy + ch) * sum_width + (x + 2 * cstepx);
			pNode[offset + 35] = pSum + (y + 2 * cstepy + ch) * sum_width + (x + 2 * cstepx + cw);

			weakIndex++;
		}
	}

	return;
}

static inline double DetectAt(const int * const *pNode, int offset, const double *regression_value, int lut_len, const double *stage_threshold, const int * weak_num_in_stages,
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


void InternalLonginusCascade::scan_cpu(glasssix::excalibur::tensor<int>& I, std::vector<CandidateRect> &rects,
	int xstep, int ystep, int xmax, int ymax, int sum_width, int factor1024x, int lut_len, bool doEarlyReject)
{
	UpdateCascade(tensor_fea_info.cpu_data(), numStages, tensor_weak_num_in_stages.cpu_data(), tensor_pNode.mutable_cpu_data(), I.cpu_data(), I.width());

	const int * const * pNode = tensor_pNode.cpu_data();
	const double *regression_value = tensor_regression_value.cpu_data();
	const double *stage_threshold = tensor_stage_threshold.cpu_data();
	const int * weak_num_in_stages = tensor_weak_num_in_stages.cpu_data();
	const double *weak_threshold = tensor_weak_threshold.cpu_data();
	const unsigned char *pLBPMAP = tensor_FeaMap.cpu_data();

	for (int iy = 0; iy < ymax; iy += ystep)
	{
		for (int ix = 0; ix < xmax; ix += xstep)
		{
			int w_offset = iy * sum_width + ix;
			double confidence = DetectAt(pNode, w_offset, regression_value, lut_len, stage_threshold,
				weak_num_in_stages, weak_threshold, pLBPMAP, numStages, doEarlyReject);
			if (confidence > 0)
			{
				CandidateRect fr;

				fr.ix = ix;
				fr.iy = iy;

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
			else if (confidence < -4)
			{
				std::vector<Point> points;
				extendWin(ix, iy, xstep, ystep, xmax, ymax, points);
				for (size_t i = 0; i < points.size(); i++)
				{
					int w_offset = points[i].y * sum_width + points[i].x;
					double confidence2 = DetectAt(pNode, w_offset, regression_value, lut_len, stage_threshold,
						weak_num_in_stages, weak_threshold, pLBPMAP, numStages, doEarlyReject);
					if (confidence2 > 0)
					{
						CandidateRect fr;

						fr.ix = points[i].x;
						fr.iy = points[i].y;

						fr.x = (fr.ix * factor1024x + 512) >> 10;
						fr.y = (fr.iy * factor1024x + 512) >> 10;
						fr.width = (win_width * factor1024x + 512) >> 10;
						fr.height = (win_height * factor1024x + 512) >> 10;

						fr.confidence = confidence2;

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
}

void InternalLonginusCascade::scan_cpu_multi_threads(glasssix::excalibur::tensor<int>& I, std::vector<CandidateRect> &rects,
	int xstep, int ystep, int xmax, int ymax, int sum_width, int factor1024x, int lut_len, bool doEarlyReject)
{
	UpdateCascade(tensor_fea_info.cpu_data(), numStages, tensor_weak_num_in_stages.cpu_data(), tensor_pNode.mutable_cpu_data(), I.cpu_data(), I.width());

	const int * const * pNode = tensor_pNode.cpu_data();
	const double *regression_value = tensor_regression_value.cpu_data();
	const double *stage_threshold = tensor_stage_threshold.cpu_data();
	const int * weak_num_in_stages = tensor_weak_num_in_stages.cpu_data();
	const double *weak_threshold = tensor_weak_threshold.cpu_data();
	const unsigned char *pLBPMAP = tensor_FeaMap.cpu_data();

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int iy = 0; iy < ymax; iy += ystep)
	{
		for (int ix = 0; ix < xmax; ix += xstep)
		{
			int w_offset = iy * sum_width + ix;
			double confidence = DetectAt(pNode, w_offset, regression_value, lut_len, stage_threshold,
				weak_num_in_stages, weak_threshold, pLBPMAP, numStages, doEarlyReject);
			if (confidence > 0)
			{
				CandidateRect fr;

				fr.ix = ix;
				fr.iy = iy;

				fr.x = (fr.ix * factor1024x + 512) >> 10;
				fr.y = (fr.iy * factor1024x + 512) >> 10;
				fr.width = (win_width * factor1024x + 512) >> 10;
				fr.height = (win_height * factor1024x + 512) >> 10;

				fr.confidence = confidence;

				fr.xstep = xstep;
				fr.ystep = ystep;
				fr.xmax = xmax;
				fr.ymax = ymax;

#ifdef _OPENMP
#pragma omp critical
#endif
				{
					rects.push_back(fr);
				}
			}
			else if(confidence < -4)
			{
				std::vector<Point> points;
				extendWin(ix, iy, xstep, ystep, xmax, ymax, points);
				for (size_t i = 0; i < points.size(); i++)
				{
					int w_offset = points[i].y * sum_width + points[i].x;
					double confidence2 = DetectAt(pNode, w_offset, regression_value, lut_len, stage_threshold,
						weak_num_in_stages, weak_threshold, pLBPMAP, numStages, doEarlyReject);
					if (confidence2 > 0)
					{
						CandidateRect fr;

						fr.ix = points[i].x;
						fr.iy = points[i].y;

						fr.x = (fr.ix * factor1024x + 512) >> 10;
						fr.y = (fr.iy * factor1024x + 512) >> 10;
						fr.width = (win_width * factor1024x + 512) >> 10;
						fr.height = (win_height * factor1024x + 512) >> 10;

						fr.confidence = confidence2;

						fr.xstep = xstep;
						fr.ystep = ystep;
						fr.xmax = xmax;
						fr.ymax = ymax;
#ifdef _OPENMP
#pragma omp critical
#endif
						{
							rects.push_back(fr);
						}
					}
				}
			}
		}
	}
}

void InternalLonginusCascade::SingleScaleDetect(glasssix::excalibur::tensor<int> &Integral, int winStep, int factor1024x, std::vector<CandidateRect> &rects, bool useMultiThreads, bool doEarlyReject)
{
	int sum_width, sum_height;
	int ystep, xstep, ymax, xmax;

	if (win_width > (Integral.width() - 1) ||
		win_height > (Integral.height() - 1))
		return;

	sum_width = Integral.width();
	sum_height = Integral.height();

	ystep = winStep;
	xstep = winStep;
	// '-1' is to avoid that
	// the face rect is out of the image range caused by the round error
	ymax = sum_height - 1 - win_height - 1;
	xmax = sum_width - 1 - win_width - 1;

	if (device_ < 0)
	{
		if (useMultiThreads)
			scan_cpu_multi_threads(Integral, rects, xstep, ystep, xmax, ymax, sum_width, factor1024x, map_lut_length[map_mode], doEarlyReject);
		else
			scan_cpu(Integral, rects, xstep, ystep, xmax, ymax, sum_width, factor1024x, map_lut_length[map_mode], doEarlyReject);
	}
	else
	{
#ifdef USE_CUDA
		scan_gpu(Integral, rects, xstep, ystep, xmax, ymax, sum_width, factor1024x, map_lut_length[map_mode], doEarlyReject);
#else
		NO_GPU;
#endif
	}
}

std::vector<face_rect_basic> InternalLonginusCascade::MultiScaleDetect(glasssix::excalibur::tensor<unsigned char> &gray, int minSize, float scale, int min_neighbors, bool useMultiThreads, bool doEarlyReject)
{
	//LOG_IF(WARNING, (doEarlyReject && device_ >= 0)) << "doEarlyReject dose not work on gpu. It will not take any benifit on speed. Automatic disable.";
	LOG_IF(WARNING, (useMultiThreads && device_ >= 0)) << "useMultiThreads is invalide when working on gpu. Automatic ignore.";

	minSize = std::max(win_width, minSize);
	int maxSize = std::min(gray.width(), gray.height());

	if (maxSize < minSize)
	{
		// create a structure vector for the output data
		return std::vector<face_rect_basic>();
	}

	// containers for the detected faces
	std::vector<CandidateRect> candidateRects;
	std::vector<ScaledMatrix> integral_pyramids;

	int scale_factor1024x = scale * 1024;
	int factor1024x = ((minSize << 10) + (win_width / 2)) / win_width;
	int factor1024x_max = (maxSize << 10) / win_width; //do not round it, to avoid the scan window be out of range
	for (; factor1024x <= factor1024x_max; factor1024x = ((factor1024x*scale_factor1024x + 512) >> 10))
	{
		int dwidth = ((gray.width() << 10) + factor1024x / 2) / factor1024x;
		int dheight = ((gray.height() << 10) + factor1024x / 2) / factor1024x;
		int dstep = (((dwidth * 8 + 7) / 8) + 4 - 1) & (~(4 - 1));

		tensor<unsigned char> tensor_psmall;
		if (device_ < 0)
		{
			//tensor_psmall = tensor<unsigned char>({ 1, 1, dheight, dwidth }, device_);
			//resize_cpu_bilinear(gray.cpu_data(), gray.height(), gray.width(), gray.channels(), tensor_psmall.mutable_cpu_data(), dheight, dwidth);

			tensor_psmall = tensor<unsigned char>({ 1, 1, dheight, dstep }, device_);
			myResize(gray.cpu_data(), gray.width(), gray.height(), gray.width() * gray.channels(),
				tensor_psmall.mutable_cpu_data(), dwidth, dheight, dstep);
		}
		else
		{
			tensor_psmall = tensor<unsigned char>({ 1, 1, dheight, dwidth }, device_);
#ifdef USE_CUDA
			resize_gpu_bilinear(gray.gpu_data(), gray.height(), gray.width(), gray.channels(), tensor_psmall.mutable_gpu_data(), dheight, dwidth, device_);
#else
			NO_GPU;
#endif
		}

		int sum_width = dwidth + 1;
		int sum_height = dheight + 1;

		tensor<int> I({ 1, 1, sum_height, sum_width }, device_);
		if (device_ < 0)
		{
			//myIntegral(tensor_psmall.cpu_data(), dwidth, dheight, dwidth, I.mutable_cpu_data(), I.width());
			myIntegral(tensor_psmall.cpu_data(), dwidth, dheight, dstep, I.mutable_cpu_data(), I.width());
		}
		else
		{
#ifdef USE_CUDA
			integral_gpu(tensor_psmall.gpu_data(), tensor_psmall.width(), tensor_psmall.height(), I.mutable_gpu_data(), sum_width, sum_height);
#else
			NO_GPU;
#endif
		}

		int winStep = ((factor1024x <= 3072) + 1) * 4;
		integral_pyramids.emplace_back(factor1024x, winStep);

		std::vector<CandidateRect> tempCandidateRects;
		SingleScaleDetect(I, winStep, factor1024x, tempCandidateRects, useMultiThreads, doEarlyReject);

		for (size_t i = 0; i < tempCandidateRects.size(); i++)
		{
			tempCandidateRects[i].index_in_image_pyramids = integral_pyramids.size() - 1;
			tempCandidateRects[i].cascade = std::shared_ptr<InternalLonginusCascade>(this, [](InternalLonginusCascade *) {});
		}

		candidateRects.insert(candidateRects.end(), tempCandidateRects.begin(), tempCandidateRects.end());
	}

	std::vector<face_rect_basic> rects;
	for (size_t i = 0; i < candidateRects.size(); i++)
	{
		rects.push_back(candidateRects[i]);
	}

	GroupRects(rects, min_neighbors);

	return rects;
}

#ifndef RELEASE_SDK
#ifdef HARDCODE_TRANSFORM
void InternalLonginusCascade::HardCode2Hpp(const std::string & filename, const std::string & cascadeName)
{
	std::ofstream out(filename);
	out << "#ifndef HARDCODE_" << cascadeName << "_HPP" << std::endl
		<< "#define HARDCODE_" << cascadeName << "_HPP" << std::endl
		<< "#include \"HardCode.hpp\"" << std::endl;

	out << "namespace glasssix" << std::endl << "{" << std::endl << "\tnamespace longinus" << std::endl << "\t{" << std::endl;

	out << "\t\t" << "static const double look_up_table_" << cascadeName << "[][" << map_lut_length[map_mode] << "] = " << std::endl << "\t\t{" << std::endl;
	for (size_t i = 0; i < numWeaks; i++)
	{
		out << "\t\t\t{ ";
		for (size_t j = 0; j < map_lut_length[map_mode]; j++)
		{
			if (j != map_lut_length[map_mode] - 1)
				out << std::setprecision(16) << tensor_regression_value.cpu_data()[i * map_lut_length[map_mode] + j] << ", ";
			else
				out << std::setprecision(16) << tensor_regression_value.cpu_data()[i * map_lut_length[map_mode] + j];
		}
		if (i != numWeaks - 1)
			out << "}," << std::endl;
		else
			out << "}" << std::endl;
	}
	out << "\t\t};" << std::endl;

	out << "\t\t" << "static const int feaInfos_" << cascadeName << "[][8] = " << std::endl << "\t\t{" << std::endl;
	for (size_t i = 0; i < numWeaks; i++)
	{
		out << "\t\t\t{ ";
		for (size_t j = 0; j < 8; j++)
		{
			if (j != 7)
				out << tensor_fea_info.cpu_data()[i * 8 + j] << ", ";
			else
				out << tensor_fea_info.cpu_data()[i * 8 + j];
		}
		if (i != numWeaks - 1)
		{
			out << " }, " << std::endl;
		}
		else
		{
			out << " }" << std::endl;
		}
	}
	out << "\t\t};" << std::endl;

	out << "\t\t" << "static const HardCodeWeak " << "weaks_" << cascadeName << "[] = " << std::endl << "\t\t{" << std::endl;
	for (size_t i = 0; i < numWeaks; i++)
	{
		out << "\t\t\t{ " << "feaInfos_" << cascadeName << "[" << i << "], ";

		out << tensor_fea_index.cpu_data()[i] << ", ";
		out << std::setprecision(16) << tensor_weak_threshold.cpu_data()[i] << ", ";

		out << "look_up_table_" << cascadeName << "[" << i << "] ";

		if (i != numWeaks - 1)
			out << "}, " << std::endl;
		else
			out << "}" << std::endl;
	}
	out << "\t\t};" << std::endl;

	out << "\t\t" << "static const int weak_num_in_stages_" << cascadeName << "[] = { ";
	for (size_t i = 0; i < numStages; i++)
	{
		if (i != numStages - 1)
			out << tensor_weak_num_in_stages.cpu_data()[i] << ", ";
		else
			out << tensor_weak_num_in_stages.cpu_data()[i];
	}
	out << " };" << std::endl;

	out << "\t\t" << "static const double stages_threshold_" << cascadeName << "[] = { ";
	for (size_t i = 0; i < numStages; i++)
	{
		if (i != numStages - 1)
			out << std::setprecision(16) << tensor_stage_threshold.cpu_data()[i] << ", ";
		else
			out << std::setprecision(16) << tensor_stage_threshold.cpu_data()[i];
	}
	out << " };" << std::endl;

	out << "\t\t" << "static const double stages_far_" << cascadeName << "[] = { ";
	for (size_t i = 0; i < numStages; i++)
	{
		if (i != numStages - 1)
			out << std::setprecision(16) << tensor_stage_far.cpu_data()[i] << ", ";
		else
			out << std::setprecision(16) << tensor_stage_far.cpu_data()[i];
	}
	out << " };" << std::endl;

	out << "\t\t" << "static const HardCodeRomanciaCascade " << cascadeName << " = " << std::endl << "\t\t{" << std::endl;
	out << "\t\t\t" << win_width << ", " << win_height << ", " << face_width << ", " << face_height << ", " << numStages << ", " << numWeaks << ", " << map_mode << ", " << fea_mode << ", " << std::endl;

	out << "\t\t\t" << "weak_num_in_stages_" << cascadeName << "," << std::endl;

	out << "\t\t\t" << "stages_threshold_" << cascadeName << "," << std::endl;

	out << "\t\t\t" << "stages_far_" << cascadeName << "," << std::endl;

	out << "\t\t\t" << "weaks_" << cascadeName << std::endl;

	out << "\t\t};" << std::endl;

	

	out << "\t}" << std::endl;

	out << "}" << std::endl;

	out << "#endif" << std::endl;

	out.close();
}
#endif
#endif