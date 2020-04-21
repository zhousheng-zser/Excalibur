#pragma once

#ifndef INTERNALLONGINUSCASCADE_HPP
#define INTERNALLONGINUSCASCADE_HPP

#include "common.hpp"
#include "BaseLonginusCascade.hpp"
#include "Primitives/tensor.hpp"

#include <string>
#include <vector>
#include <memory>

namespace glasssix
{
	namespace longinus
	{
		class EXPORT_LONGINUS InternalLonginusCascade : public BaseLonginusCascade
		{
		public:
			InternalLonginusCascade();
#ifdef TRIAL
			void LoadCascade(const std::string& filename, int device = -1);
			virtual void LoadCascade(LonginusCascadeType cascadeType, int device = -1);
#endif
			void SingleScaleDetect(memory::tensor<int> &Integral, int winStep, int factor1024x, std::vector<CandidateRect> &rects, bool useMultiThreads = false, bool doEarlyReject = false);
			std::vector<face_rect_basic> MultiScaleDetect(memory::tensor<unsigned char> &gray, int minSize, float scale, int min_neighbors, bool useMultiThreads = false, bool doEarlyReject = false);

#ifndef RELEASE_SDK
#ifdef HARDCODE_TRANSFORM
			void HardCode2Hpp(const std::string &filename, const std::string &modelName);
#endif
#endif
			virtual int getWinWidth() const { return win_width; }
			virtual int getWinHeight() const { return win_height; }
			virtual bool isEmpty() { return (numStages == 0); }

		private:
			int numStages;
			int numWeaks;
			int win_width;
			int win_height;
			int face_width;
			int face_height;
			int map_mode;
			int fea_mode;

			memory::tensor<int> tensor_weak_num_in_stages;
			memory::tensor<double> tensor_stage_threshold;
			memory::tensor<double> tensor_stage_far;
			memory::tensor<int> tensor_fea_index;
			memory::tensor<int> tensor_fea_info;
			memory::tensor<double> tensor_weak_threshold;
			memory::tensor<double> tensor_regression_value;
			
			memory::tensor<const int *> tensor_pNode;
			memory::tensor<unsigned char> tensor_FeaMap;

			int device_;

		private:
			void scan_cpu(memory::tensor<int>& I, std::vector<CandidateRect> &rects,
				int xstep, int ystep, int xmax, int ymax, int sum_width, int factor1024x, int lut_len, bool doEarlyReject = false);
			void scan_cpu_multi_threads(memory::tensor<int>& I, std::vector<CandidateRect> &rects,
				int xstep, int ystep, int xmax, int ymax, int sum_width, int factor1024x, int lut_len, bool doEarlyReject = false);
#ifdef USE_CUDA
			void scan_gpu(memory::tensor<int>& I, std::vector<CandidateRect> &rects,
				int xstep, int ystep, int xmax, int ymax, int sum_width, int factor1024x, int lut_len, bool doEarlyReject = false);
#endif
		};
	}
}

#endif
