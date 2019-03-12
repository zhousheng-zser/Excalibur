#ifndef INTERNALLONGINUSCASCADE_HPP
#define INTERNALLONGINUSCASCADE_HPP

#include <string>
#include <vector>
#include <memory>

#include <glasssix\tensor.hpp>
#include "common.hpp"
#include "BaseLonginusCascade.hpp"

namespace glasssix
{
	namespace longinus
	{
		class EXPORT_LONGINUS InternalLonginusCascade : public BaseLonginusCascade
		{
		public:
			InternalLonginusCascade();
#ifdef Internal_SDK
			void LoadCascade(const std::string& filename, int device = -1);
#endif
			virtual void LoadCascade(LonginusCascadeType cascadeType, int device = -1);
			void SingleScaleDetect(excalibur::tensor<int> &Integral, int winStep, int factor1024x, std::vector<CandidateRect> &rects, bool useMultiThreads = false, bool doEarlyReject = false);
			std::vector<FaceRect> MultiScaleDetect(excalibur::tensor<unsigned char> &gray, int minSize, float scale, int min_neighbors, bool useMultiThreads = false, bool doEarlyReject = false);

#ifdef Internal_SDK
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

			excalibur::tensor<int> tensor_weak_num_in_stages;
			excalibur::tensor<double> tensor_stage_threshold;
			excalibur::tensor<double> tensor_stage_far;
			excalibur::tensor<int> tensor_fea_index;
			excalibur::tensor<int> tensor_fea_info;
			excalibur::tensor<double> tensor_weak_threshold;
			excalibur::tensor<double> tensor_regression_value;
			
			excalibur::tensor<const int *> tensor_pNode;
			excalibur::tensor<unsigned char> tensor_FeaMap;

			int device_;

		private:
			void scan_cpu(excalibur::tensor<int>& I, std::vector<CandidateRect> &rects,
				int xstep, int ystep, int xmax, int ymax, int sum_width, int factor1024x, int lut_len, bool doEarlyReject = false);
			void scan_cpu_multi_threads(excalibur::tensor<int>& I, std::vector<CandidateRect> &rects,
				int xstep, int ystep, int xmax, int ymax, int sum_width, int factor1024x, int lut_len, bool doEarlyReject = false);
#ifdef USE_CUDA
			void scan_gpu(excalibur::tensor<int>& I, std::vector<CandidateRect> &rects,
				int xstep, int ystep, int xmax, int ymax, int sum_width, int factor1024x, int lut_len, bool doEarlyReject = false);
#endif
		};
	}
}

#endif

