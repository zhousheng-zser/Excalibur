#ifndef INTERNALROMANCIACASCADE_HPP
#define INTERNALROMANCIACASCADE_HPP

#include <string>
#include <vector>
#include <memory>

#include <glasssix/tensor.hpp>
#include "common.hpp"
#include "BaseRomanciaCascade.hpp"

namespace glasssix
{
	namespace longinus
	{
		class ROMANCIA_LIB InternalRomanciaCascade : public BaseRomanciaCascade
		{
		public:
			InternalRomanciaCascade();
#ifdef Internal_SDK
			void LoadCascade(const std::string& filename, int device = -1);
#endif
			virtual void LoadCascade(RomanciaCascadeType cascadeType, int device = -1);
			void SingleScaleDetect(tensor<int> &Integral, int winStep, int factor1024x, std::vector<CandidateRect> &rects, bool useMultiThreads = false, bool doEarlyReject = false);
			std::vector<FaceRect> MultiScaleDetect(tensor<unsigned char> &gray, int minSize, float scale, int min_neighbors, bool useMultiThreads = false, bool doEarlyReject = false);

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

			tensor<int> tensor_weak_num_in_stages;
			tensor<double> tensor_stage_threshold;
			tensor<double> tensor_stage_far;
			tensor<int> tensor_fea_index;
			tensor<int> tensor_fea_info;
			tensor<double> tensor_weak_threshold;
			tensor<double> tensor_regression_value;
			
			tensor<const int *> tensor_pNode;
			tensor<unsigned char> tensor_FeaMap;

			int device_;

		private:
			void scan_cpu(glasssix::tensor<int>& I, std::vector<CandidateRect> &rects,
				int xstep, int ystep, int xmax, int ymax, int sum_width, int factor1024x, int lut_len, bool doEarlyReject = false);
			void scan_cpu_multi_threads(glasssix::tensor<int>& I, std::vector<CandidateRect> &rects,
				int xstep, int ystep, int xmax, int ymax, int sum_width, int factor1024x, int lut_len, bool doEarlyReject = false);
			void scan_gpu(glasssix::tensor<int>& I, std::vector<CandidateRect> &rects,
				int xstep, int ystep, int xmax, int ymax, int sum_width, int factor1024x, int lut_len, bool doEarlyReject = false);
		};
	}
}

#endif

