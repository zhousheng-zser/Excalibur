#ifndef _ROMANCIADETECTOR_HPP_
#define _ROMANCIADETECTOR_HPP_

#include <vector>
#include <string>
#include <memory>
#include "BaseLonginusCascade.hpp"
#include "../Romancia/romancia.hpp"
#include "../Damocles/damocles.hpp"
#include "matcher.hpp"

namespace glasssix
{
	namespace longinus
	{
		typedef enum DetectionType
		{
			FRONTALVIEW,
			FRONTALVIEW_REINFORCE,
			MULTIVIEW,
			MULTIVIEW_REINFORCE
		}DetectionType;
		struct FaceRectwithFaceInfo : public FaceRect
		{
			Point2f pts[5];
			float yaw;
			float pitch;
			float roll;
			//float prob;

			FaceRectwithFaceInfo() {}
			FaceRectwithFaceInfo(const FaceRect& rect)
			{
				*dynamic_cast<FaceRect *>(this) = rect;
			}

			FaceRectwithFaceInfo &operator = (const FaceRect& rect)
			{
				*dynamic_cast<FaceRect *>(this) = rect;
				return *this;
			}
		};

		class LonginusDetector
		{
		public:
			LonginusDetector(int device = -1);
			virtual ~LonginusDetector();
			std::vector<FaceRect> detect(unsigned char *gray, int width, int height, int step, int minSize, float scale,
				int minNeighbors, bool useMultiThreads = false, bool doEarlyReject = false);
			std::vector<FaceRectwithFaceInfo> detect(unsigned char *gray, int width, int height, int step, int minSize, float scale,
				int minNeighbors, int order = 0, bool useMultiThreads = false, bool doEarlyReject = false);

			std::vector<FaceRectwithFaceInfo> detectEx(const unsigned char* image, const int channels, const int height, const int width,
				const int minSize, const float* threshold, const float factor, const int stage) const;

			std::vector<Match_Retval> match(std::vector<FaceRect> &faceRect, const int frame_extract_frequency) const;

			std::vector<Match_Retval> match(std::vector<FaceRectwithFaceInfo> &faceRect, const int frame_extract_frequency) const;

			std::vector<unsigned char> alignFace(const unsigned char* ori_image, int n, int channels, int height, int width, 
				std::vector<std::vector<int>> bbox, std::vector<std::vector<int> >landmarks) const;


#ifdef Internal_SDK
			void load(std::vector<std::string> cascades, int device = -1);
#endif
			void set(DetectionType detectionType, int device = -1);


		private:
			int device_;
			std::vector<std::shared_ptr<BaseLonginusCascade>> *cascades_;
			std::unique_ptr<vBanshee> bansheelia_;
			std::unique_ptr<vDamocles> diodorus_;
			std::vector<unsigned char> data_;
			std::unique_ptr<Matcher> matcher_;
		};
	}
}

#endif// !_ROMANCIADETECTOR_HPP_