#ifndef ROMANCIADETECTOR_HPP
#define ROMANCIADETECTOR_HPP

#include <vector>
#include <string>
#include <memory>
#include "BaseLonginusCascade.hpp"
#include "../Romancia/include/Romancia.hpp"
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
		struct FaceRectWithLandmark : public FaceRect
		{
			Point2f pts[5];
			float yaw;
			float pitch;
			float roll;
			float prob;

			FaceRectWithLandmark() {}
			FaceRectWithLandmark(const FaceRect& rect)
			{
				*dynamic_cast<FaceRect *>(this) = rect;
			}

			FaceRectWithLandmark &operator = (const FaceRect& rect)
			{
				*dynamic_cast<FaceRect *>(this) = rect;
				return *this;
			}
		};

		class LonginusDetector
		{
		public:
			LonginusDetector();
			virtual ~LonginusDetector();
			std::vector<FaceRect> detect(unsigned char *gray, int width, int height, int step, int minSize, float scale,
				int minNeighbors, bool useMultiThreads = false, bool doEarlyReject = false);
			std::vector<FaceRectWithLandmark> detectWithLandmark(unsigned char *gray, int width, int height, int step, int minSize, float scale,
				int minNeighbors, bool useMultiThreads = false, bool doEarlyReject = false, int order = 0);

			std::vector<Match_Retval> match(std::vector<FaceRect> &faceRect, const int frame_extract_frequency);

			std::vector<unsigned char> alignFace(const unsigned char* ori_image, int n, int channels, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int> >landmarks);


#ifdef Internal_SDK
			void load(std::vector<std::string> cascades, int device = -1);
#endif
			void set(DetectionType detectionType, int device = -1);


		private:
			int device_;
			std::vector<std::shared_ptr<BaseLonginusCascade>> *cascades_;
			std::unique_ptr<BaseNet> landmarkNet_;
			std::vector<unsigned char> data_;
			std::unique_ptr<Matcher> matcher_;
		};
	}
}

#endif