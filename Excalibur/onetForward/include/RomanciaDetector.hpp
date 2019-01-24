#ifndef ROMANCIADETECTOR_HPP
#define ROMANCIADETECTOR_HPP

#include <vector>
#include <string>
#include <memory>
#include "BaseRomanciaCascade.hpp"

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
		class RomanciaDetector
		{
		public:
			RomanciaDetector();
			std::vector<FaceRect> detect(unsigned char *gray, int width, int height, int minSize, float scale, int minNeighbors, bool useMultiThreads = false, bool doEarlyReject = false);
#ifdef Internal_SDK
			void load(std::vector<std::string> cascades, int device = -1);
#endif
			void set(DetectionType detectionType, int device = -1);
		private:
			int device_;
			std::vector<std::shared_ptr<BaseRomanciaCascade>> cascades_;
		};
	}
}

#endif