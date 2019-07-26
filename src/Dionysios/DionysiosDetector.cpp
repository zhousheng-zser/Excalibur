#include "Diodorus.hpp"
#include "DionysiosDetector.hpp"

namespace glasssix
{
	namespace dionysios
	{
		DionysiosDetector::DionysiosDetector(int device)
		{
			dionysiosia = new Diodorus(device);
		}

		DionysiosDetector::~DionysiosDetector()
		{
			delete dionysiosia;
		}

		bool DionysiosDetector::aliveDetect(const unsigned char* srcColorVSL, std::vector<glasssix::longinus::FaceRectwithFaceInfo> face_infoVSL,
			int channels, int height, int width, int order,
			const unsigned char* srcColorNIR, std::vector<glasssix::longinus::FaceRectwithFaceInfo> face_infoNIR)
		{
			return dionysiosia->aliveDetect(srcColorVSL, face_infoVSL, channels, height, width, order, srcColorNIR, face_infoNIR);
		}

		std::string DionysiosDetector::getVersion()
		{
#ifdef TRIAL
			return std::string("Glasssix Trial FaceSDK");
#else
			return std::string("Glasssix");
#endif // TRIAL	
		}
	}
}