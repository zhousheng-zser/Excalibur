#ifndef _VDIODORUS_HPP_
#define _VDIODORUS_HPP_
#include <vector>
#include "../Longinus/LonginusDetector.hpp"

namespace glasssix
{
	namespace dionysios
	{
		class vDiodorus
		{
		public:

			vDiodorus() {}

			vDiodorus(int device) {}

			virtual ~vDiodorus() {}

			virtual bool aliveDetect(const unsigned char* srcColorVSL, std::vector<glasssix::longinus::FaceRectwithFaceInfo> face_infoVSL,
				int channels, int height, int width, int order,
				const unsigned char* srcColorNIR, std::vector<glasssix::longinus::FaceRectwithFaceInfo> face_infoNIR = std::vector<longinus::FaceRectwithFaceInfo>()) = 0;

		};
	}
}

#endif // !_VDIODORUS_HPP_