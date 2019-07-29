#ifndef _DIONYSIOS_HPP_
#define _DIONYSIOS_HPP_

#include "vDiodorus.hpp"

#ifdef EXPORT_DIONYSIOS
#undef EXPORT_DIONYSIOS
#ifdef _MSC_VER // For Windows
#ifdef _WINDLL // Dynamic lib
#define EXPORT_DIONYSIOS __declspec(dllexport)
#else // Static lib
#define EXPORT_DIONYSIOS
#endif // !_WINDLL
#elif defined(__linux__) // For Linux
#define EXPORT_DIONYSIOS
#endif
#else
#ifdef _MSC_VER
#define EXPORT_DIONYSIOS __declspec(dllimport)
#elif defined(__linux__)
#define EXPORT_DIONYSIOS
#endif
#endif

namespace glasssix
{
	namespace dionysios
	{
		class EXPORT_DIONYSIOS DionysiosDetector
		{
		public:

			DionysiosDetector() {}

			DionysiosDetector(int device);

			~DionysiosDetector();

			bool aliveDetect(const unsigned char* srcColorVSL, std::vector<glasssix::longinus::FaceRectwithFaceInfo> face_infoVSL,
				int channels, int height, int width, int order,
				const unsigned char* srcColorNIR, std::vector<glasssix::longinus::FaceRectwithFaceInfo> face_infoNIR = std::vector<longinus::FaceRectwithFaceInfo>());

			static std::string getVersion();

		private:
			vDiodorus *dionysiosia;
		};
	}
}

#endif // !_DIONYSIOS_HPP_