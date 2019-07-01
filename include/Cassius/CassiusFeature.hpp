#ifndef _CASSIUS_FEATURE_HPP_
#define _CASSIUS_FEATURE_HPP_


#ifdef EXPORT_CASSIUS
#undef EXPORT_CASSIUS
#ifdef _MSC_VER // For Windows
#ifdef _WINDLL // Dynamic lib
#define EXPORT_CASSIUS __declspec(dllexport)
#else // Static lib
#define EXPORT_CASSIUS
#endif // !_WINDLL
#elif defined(__linux__) // For Linux
#define EXPORT_CASSIUS
#endif
#else
#ifdef _MSC_VER
#define EXPORT_CASSIUS __declspec(dllimport)
#elif defined(__linux__)
#define EXPORT_CASSIUS
#endif
#endif

#include "vunicorn.hpp"
#include <string>

namespace glasssix
{
	namespace cassius
	{
		class EXPORT_CASSIUS CassiusFeature
		{
		public:

			CassiusFeature() {}

			CassiusFeature(int device);

			~CassiusFeature();

			std::vector<std::vector<float> > Forward(const unsigned char* input_data, unsigned num, int order = 0) const;

			static std::string getVersion();

		private:

			vUnicorn* unicornia_;

			std::vector<std::vector<float> > Forward(const float* input_data, unsigned num, int order = 0) const;
		};
	}
}

#endif // !_CASSIUS_FEATURE_HPP_