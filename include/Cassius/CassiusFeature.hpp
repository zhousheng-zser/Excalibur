#ifndef _CASSIUS_HPP_
#define _CASSIUS_HPP_

#include "vunicorn.hpp"

#ifdef EXPORT_CASSIUS
#undef EXPORT_CASSIUS
#ifdef _MSC_VER
#define EXPORT_CASSIUS __declspec(dllexport)
#else
#define EXPORT_CASSIUS
#endif
#else
#ifdef _MSC_VER
#define EXPORT_CASSIUS __declspec(dllimport)
#else
#define EXPORT_CASSIUS
#endif
#endif

namespace glasssix
{
	namespace cassius
	{
		class EXPORT_CASSIUS CassiusFeature
		{
		public:

			CassiusFeature();

			CassiusFeature(int device);

			~CassiusFeature();

			std::vector<std::vector<float> > Forward(const float* input_data, unsigned num, int order = 0) const ;

			std::vector<std::vector<float> > Forward(const unsigned char* input_data, unsigned num, int order = 0) const;

		private:
			vUnicorn *unicornia_;
		};
	}
}

#endif // !_CASSIUS_HPP_