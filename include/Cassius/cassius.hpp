#ifndef _CASSIUS_HPP_
#define _CASSIUS_HPP_

#include "vUnicorn.hpp"

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
		class EXPORT_CASSIUS Cassius
		{
		public:

			Cassius() {}

			Cassius(int device);

			~Cassius();

			std::vector<std::vector<float> > Forward(const float* input_data, unsigned num, int order = 0);

			std::vector<std::vector<float> > Forward(const unsigned char* input_data, unsigned num, int order = 0);

		private:
			vUnicorn *unicornia_;
		};
	}
}

#endif // !_CASSIUS_HPP_