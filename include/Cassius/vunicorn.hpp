#pragma once
#ifndef _VUNICORN_HPP_
#define _VUNICORN_HPP_
#include <vector>

namespace glasssix
{
	namespace cassius
	{
		class vUnicorn
		{
		public:

			vUnicorn() {}

			vUnicorn(int device) {}

			virtual ~vUnicorn() {}

			virtual std::vector<std::vector<float> > Forward(const float* input_data, unsigned num, int order) = 0;

			virtual std::vector<std::vector<float> > Forward(const unsigned char* input_data, unsigned num, int order) = 0;

		};
	}
}

#endif // !_VUNICORN_HPP_