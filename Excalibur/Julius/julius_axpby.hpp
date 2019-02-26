#pragma once
#ifndef _JULIUS_AXPBY_HPP_
#define _JULIUS_AXPBY_HPP_
#include "simd_helper.hpp"

namespace glasssix
{
	namespace excalibur
	{
		namespace juliusblas
		{
			void cblas_saxpby(const int N, const float alpha, const float* X,
				const int incX, const float beta, float* Y, const int incY);
		}
	}
}
#endif // !_JULIUS_AXPBY_HPP_
