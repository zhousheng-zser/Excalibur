#pragma once
#ifndef EXCALIBUR_UTIL_FP16_H_
#define EXCALIBUR_UTIL_FP16_H_

#include <algorithm>
#include <cfloat>
#include <iosfwd>
#include <glog/logging.h>

#ifndef CPU_ONLY
#ifdef __CUDACC__
#include "half.cuh"
#endif
#include "half.hpp"

namespace Excalibur {
	typedef half_float::half float16;
}
#else
#define EXCALIBUR_UTIL_HD
#define EXCALIBUR_UTIL_IHD inline
#endif
#endif