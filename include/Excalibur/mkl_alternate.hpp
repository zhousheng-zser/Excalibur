#pragma once
#ifndef _MKL_ALTERNATE_HPP_
#define _MKL_ALTERNATE_HPP_
#include  <glasssix/accelerator.hpp>
#ifdef USE_MKL

#include <mkl.h>

#ifdef USE_MKLDNN

//#include <mkl_dnn.h>

#endif
#else  // If use MKL, simply include the MKL header

#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#elif USE_OPENBLAS
extern "C" {
#include <cblas.h>
}
#else // USE_JULIUSBLAS
#include "../Julius/julius.hpp"
#endif  
#include <math.h>

// A simple way to define the vsl unary functions. The operation should
// be in the form e.g. y[i] = sqrt(a[i])
#ifndef DEFINE_VSL_UNARY_FUNC(name, operation)

#define DEFINE_VSL_UNARY_FUNC(name, operation) \
  template<typename Dtype> \
  void v##name(const int n, const Dtype* a, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(y); \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int n, const float* a, float* y) { \
    v##name<float>(n, a, y); \
  }

DEFINE_VSL_UNARY_FUNC(Sqr, y[i] = a[i] * a[i]);
DEFINE_VSL_UNARY_FUNC(Exp, y[i] = exp(a[i]));
DEFINE_VSL_UNARY_FUNC(Ln, y[i] = log(a[i]));
DEFINE_VSL_UNARY_FUNC(Abs, y[i] = fabs(a[i]));

#endif // !DEFINE_VSL_UNARY_FUNC(name, operation)

// A simple way to define the vsl unary functions with singular parameter b.
// The operation should be in the form e.g. y[i] = pow(a[i], b)
#ifndef DEFINE_VSL_UNARY_FUNC_WITH_PARAM(name, operation)

#define DEFINE_VSL_UNARY_FUNC_WITH_PARAM(name, operation) \
  template<typename Dtype> \
  void v##name(const int n, const Dtype* a, const Dtype b, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(y); \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int n, const float* a, const float b, float* y) { \
    v##name<float>(n, a, b, y); \
  }

DEFINE_VSL_UNARY_FUNC_WITH_PARAM(Powx, y[i] = pow(a[i], b));

#endif // !DEFINE_VSL_UNARY_FUNC_WITH_PARAM(name, operation)



// A simple way to define the vsl binary functions. The operation should
// be in the form e.g. y[i] = a[i] + b[i]
#ifndef DEFINE_VSL_BINARY_FUNC(name, operation)

#define DEFINE_VSL_BINARY_FUNC(name, operation) \
  template<typename Dtype> \
  void v##name(const int n, const Dtype* a, const Dtype* b, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(a); CHECK(b); CHECK(y); \
    for (int i = 0; i < n; ++i) { operation; } \
  } \
  inline void vs##name( \
    const int n, const float* a, const float* b, float* y) { \
    v##name<float>(n, a, b, y); \
  }

DEFINE_VSL_BINARY_FUNC(Add, y[i] = a[i] + b[i]);
DEFINE_VSL_BINARY_FUNC(Sub, y[i] = a[i] - b[i]);
DEFINE_VSL_BINARY_FUNC(Mul, y[i] = a[i] * b[i]);
DEFINE_VSL_BINARY_FUNC(Div, y[i] = a[i] / b[i]);

#endif // !DEFINE_VSL_BINARY_FUNC(name, operation)

#ifdef USE_OPENBLAS
// In addition, MKL comes with an additional function axpby that is not present
// in standard blas. We will simply use a two-step (inefficient, of course) way
// to mimic that.
inline void cblas_saxpby(const int N, const float alpha, const float* X,
	const int incX, const float beta, float* Y,
	const int incY) {
	cblas_sscal(N, beta, Y, incY);
	cblas_saxpy(N, alpha, X, incX, Y, incY);
}
#endif// !USE_OPENBLAS
#endif // USE_MKL
#endif