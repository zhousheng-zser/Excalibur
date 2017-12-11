#pragma once
#ifndef _MATH_HELPER_HPP_
#define _MATH_HELPER_HPP_

#ifdef USE_SSE
#include <immintrin.h>
#endif
#include <cstdint>

namespace excalibur
{
	class MathHelper
	{
	public:
		static inline void UInt8ToInt32CPU(const uint8_t* src, int32_t* dest,
			int32_t len) {
			for (int32_t i = 0; i < len; i++)
				*(dest++) = static_cast<int32_t>(*(src++));
		}

		static inline void VectorAddCPU(const int32_t* x, const int32_t* y, int32_t* z,
			int32_t len) {
			int32_t i;
#ifdef USE_SSE
			__m128i x1;
			__m128i y1;
			const __m128i* x2 = reinterpret_cast<const __m128i*>(x);
			const __m128i* y2 = reinterpret_cast<const __m128i*>(y);
			__m128i* z2 = reinterpret_cast<__m128i*>(z);

			for (i = 0; i < len - 4; i += 4) {
				x1 = _mm_loadu_si128(x2++);
				y1 = _mm_loadu_si128(y2++);
				_mm_storeu_si128(z2++, _mm_add_epi32(x1, y1));
			}
			for (; i < len; i++)
				*(z + i) = (*(x + i)) + (*(y + i));
#else
			for (i = 0; i < len; i++)
				*(z + i) = (*(x + i)) + (*(y + i));
#endif
		}

		static inline void VectorSubCPU(const int32_t* x, const int32_t* y, int32_t* z,
			int32_t len) {
			int32_t i;
#ifdef USE_SSE
			__m128i x1;
			__m128i y1;
			const __m128i* x2 = reinterpret_cast<const __m128i*>(x);
			const __m128i* y2 = reinterpret_cast<const __m128i*>(y);
			__m128i* z2 = reinterpret_cast<__m128i*>(z);

			for (i = 0; i < len - 4; i += 4) {
				x1 = _mm_loadu_si128(x2++);
				y1 = _mm_loadu_si128(y2++);

				_mm_storeu_si128(z2++, _mm_sub_epi32(x1, y1));
			}
			for (; i < len; i++)
				*(z + i) = (*(x + i)) - (*(y + i));
#else
			for (i = 0; i < len; i++)
				*(z + i) = (*(x + i)) - (*(y + i));
#endif
		}

		static inline void VectorAbsCPU(const int32_t* src, int32_t* dest, int32_t len) {
			int32_t i;
#ifdef USE_SSE
			__m128i val;
			__m128i val_abs;
			const __m128i* x = reinterpret_cast<const __m128i*>(src);
			__m128i* y = reinterpret_cast<__m128i*>(dest);

			for (i = 0; i < len - 4; i += 4) {
				val = _mm_loadu_si128(x++);
				val_abs = _mm_abs_epi32(val);
				_mm_storeu_si128(y++, val_abs);
			}
			for (; i < len; i++)
				dest[i] = (src[i] >= 0 ? src[i] : -src[i]);
#else
			for (i = 0; i < len; i++)
				dest[i] = (src[i] >= 0 ? src[i] : -src[i]);
#endif
		}

		static inline void SquareCPU(const int32_t* src, uint32_t* dest, int32_t len) {
			int32_t i;
#ifdef USE_SSE
			__m128i x1;
			const __m128i* x2 = reinterpret_cast<const __m128i*>(src);
			__m128i* y2 = reinterpret_cast<__m128i*>(dest);

			for (i = 0; i < len - 4; i += 4) {
				x1 = _mm_loadu_si128(x2++);
				_mm_storeu_si128(y2++, _mm_mullo_epi32(x1, x1));
			}
			for (; i < len; i++)
				*(dest + i) = (*(src + i)) * (*(src + i));
#else
			for (i = 0; i < len; i++)
				*(dest + i) = (*(src + i)) * (*(src + i));
#endif
		}

		static inline float VectorInnerProductCPU(const float* x, const float* y,
			int32_t len) {
			float prod = 0;
			int32_t i;
#ifdef USE_SSE
			__m128 x1;
			__m128 y1;
			__m128 z1 = _mm_setzero_ps();
			float buf[4];

			for (i = 0; i < len - 4; i += 4) {
				x1 = _mm_loadu_ps(x + i);
				y1 = _mm_loadu_ps(y + i);
				z1 = _mm_add_ps(z1, _mm_mul_ps(x1, y1));
			}
			_mm_storeu_ps(&buf[0], z1);
			prod = buf[0] + buf[1] + buf[2] + buf[3];
			for (; i < len; i++)
				prod += x[i] * y[i];
#else
			for (i = 0; i < len; i++)
				prod += x[i] * y[i];
#endif
			return prod;
		}
	};
}

#endif