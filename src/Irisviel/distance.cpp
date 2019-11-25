#include "distance.hpp"
#include "../../include/Julius/simd_helper.hpp"

#include <cmath>

#ifdef __linux__
#define __cpuid(out, infoType)\
	asm("cpuid": "=a" (out[0]), "=b" (out[1]), "=c" (out[2]), "=d" (out[3]): "a" (infoType));
#endif

using namespace std;

namespace glasssix 
{
	namespace irisviel 
	{
		float DistanceL2::compare(const float* a, const float* b, unsigned size)
		{
			float result = 0;
#if SIMD_TYPE >= SIMDTYPE_AVX
#define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
			tmp1 = _mm256_loadu_ps(addr1);\
			tmp2 = _mm256_loadu_ps(addr2);\
			tmp1 = _mm256_sub_ps(tmp1, tmp2); \
			dest = mm_fmadd_ps(tmp1, tmp1, dest);

			__m256 sum = mm_setzero_ps();
			__m256 l0, l1, l2, l3;
			__m256 r0, r1, r2, r3;
			unsigned D = (size + 7) & ~7U; // # dim aligned up to 256 bits, or 8 floats
			unsigned DR = D % 32;
			unsigned DD = D - DR;
			const float *l = a;
			const float *r = b;
			const float *e_l = l + DD;
			const float *e_r = r + DD;
			switch (DR)
			{
			case 24:
				AVX_L2SQR(e_l + 16, e_r + 16, sum, l2, r2);
			case 16:
				AVX_L2SQR(e_l + 8, e_r + 8, sum, l1, r1);
			case 8:
				AVX_L2SQR(e_l, e_r, sum, l0, r0);
			}

			for (unsigned i = 0; i < DD; i += 32, l += 32, r += 32)
			{
				AVX_L2SQR(l, r, sum, l0, r0);
				AVX_L2SQR(l + 8, r + 8, sum, l1, r1);
				AVX_L2SQR(l + 16, r + 16, sum, l2, r2);
				AVX_L2SQR(l + 24, r + 24, sum, l3, r3);
			}
			result = _mm256_sumall_ps(sum);
#undef AVX_L2SQR
#elif SIMD_TYPE >= SIMDTYPE_SSE
#define SSE_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
			tmp1 = _mm_load_ps(addr1);\
			tmp2 = _mm_load_ps(addr2);\
			tmp1 = _mm_sub_ps(tmp1, tmp2); \
			dest = mm_fmadd_ps(tmp1, tmp1, dest);

			__m128 sum = mm_setzero_ps();
			__m128 l0, l1, l2, l3;
			__m128 r0, r1, r2, r3;
			unsigned D = (size + 3) & ~3U;
			unsigned DR = D % 16;
			unsigned DD = D - DR;
			const float *l = a;
			const float *r = b;
			const float *e_l = l + DD;
			const float *e_r = r + DD;
			switch (DR)
			{
			case 12:
				SSE_L2SQR(e_l + 8, e_r + 8, sum, l2, r2);
			case 8:
				SSE_L2SQR(e_l + 4, e_r + 4, sum, l1, r1);
			case 4:
				SSE_L2SQR(e_l, e_r, sum, l0, r0);
			default:
				break;
			}
			for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16)
			{
				SSE_L2SQR(l, r, sum, l0, r0);
				SSE_L2SQR(l + 4, r + 4, sum, l1, r1);
				SSE_L2SQR(l + 8, r + 8, sum, l2, r2);
				SSE_L2SQR(l + 12, r + 12, sum, l3, r3);
			}
			result = _mm_sumall_ps(sum);
#undef SSE_L2SQR
#else
			float diff0, diff1, diff2, diff3;
			const float* last = a + size;
			const float* unroll_group = last - 3;

			/* Process 4 items with each loop for efficiency. */
			while (a < unroll_group) {
				diff0 = a[0] - b[0];
				diff1 = a[1] - b[1];
				diff2 = a[2] - b[2];
				diff3 = a[3] - b[3];
				result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
				a += 4;
				b += 4;
			}
			/* Process last 0-3 pixels.  Not needed for standard vector lengths. */
			while (a < last) {
				diff0 = *a++ - *b++;
				result += diff0 * diff0;
			}
#endif
			if (!isfinite(result))
			{
				throw nsg_calculate_error("infinite number");
			}
			
			return result;
		}

		float DistanceInnerProduct::compare(const float* a, const float* b, unsigned size)
		{
			float result = 0;
#if SIMD_TYPE >= SIMDTYPE_AVX
#define AVX_DOT(addr1, addr2, dest, tmp1, tmp2) \
		  tmp1 = _mm256_loadu_ps(addr1);\
          tmp2 = _mm256_loadu_ps(addr2);\
		  dest = mm_fmadd_ps(tmp1, tmp2, dest);

			__m256 sum = mm_setzero_ps();
			__m256 l0, l1, l2, l3;
			__m256 r0, r1, r2, r3;
			unsigned D = (size + 7) & ~7U; // # dim aligned up to 256 bits, or 8 floats
			unsigned DR = D % 32;
			unsigned DD = D - DR;
			const float *l = a;
			const float *r = b;
			const float *e_l = l + DD;
			const float *e_r = r + DD;
			switch (DR) 
			{
			case 24:
				AVX_DOT(e_l + 16, e_r + 16, sum, l2, r2);
			case 16:
				AVX_DOT(e_l + 8, e_r + 8, sum, l1, r1);
			case 8:
				AVX_DOT(e_l, e_r, sum, l0, r0);
			}
			for (unsigned i = 0; i < DD; i += 32, l += 32, r += 32) 
			{
				AVX_DOT(l, r, sum, l0, r0);
				AVX_DOT(l + 8, r + 8, sum, l1, r1);
				AVX_DOT(l + 16, r + 16, sum, l2, r2);
				AVX_DOT(l + 24, r + 24, sum, l3, r3);
			}
			result = _mm256_sumall_ps(sum);
#undef AVX_DOT
#elif SIMD_TYPE >= SIMDTYPE_SSE
#define SSE_DOT(addr1, addr2, dest, tmp1, tmp2) \
          tmp1 = _mm_loadu_ps(addr1);\
          tmp2 = _mm_loadu_ps(addr2);\
          dest = mm_fmadd_ps(tmp1, tmp2, dest);

			__m128 sum = mm_setzero_ps();
			__m128 l0, l1, l2, l3;
			__m128 r0, r1, r2, r3;
			unsigned D = (size + 3) & ~3U;
			unsigned DR = D % 16;
			unsigned DD = D - DR;
			const float *l = a;
			const float *r = b;
			const float *e_l = l + DD;
			const float *e_r = r + DD;
			switch (DR) 
			{
			case 12:
				SSE_DOT(e_l + 8, e_r + 8, sum, l2, r2);
			case 8:
				SSE_DOT(e_l + 4, e_r + 4, sum, l1, r1);
			case 4:
				SSE_DOT(e_l, e_r, sum, l0, r0);
			default:
				break;
			}
			for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) 
			{
				SSE_DOT(l, r, sum, l0, r0);
				SSE_DOT(l + 4, r + 4, sum, l1, r1);
				SSE_DOT(l + 8, r + 8, sum, l2, r2);
				SSE_DOT(l + 12, r + 12, sum, l3, r3);
			}
			result = _mm_sumall_ps(sum);
#undef SSE_DOT
#else
			float dot0, dot1, dot2, dot3;
			const float* last = a + size;
			const float* unroll_group = last - 3;

			/* Process 4 items with each loop for efficiency. */
			while (a < unroll_group) {
				dot0 = a[0] * b[0];
				dot1 = a[1] * b[1];
				dot2 = a[2] * b[2];
				dot3 = a[3] * b[3];
				result += dot0 + dot1 + dot2 + dot3;
				a += 4;
				b += 4;
			}
			/* Process last 0-3 pixels.  Not needed for standard vector lengths. */
			while (a < last) 
			{
				result += *a++ * *b++;
			}
#endif

			if (!isfinite(result))
			{
				throw nsg_calculate_error("infinite number");
			}

			return result;
		}

		float DistanceFastL2::norm(const float* a, unsigned size)
		{
			float result = 0;
#if SIMD_TYPE >= SIMDTYPE_AVX
#define AVX_L2NORM(addr, dest, tmp) \
	tmp = _mm256_loadu_ps(addr); \
	dest = mm_fmadd_ps(tmp, tmp, dest);

			__m256 sum = mm_setzero_ps();
			__m256 l0, l1, l2, l3;
			unsigned D = (size + 7) & ~7U; // # dim aligned up to 256 bits, or 8 floats
			unsigned DR = D % 32;
			unsigned DD = D - DR;
			const float *l = a;
			const float *e_l = l + DD;
			switch (DR) 
			{
			case 24:
				AVX_L2NORM(e_l + 16, sum, l2);
			case 16:
				AVX_L2NORM(e_l + 8, sum, l1);
			case 8:
				AVX_L2NORM(e_l, sum, l0);
			}
			for (unsigned i = 0; i < DD; i += 32, l += 32) 
			{
				AVX_L2NORM(l, sum, l0);
				AVX_L2NORM(l + 8, sum, l1);
				AVX_L2NORM(l + 16, sum, l2);
				AVX_L2NORM(l + 24, sum, l3);
			}
			result = _mm256_sumall_ps(sum);
#undef AVX_L2NORM
#elif SIMD_TYPE >= SIMDTYPE_SSE
#define SSE_L2NORM(addr, dest, tmp) \
    tmp = _mm_loadu_ps(addr); \
	dest = mm_fmadd_ps(tmp, tmp, dest);

			__m128 sum = mm_setzero_ps();
			__m128 l0, l1, l2, l3;
			unsigned D = (size + 3) & ~3U;
			unsigned DR = D % 16;
			unsigned DD = D - DR;
			const float *l = a;
			const float *e_l = l + DD;
			switch (DR) {
			case 12:
				SSE_L2NORM(e_l + 8, sum, l2);
			case 8:
				SSE_L2NORM(e_l + 4, sum, l1);
			case 4:
				SSE_L2NORM(e_l, sum, l0);
			default:
				break;
			}
			for (unsigned i = 0; i < DD; i += 16, l += 16) 
			{
				SSE_L2NORM(l, sum, l0);
				SSE_L2NORM(l + 4, sum, l1);
				SSE_L2NORM(l + 8, sum, l2);
				SSE_L2NORM(l + 12, sum, l3);
			}
			result = _mm_sumall_ps(sum);
#undef SSE_L2NORM
#else
			float dot0, dot1, dot2, dot3;
			const float* last = a + size;
			const float* unroll_group = last - 3;

			/* Process 4 items with each loop for efficiency. */
			while (a < unroll_group) 
			{
				dot0 = a[0] * a[0];
				dot1 = a[1] * a[1];
				dot2 = a[2] * a[2];
				dot3 = a[3] * a[3];
				result += dot0 + dot1 + dot2 + dot3;
				a += 4;
			}
			/* Process last 0-3 pixels.  Not needed for standard vector lengths. */
			while (a < last) 
			{
				result += (*a) * (*a);
				a++;
			}
#endif
			
			if (!isfinite(result))
			{
				throw nsg_calculate_error("infinite number");
			}
			else if (abs(result) < 1e-5)
			{
				throw nsg_calculate_error("zero vector");
			}

			return result;
		}

		float DistanceFastL2::compare(const float* a, float norma, const float* b, float normb, unsigned size)
		{
			float result = -2 * DistanceInnerProduct::compare(a, b, size);
			result = result + norma + normb;//(a-b)*(a-b)=a^2 + b^2 - 2*a*b

			if (!isfinite(result))
			{
				throw nsg_calculate_error("infinite number");
			}

			return result;
		}

		float DistanceCosine::norm(const float* a, unsigned size)
		{
			float result = 0;
#if SIMD_TYPE >= SIMDTYPE_AVX
#define AVX_L2NORM2(addr, dest, tmp) \
	tmp = _mm256_loadu_ps(addr); \
	dest = mm_fmadd_ps(tmp, tmp, dest);

			__m256 sum = mm_setzero_ps();
			__m256 l0, l1, l2, l3;
			unsigned D = (size + 7) & ~7U; // # dim aligned up to 256 bits, or 8 floats
			unsigned DR = D % 32;
			unsigned DD = D - DR;
			const float *l = a;
			const float *e_l = l + DD;
			switch (DR) 
			{
			case 24:
				AVX_L2NORM2(e_l + 16, sum, l2);
			case 16:
				AVX_L2NORM2(e_l + 8, sum, l1);
			case 8:
				AVX_L2NORM2(e_l, sum, l0);
			}
			for (unsigned i = 0; i < DD; i += 32, l += 32) 
			{
				AVX_L2NORM2(l, sum, l0);
				AVX_L2NORM2(l + 8, sum, l1);
				AVX_L2NORM2(l + 16, sum, l2);
				AVX_L2NORM2(l + 24, sum, l3);
			}
			result = _mm256_sumall_ps(sum);
#undef AVX_L2NORM2
#elif SIMD_TYPE >= SIMDTYPE_SSE
#define SSE_L2NORM2(addr, dest, tmp) \
    tmp = _mm_loadu_ps(addr); \
	dest = mm_fmadd_ps(tmp, tmp, dest);

			__m128 sum = mm_setzero_ps();
			__m128 l0, l1, l2, l3;
			unsigned D = (size + 3) & ~3U;
			unsigned DR = D % 16;
			unsigned DD = D - DR;
			const float *l = a;
			const float *e_l = l + DD;
			switch (DR) 
			{
			case 12:
				SSE_L2NORM2(e_l + 8, sum, l2);
			case 8:
				SSE_L2NORM2(e_l + 4, sum, l1);
			case 4:
				SSE_L2NORM2(e_l, sum, l0);
			default:
				break;
			}
			for (unsigned i = 0; i < DD; i += 16, l += 16) 
			{
				SSE_L2NORM2(l, sum, l0);
				SSE_L2NORM2(l + 4, sum, l1);
				SSE_L2NORM2(l + 8, sum, l2);
				SSE_L2NORM2(l + 12, sum, l3);
			}
			result = _mm_sumall_ps(sum);
#undef SSE_L2NORM2
#else
			float dot0, dot1, dot2, dot3;
			const float* last = a + size;
			const float* unroll_group = last - 3;

			/* Process 4 items with each loop for efficiency. */
			while (a < unroll_group) 
			{
				dot0 = a[0] * a[0];
				dot1 = a[1] * a[1];
				dot2 = a[2] * a[2];
				dot3 = a[3] * a[3];
				result += dot0 + dot1 + dot2 + dot3;
				a += 4;
			}
			/* Process last 0-3 pixels.  Not needed for standard vector lengths. */
			while (a < last) 
			{
				result += (*a) * (*a);
				a++;
			}
#endif
			result = sqrt(result);

			if (!isfinite(result))
			{
				throw nsg_calculate_error("infinite number");
			}
			else if (abs(result) < 1e-5)
			{
				throw nsg_calculate_error("zero vector");
			}

			return result;
		}

		float DistanceCosine::compare(const float* a, float norma, const float* b, float normb, unsigned size)
		{
			float result = DistanceInnerProduct::compare(a, b, size);
			result = result / (norma * normb);
			result = 1-result;//more similar, distance should be closer, so we add minus before result

			if (!isfinite(result))
			{
				throw nsg_calculate_error("infinite number");
			}

			return result;
		}
	}
}
