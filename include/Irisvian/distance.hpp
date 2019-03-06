#ifndef _DISTANCE_HPP_
#define _DISTANCE_HPP_



//#ifdef _MSC_VER
//#include <intrin.h>
//#else
//#include <x86intrin.h>
//#include <avxintrin.h>
//#endif
//
//#include <immintrin.h>
//#include <xmmintrin.h>
//
//#include <iostream>
//#include <fstream>
//
//#include <vector>
//#include <bitset>
//#include <array>
//#include <string>
//#include <math.h>


namespace glasssix {
	namespace Irisvian {

		class DistanceL2 
		{
		public:
			static float compare(const float* a, const float* b, unsigned size);
		};


		class DistanceInnerProduct
		{
		public:
			static float compare(const float* a, const float* b, unsigned size);
		};


		class DistanceFastL2 : public DistanceInnerProduct 
		{
		public:
			static float norm(const float* a, unsigned size);

            static float compare(const float* a, float norma, const float* b, float normb, unsigned size);
		};


		class DistanceCosine : public DistanceInnerProduct 
		{
		public:
			static float norm(const float* a, unsigned size);

            static float compare(const float* a, float norma, const float* b, float normb, unsigned size);
		};
	}
}

#endif // _DISTANCE_HPP_