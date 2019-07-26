#ifndef _DISTANCE_HPP_
#define _DISTANCE_HPP_

namespace glasssix {
	namespace irisviel {

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