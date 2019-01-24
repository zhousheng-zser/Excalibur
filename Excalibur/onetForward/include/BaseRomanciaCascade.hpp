#ifndef BASECASCADE_HPP
#define BASECASCADE_HPP
#include <vector>
#include "common.hpp"

namespace glasssix
{
	namespace longinus
	{
		typedef enum RomanciaCascadeType
		{
			FRONTAL,
			FRONTAL_REINFORCE,
			LEFT_PROFILE,
			LEFT_PROFILE_REINFORCE,
			RIGHT_PROFILE,
			RIGHT_PROFILE_REINFORCE
		}RomanciaCascadeType;

		class BaseRomanciaCascade
		{
		public:
			virtual ~BaseRomanciaCascade() {}
			virtual void LoadCascade(RomanciaCascadeType cascadeType, int device = -1) = 0;
			virtual int getWinWidth() const = 0;
			virtual int getWinHeight() const = 0;
			virtual bool isEmpty() = 0;
		};
	}
}

#endif

