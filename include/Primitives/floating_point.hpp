#pragma once

#include <cmath>
#include <limits>
#include <type_traits>

namespace glasssix
{
	inline constexpr int floating_point_default_ulp = 2;

	template<typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
	bool almost_equals(T x, T y, int ulp = floating_point_default_ulp) noexcept
	{
		// the machine epsilon has to be scaled to the magnitude of the values used
		// and multiplied by the desired precision in ULPs (units in the last place)
		// unless the result is subnormal.
		return std::fabs(x - y) <= std::numeric_limits<T>::epsilon() * std::fabs(x + y) * ulp || std::fabs(x - y) < std::numeric_limits<T>::min();
	}
}
