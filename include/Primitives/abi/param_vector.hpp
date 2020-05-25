#pragma once

#include "meta.hpp"
#include "base.hpp"
#include "base_abi.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <iterator>
#include <type_traits>

namespace glasssix::exposing
{
	template<typename T>
	class param_vector;
}

namespace glasssix::exposing::impl
{
	template<typename T>
	struct abi<param_vector<T>>
	{
		using identity_type = type_identity_generic_interface;
		static constexpr guid id{ "DCB2A5A5-1D17-4E0A-83C2-640912AECD25" };

		struct type : abi_unknown_object
		{

		};
	};
}

namespace glasssix::exposing
{
	template<typename T>
	class param_vector
	{

	};
}
