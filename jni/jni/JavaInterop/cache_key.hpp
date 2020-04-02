#pragma once

#include "hash_utils.hpp"

#include <array>
#include <numeric>
#include <functional>
#include <type_traits>

#include <jni.h>

namespace glasssix::jni
{
	struct cache_key
	{
		int class_part;
		int field_part;
		int method_part;
	};
}

namespace glasssix::jni::utils
{
	template<auto Enum>
	inline constexpr auto arg_enum_v = static_cast<std::enable_if_t<std::is_enum_v<decltype(Enum)>, int>>(Enum);

	template<typename T>
	constexpr auto make_cache_key(int key) noexcept
	{
		constexpr int mask = std::numeric_limits<int>::max();

		if constexpr (std::is_same_v<T, jclass>)
		{
			return cache_key{ key, mask, mask };
		}
		else if constexpr (std::is_same_v<T, jfieldID>)
		{
			return cache_key{ mask, key, mask };
		}
		else if constexpr (std::is_same_v<T, jmethodID>)
		{
			return cache_key{ mask, mask, key };
		}
		else
		{
			return cache_key{ mask, mask, mask };
		}
	}
}

namespace std
{
	template<> struct hash<glasssix::jni::cache_key>
	{
		std::size_t operator()(const glasssix::jni::cache_key& item) const noexcept
		{
			return glasssix::utils::hash_all(item.class_part, item.field_part, item.method_part);
		}
	};

	template<> struct equal_to<glasssix::jni::cache_key>
	{
		constexpr bool operator()(const glasssix::jni::cache_key& left, const glasssix::jni::cache_key& right) const noexcept
		{
			return left.class_part == right.class_part && left.field_part == right.field_part && left.method_part == right.method_part;
		}
	};
}
