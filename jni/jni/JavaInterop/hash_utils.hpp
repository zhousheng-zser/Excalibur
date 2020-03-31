#pragma once

#include <cstddef>
#include <type_traits>

namespace glasssix::utils
{
#if (defined(_WIN32) && !defined(_WIN64)) || (defined(__ARM_32BIT_STATE) && __ARM_32BIT_STATE)
	template<typename T>
	void hash_combine(std::size_t& result, T&& value) noexcept
	{
		if constexpr (std::is_array_v<T>)
		{
			for (auto& item : std::forward<T>(value))
			{
				hash_combine(result, item);
			}
		}
		else
		{
			using pure_type = std::decay_t<T>;

			constexpr std::size_t magic_factor = 0x9E3779B9;
			auto hash = std::hash<pure_type>{}(std::forward<T>(value));

			result ^= hash + magic_factor + (result << 6) + (result >> 2);
		}
	}
#elif defined(_WIN64) || (defined(__ARM_64BIT_STATE) && __ARM_64BIT_STATE)
	template<typename T>
	void hash_combine(std::size_t& result, T&& value) noexcept
	{
		if constexpr (std::is_array_v<T>)
		{
			for (auto& item : std::forward<T>(value))
			{
				hash_combine(result, item);
			}
		}
		else
		{
			using pure_type = std::decay_t<T>;

			constexpr std::size_t magic_factor = 0xC6A4A7935BD1E995;
			auto hash = std::hash<pure_type>{}(std::forward<T>(value));

			hash *= magic_factor;
			hash ^= hash >> 47;
			hash *= magic_factor;

			result ^= hash;
			result *= magic_factor;

			// Completely arbitrary number, to prevent 0's from hashing to 0.
			result += 0xE6546B64;
		}
	}
#endif

	template<typename... Args>
	constexpr std::size_t hash_all(Args&&... args) noexcept
	{
		std::size_t result = 0;

		(hash_combine(result, std::forward<Args>(args)), ...);

		return result;
	}
}
