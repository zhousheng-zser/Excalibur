#pragma once

#include <cctype>
#include <string>
#include <cstddef>
#include <utility>
#include <algorithm>

namespace glasssix
{
	namespace utils
	{
#if defined(_MSC_VER) && !defined(_WIN64)
		template<typename T>
		void hash_combine(std::size_t& result, T&& value)
		{
			using pure_type = std::decay_t<T>;

			constexpr std::size_t magic_factor = 0x9E3779B9;
			auto hash = std::hash<pure_type>{}(std::forward<T>(value));

			result ^= hash + magic_factor + (result << 6) + (result >> 2);
		}
#elif defined(_MSC_VER) && defined(_WIN64)
		template<typename T>
		void hash_combine(std::size_t& result, T&& value)
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
#endif
		template<typename... Args>
		std::size_t hash_all(Args&&... args)
		{
			std::size_t result = 0;
			int dummy[] = { (hash_combine(result, std::forward<Args>(args)), 0)... };

			return result;
		}
	}

	/// <summary>
	/// A hash function for std::string with a case-insensitive calculator.
	/// </summary>
	struct case_insensitive_string_hash
	{
		auto operator()(const std::string& value) const
		{
			std::size_t result = 0;

			std::for_each(value.cbegin(), value.cend(), [&](int c) { utils::hash_combine(result, std::tolower(c)); });

			return result;
		}
	};

	/// <summary>
	/// A case-insensitive comparer for std::string.
	/// </summary>
	struct case_insensitive_string_comparer
	{
		bool operator()(const std::string& left, const std::string& right) const
		{
			return left.size() == right.size() && std::equal(std::begin(left), std::end(left), std::begin(right), std::end(right), [](int left, int right) { return left == right || std::tolower(left) == std::tolower(right); });
		}
	};
}
