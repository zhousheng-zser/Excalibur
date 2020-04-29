#pragma once

#include <array>
#include <tuple>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <optional>
#include <string_view>
#include <type_traits>

namespace glasssix::exposing::meta
{
	namespace details
	{
		template<typename Number, std::size_t... Indexes>
		constexpr auto make_number_impl(const std::array<std::uint8_t, sizeof(Number)>& data, std::index_sequence<Indexes...>, bool big_endian) noexcept -> std::enable_if_t<std::is_arithmetic_v<Number>, Number>
		{
			constexpr std::ptrdiff_t total_bits = sizeof(Number) * CHAR_BIT;
			constexpr std::ptrdiff_t max_move_bits = total_bits - CHAR_BIT;
			std::ptrdiff_t baseline_move_bits = big_endian ? max_move_bits : 0;
			std::ptrdiff_t sign = big_endian ? -1 : 1;

			return static_cast<Number>(((static_cast<std::uintmax_t>(data[Indexes]) << (baseline_move_bits + sign * static_cast<std::ptrdiff_t>(Indexes) * CHAR_BIT)) + ...));
		}

		constexpr std::optional<std::uint8_t> from_hexadecimal_character(int character) noexcept
		{
			if (character >= '0' && character <= '9')
			{
				return character - '0';
			}

			if (character >= 'A' && character <= 'F')
			{
				return character - 'A' + 0xA;
			}

			if (character >= 'a' && character <= 'f')
			{
				return character - 'a' + 0xA;
			}

			return std::nullopt;
		}
	}

	template<typename...T>
	using tuple_cat_t = decltype(std::tuple_cat(std::declval<T>()...));

	template<template<typename> typename Condition, typename>
	struct tuple_if;

	template<template<typename> typename Condition, typename... Args> 
	struct tuple_if<Condition, std::tuple<Args...>>
	{
		using type = tuple_cat_t<typename std::conditional<Condition<Args>::value, std::tuple<Args>, std::tuple<>>::type...>;
	};

	template<template<typename> typename Condition, typename Tuple>
	using tuple_if_t = typename tuple_if<Condition, Tuple>::type;

	template<typename Tuple>
	struct tuple_first;

	template<typename... Args>
	struct tuple_first<std::tuple<Args...>>
	{
		using type = std::tuple_element_t<0, std::tuple<Args...>>;
	};

	template<typename Tuple>
	using tuple_first_t = typename tuple_first<Tuple>::type;

	template<typename... Args>
	struct first_of_template_arguments
	{
		using type = tuple_first<std::tuple<Args...>>
	};

	/// <summary>
	/// Gets the first argument of variadic parameters.
	/// </summary>
	template<typename... Args>
	using first_of_template_arguments_t = typename first_of_template_arguments<Args...>::type;

	template<template<typename> typename Condition, typename... Args>
	struct first_of_template_arguments_if
	{
		using type = tuple_first<tuple_if_t<Condition, std::tuple<Args...>>>;
	};

	/// <summary>
	/// Gets the first argument that santisfies a specified condition, of variadic parameters.
	/// </summary>
	template<template<typename> typename Condition, typename... Args>
	using first_of_template_arguments_if_t = typename first_of_template_arguments_if<Condition, Args...>::type;

	/// <summary>
	/// Gets the size of hexadecimal characters which represent the data of a type in standard layout.
	/// </summary>
	template<typename T, typename = std::enable_if_t<std::is_standard_layout_v<T>>>
	inline constexpr std::size_t hexadecimal_character_size_v = sizeof(T) * 2;

	/// <summary>
	/// Retrieves a reference to the first member of an object arranged in standard layout.
	/// </summary>
	/// <typeparam name="FirstMember">The type of the first member</typeparam>
	/// <typeparam name="T">The object type</typeparam>
	/// <param name="obj">The object</param>
	/// <returns>The reference to the first member</returns>
	template<typename FirstMember, typename T, typename = std::enable_if_t<std::conjunction_v<std::is_standard_layout<std::decay_t<T>>, std::is_lvalue_reference<T>>>>
	constexpr decltype(auto) get_standard_layout_first_member(T&& obj) noexcept
	{
		using result_type = std::conditional_t<std::is_const_v<std::remove_reference_t<T>>, std::add_const_t<FirstMember>&, FirstMember&>;

		return reinterpret_cast<result_type>(std::forward<T>(obj));
	}

	/// <summary>
	/// Retrieves a reference to an object arranged in standard layout by the first member of it.
	/// </summary>
	/// <typeparam name="T">The object type</typeparam>
	/// <typeparam name="FirstMember">The type of the first member</typeparam>
	/// <param name="member">The first member</param>
	/// <returns>The reference to the object</returns>
	template<typename T, typename FirstMember, typename = std::enable_if_t<std::conjunction_v<std::is_standard_layout<T>, std::is_lvalue_reference<FirstMember>>>>
	constexpr decltype(auto) get_standard_layout_from_first_member(FirstMember&& member) noexcept
	{
		using result_type = std::conditional_t<std::is_const_v<std::remove_reference_t<FirstMember>>, std::add_const_t<T>&, T&>;

		return reinterpret_cast<result_type>(std::forward<FirstMember>(member));
	}

	/// <summary>
	/// Combines multiple bytes into a number.
	/// </summary>
	/// <param name="data">The bytes</param>
	/// <param name="big_endian">A boolean that indicates whether the byte order is big-endian</param>
	/// <returns>The number</returns>
	template<typename Number>
	constexpr auto make_number(const std::array<std::uint8_t, sizeof(Number)>& data, bool big_endian = true) noexcept -> std::enable_if_t<std::is_arithmetic_v<Number>, Number>
	{
		return details::make_number_impl<Number>(data, std::make_index_sequence<sizeof(Number)>{}, big_endian);
	}

	/// <summary>
	/// Parses a string containing hexadecimal digits into a number.
	/// </summary>
	/// <param name="str">The string</param>
	/// <param name="big_endian">A boolean that indicates whether the byte order is big-endian</param>
	/// <returns>The number</returns>
	template<typename Number>
	constexpr auto to_number(std::string_view str, bool big_endian = true) -> std::enable_if_t<std::is_arithmetic_v<Number>, Number>
	{
		// Ensures security.
		if (str.size() / hexadecimal_character_size_v<std::uint8_t> < sizeof(Number))
		{
			return Number{};
		}

		std::array<std::uint8_t, sizeof(Number)> result{};
		auto source_ptr = str.data();
		auto destination_ptr = result.data();

		// Converts hexadecimal characters to raw bytes.
		for (auto end_ptr = result.data() + result.size(); destination_ptr < end_ptr; source_ptr += hexadecimal_character_size_v<std::uint8_t>, destination_ptr++)
		{
			if (auto first_part = details::from_hexadecimal_character(source_ptr[0]), second_part = details::from_hexadecimal_character(source_ptr[1]); first_part && second_part)
			{
				// Combines two nibbles into one single byte.
				*destination_ptr = static_cast<std::uint8_t>(((*first_part) << 4) + *second_part);
				continue;
			}

			return Number{};
		}

		return make_number<Number>(result, big_endian);
	}
}
