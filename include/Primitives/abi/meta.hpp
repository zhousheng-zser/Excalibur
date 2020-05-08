#pragma once

#include <array>
#include <tuple>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <optional>
#include <string_view>
#include <type_traits>

namespace glasssix::exposing::meta
{
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
		using type = tuple_first<std::tuple<Args...>>;
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

	template<typename Array, typename = void>
	struct is_std_array : std::false_type {};

	template<typename T, std::size_t Size>
	struct is_std_array<std::array<T, Size>> : std::true_type {};

	/// <summary>
	/// Checks whether a type is a std::array.
	/// </summary>
	template<typename Array>
	inline constexpr bool is_std_array_v = is_std_array<Array>::value;

	template<typename Array>
	struct std_array_traits;

	template<typename T, std::size_t Size>
	struct std_array_traits<std::array<T, Size>>
	{
		using element_type = T;
		static constexpr std::size_t size = Size;
	};

	/// <summary>
	/// Gets the element type of a std::array.
	/// </summary>
	template<typename Array>
	using std_array_element_t = typename std_array_traits<std::decay_t<Array>>::element_type;

	/// <summary>
	/// Gets the size of a std::array.
	/// </summary>
	template<typename Array>
	inline constexpr std::size_t std_array_size_v = std_array_traits<std::decay_t<Array>>::size;

	/// <summary>
	/// Get the sum of numbers.
	/// </summary>
	/// <typeparam name="Numbers">The numeric types</typeparam>
	/// <param name="...args">The numbers</param>
	/// <returns>The sum</returns>
	template<typename... Numbers, typename = std::enable_if_t<std::conjunction_v<std::is_arithmetic<Numbers>...>>>
	constexpr auto sum(Numbers&&... args) noexcept
	{
		using result_type = std::common_type_t<Numbers...>;

		return (static_cast<result_type>(std::forward<Numbers>(args)) + ...);
	}

	namespace details
	{
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

		template<typename Number, typename Callable, std::size_t... Indexes, typename = std::enable_if_t<std::is_arithmetic_v<Number>>>
		constexpr auto number_move_bits_helper(std::index_sequence<Indexes...>, bool big_endian, Callable&& handler) noexcept
		{
			constexpr std::ptrdiff_t total_bits = sizeof(Number) * CHAR_BIT;
			constexpr std::ptrdiff_t max_move_bits = total_bits - CHAR_BIT;
			std::ptrdiff_t baseline_move_bits = big_endian ? max_move_bits : 0;
			std::ptrdiff_t sign = big_endian ? -1 : 1;

			return std::forward<Callable>(handler)((std::pair{ Indexes, baseline_move_bits + sign * static_cast<std::ptrdiff_t>(Indexes) * CHAR_BIT })...);
		}

		template<typename T, typename U, std::size_t Size, std::size_t... Indexes>
		constexpr void set_array_value_helper(T* result, const std::array<U, Size>& source, std::index_sequence<Indexes...>) noexcept
		{
			((result[Indexes] = static_cast<T>(source[Indexes])), ...);
		}

		template<std::size_t... Indexes, typename... Arrays>
		constexpr auto concat_arrays_impl(std::index_sequence<Indexes...>, Arrays&&... arrays) noexcept
		{
			constexpr std::array<std::size_t, sizeof...(Arrays) + 1> sizes{ 0, std_array_size_v<Arrays>... };
			std::array<std::common_type_t<std_array_element_t<Arrays>...>, (std_array_size_v<Arrays> + ...)> result{};
			std::size_t offset = 0;

			return (details::set_array_value_helper(result.data() + (offset += sizes[Indexes]), std::forward<Arrays>(arrays), std::make_index_sequence<std_array_size_v<Arrays>>{}), ..., result);
		}

		template<bool left, typename UnsignedNumber, typename = std::enable_if_t<std::is_unsigned_v<UnsignedNumber>>>
		constexpr auto bitwise_rotate_impl(UnsignedNumber number, int bits) noexcept
		{
			using limits_type = std::numeric_limits<UnsignedNumber>;
			using rotate_impl_type = UnsignedNumber(*)(UnsignedNumber number, int bits);

			constexpr auto numeric_bits = limits_type::digits;
			constexpr auto rotl_impl = [](UnsignedNumber number, int bits) { return (number << bits) | (number >> (numeric_bits - bits)); };
			constexpr auto rotr_impl = [](UnsignedNumber number, int bits) { return (number >> bits) | (number << (numeric_bits - bits)); };
			constexpr auto rotate_impl = left ? static_cast<rotate_impl_type>(rotl_impl) : rotr_impl;

			if (bits == 0)
			{
				return number;
			}

			bits %= numeric_bits;

			if (bits > 0)
			{
				return rotate_impl(number, bits);
			}
			else
			{
				return bitwise_rotate_impl<!left>(number, -bits);
			}
		}

		template<typename Callable, std::size_t... Indexes>
		constexpr auto apply_index_sequence_impl(Callable&& handler, std::index_sequence<Indexes...>) noexcept
		{
			return std::forward<Callable>(handler)(Indexes...);
		}
	}

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
	/// <typeparam name="Number">The numeric type</typeparam>
	/// <param name="data">The bytes</param>
	/// <param name="big_endian">A boolean that indicates whether the byte order is big-endian</param>
	/// <returns>The number</returns>
	template<typename Number>
	constexpr auto make_number(const std::array<std::uint8_t, sizeof(Number)>& data, bool big_endian = true) noexcept -> std::enable_if_t<std::is_arithmetic_v<Number>, Number>
	{
		return details::number_move_bits_helper<Number>(std::make_index_sequence<sizeof(Number)>{}, big_endian, [&](auto&&... parts)
			{
				return ((static_cast<std::uintmax_t>(data[std::forward<decltype(parts)>(parts).first]) << std::forward<decltype(parts)>(parts).second) + ...);
			});
	}

	/// <summary>
	/// Parses a string containing hexadecimal digits into a number.
	/// </summary>
	/// <typeparam name="Number">The numeric type</typeparam>
	/// <param name="str">The string</param>
	/// <param name="big_endian">A boolean that indicates whether the byte order is big-endian</param>
	/// <returns>The number</returns>
	template<typename Number>
	constexpr auto to_number(std::string_view str, bool big_endian = true) noexcept -> std::enable_if_t<std::is_arithmetic_v<Number>, Number>
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

	/// <summary>
	/// Retrieves the bytes of a number.
	/// </summary>
	/// <typeparam name="Number">The numeric type</typeparam>
	/// <param name="number">The number</param>
	/// <param name="big_endian">A boolean that indicates whether the byte order is big-endian</param>
	/// <returns>The array</returns>
	template<typename Number, typename = std::enable_if_t<std::is_arithmetic_v<Number>>>
	constexpr auto to_array(Number number, bool big_endian = true) noexcept
	{
		return details::number_move_bits_helper<Number>(std::make_index_sequence<sizeof(Number)>{}, big_endian, [&](auto&&... parts)
			{
				return std::array<std::uint8_t, sizeof(Number)>
				{
					((static_cast<std::uintmax_t>(number) >> std::forward<decltype(parts)>(parts).second) & 0xFF)...
				};
			});
	}

	/// <summary>
	/// Concatenates arrays.
	/// </summary>
	/// <typeparam name="...Arrays">The array types</typeparam>
	/// <param name="...arrays">The arrays</param>
	/// <returns>The merged array</returns>
	template<typename... Arrays, typename = std::enable_if_t<std::conjunction_v<is_std_array<std::decay_t<Arrays>>...>>>
	constexpr auto concat_arrays(Arrays&&... arrays) noexcept
	{
		return details::concat_arrays_impl(std::make_index_sequence<sizeof...(Arrays)>{}, std::forward<Arrays>(arrays)...);
	}

	/// <summary>
	/// Computes the result of bitwise left-rotating the value of "number" by "bits" positions.
	/// This operation is also known as a left circular shift.
	/// </summary>
	/// <typeparam name="UnsignedNumber">The unsigned numeric type</typeparam>
	/// <param name="number">The unsigned number</param>
	/// <param name="bits">The bits</param>
	/// <returns>The result</returns>
	template<typename UnsignedNumber, typename = std::enable_if_t<std::is_unsigned_v<UnsignedNumber>>>
	constexpr auto rotl(UnsignedNumber number, int bits) noexcept
	{
		return details::bitwise_rotate_impl<true>(number, bits);
	}

	/// <summary>
	/// Computes the result of bitwise right-rotating the value of "number" by "bits" positions.
	/// This operation is also known as a right circular shift.
	/// </summary>
	/// <typeparam name="UnsignedNumber">The unsigned numeric type</typeparam>
	/// <param name="number">The unsigned number</param>
	/// <param name="bits">The bits</param>
	/// <returns>The result</returns>
	template<typename UnsignedNumber, typename = std::enable_if_t<std::is_unsigned_v<UnsignedNumber>>>
	constexpr auto rotr(UnsignedNumber number, int bits) noexcept
	{
		return details::bitwise_rotate_impl<false>(number, bits);
	}

	/// <summary>
	/// Sets certain bit of an unsigned number.
	/// </summary>
	/// <typeparam name="UnsignedNumber">The numeric type</typeparam>
	/// <param name="number">The number</param>
	/// <param name="offset">The bit offset</param>
	/// <param name="bit">The bit value</param>
	template<typename UnsignedNumber, typename = std::enable_if_t<std::is_unsigned_v<UnsignedNumber>>>
	constexpr void set_number_bit(UnsignedNumber& number, int offset, std::uint8_t bit) noexcept
	{
		number = (number & ~(1UL << offset)) | (bit << offset);
	}

	/// <summary>
	/// Calculates the binary logarithm of an unsigned number.
	/// </summary>
	/// <typeparam name="UnsignedNumber">The numeric type</typeparam>
	/// <param name="number">The number</param>
	/// <returns>The result</returns>
	template<typename UnsignedNumber, typename = std::enable_if_t<std::is_unsigned_v<UnsignedNumber>>>
	constexpr auto log2(UnsignedNumber number) noexcept
	{
		UnsignedNumber result{};

		while ((number >>= 1) != 0)
		{
			result++;
		}

		return result;
	}

	/// <summary>
	/// Applies an index sequence to a callable handler.
	/// </summary>
	/// <typeparam name="Callable">The callable type</typeparam>
	/// <param name="Size">The size of the sequence</param>
	/// <param name="handler">The handler</param>
	/// <returns>The result of the callable handler</returns>
	template<std::size_t Size, typename Callable>
	constexpr decltype(auto) apply_index_sequence(Callable&& handler) noexcept
	{
		return details::apply_index_sequence_impl(std::forward<Callable>(handler), std::make_index_sequence<Size>{});
	}
}
