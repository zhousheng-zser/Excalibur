#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
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

	template <template <typename> typename Condition, typename>
	struct tuple_if;

	template <template <typename> typename Condition, typename... Args>
	struct tuple_if<Condition, std::tuple<Args...>>
	{ 
		using type = tuple_cat_t<typename std::conditional<Condition<T>::value, std::tuple<Args>, std::tuple<>>::type...>;
	};

	template <template <typename> typename Condition, typename T>
	using tuple_if_t = typename tuple_if<Condition, T>::type;

	template<typename T, typename = std::enable_if_t<std::is_standard_layout_v<T>>>
	inline constexpr std::size_t hexadecimal_character_size_v = sizeof(T) * 2;

	template<typename Number>
	constexpr auto make_number(const std::array<std::uint8_t, sizeof(Number)>& data, bool big_endian = true) noexcept -> std::enable_if_t<std::is_arithmetic_v<Number>, Number>
	{
		return details::make_number_impl<Number>(data, std::make_index_sequence<sizeof(Number)>{}, big_endian);
	}

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
