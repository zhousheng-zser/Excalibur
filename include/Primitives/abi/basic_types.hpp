#pragma once

#include "base.hpp"

namespace glasssix::abi::impl
{
	template<> inline constexpr auto& name_v<bool> = "b1";
	template<> inline constexpr auto& name_v<std::int8_t> = "i1";
	template<> inline constexpr auto& name_v<std::int16_t> = "i2";
	template<> inline constexpr auto& name_v<std::int32_t> = "i4";
	template<> inline constexpr auto& name_v<std::int64_t> = "i8";
	template<> inline constexpr auto& name_v<std::uint8_t> = "u1";
	template<> inline constexpr auto& name_v<std::uint16_t> = "u2";
	template<> inline constexpr auto& name_v<std::uint32_t> = "u4";
	template<> inline constexpr auto& name_v<std::uint64_t> = "u8";
	template<> inline constexpr auto& name_v<float> = "f4";
	template<> inline constexpr auto& name_v<double> = "f8";

	template<> struct category<bool>
	{
		using type = category_basic_tag;
	};

	template<> struct category<std::int8_t>
	{
		using type = category_basic_tag;
	};

	template<> struct category<std::int16_t>
	{
		using type = category_basic_tag;
	};

	template<> struct category<std::int32_t>
	{
		using type = category_basic_tag;
	};

	template<> struct category<std::int64_t>
	{
		using type = category_basic_tag;
	};

	template<> struct category<std::uint8_t>
	{
		using type = category_basic_tag;
	};

	template<> struct category<std::uint16_t>
	{
		using type = category_basic_tag;
	};

	template<> struct category<std::uint32_t>
	{
		using type = category_basic_tag;
	};

	template<> struct category<std::uint64_t>
	{
		using type = category_basic_tag;
	};

	template<> struct category<float>
	{
		using type = category_basic_tag;
	};

	template<> struct category<double>
	{
		using type = category_basic_tag;
	};
}
