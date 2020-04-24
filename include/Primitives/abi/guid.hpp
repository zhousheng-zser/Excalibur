#pragma once

#include "meta_utils.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace glasssix::abi
{
	namespace details
	{
		struct size_offset
		{
			std::size_t offset;
			std::size_t size;

			constexpr std::size_t offset_end() const noexcept
			{
				return offset + size;
			}
		};
	}

	/// <summary>
	/// Defines a GUID.
	/// </summary>
	struct guid
	{
		std::uint32_t data1;
		std::uint16_t data2;
		std::uint16_t data3;
		std::array<std::uint8_t, 8> data4;

		constexpr guid() : data1{}, data2{}, data3{}, data4{}
		{
		}

		constexpr guid(std::uint32_t data1, std::uint16_t data2, std::uint16_t data3, const std::array<std::uint8_t, 8>& data4) : data1{ data1 }, data2{ data2 }, data3{ data3 }, data4{ data4 }
		{
		}

		constexpr guid(std::string_view str) : guid{}
		{
			constexpr std::size_t guid_string_size = 36;

			if (str.size() < guid_string_size)
			{
				return;
			}

			constexpr details::size_offset offset_data1{ 0, meta_utils::hexadecimal_character_size_v<std::uint32_t> };
			constexpr details::size_offset offset_data2{ offset_data1.offset_end() + 1, meta_utils::hexadecimal_character_size_v<std::uint16_t> };
			constexpr details::size_offset offset_data3{ offset_data2.offset_end() + 1,  meta_utils::hexadecimal_character_size_v<std::uint16_t> };
			constexpr details::size_offset offset_data4_first{ offset_data3.offset_end() + 1, meta_utils::hexadecimal_character_size_v<std::uint8_t> * 2 };
			constexpr details::size_offset offset_data4_second{ offset_data4_first.offset_end() + 1, meta_utils::hexadecimal_character_size_v<std::uint8_t> * 6 };

			data1 = meta_utils::to_number<std::uint32_t>(str.substr(offset_data1.offset, offset_data1.size));
			data2 = meta_utils::to_number<std::uint16_t>(str.substr(offset_data2.offset, offset_data2.size));
			data3 = meta_utils::to_number<std::uint16_t>(str.substr(offset_data3.offset, offset_data3.size));

			auto assign_to_data4 = [&](const details::size_offset& source_offset, std::size_t offset, std::size_t size)
			{
				for (std::size_t i = offset, j = source_offset.offset; i < offset + size; i++, j += meta_utils::hexadecimal_character_size_v<std::uint8_t>)
				{
					data4[i] = meta_utils::to_number<std::uint8_t>(str.substr(j, meta_utils::hexadecimal_character_size_v<std::uint8_t>));
				}
			};

			assign_to_data4(offset_data4_first, 0, 2);
			assign_to_data4(offset_data4_second, 2, 6);
		}

		constexpr bool operator==(const guid& right) const noexcept
		{
			// Both std::array<T>::operator== and std::equal are not constexpr functions.
			return data1 == right.data1 && data2 == right.data2 && data3 == right.data3 && [&]
			{
				for (auto left_ptr = data4.data(), right_ptr = right.data4.data(), end_ptr = data4.data() + data4.size(); left_ptr < end_ptr; left_ptr++, right_ptr++)
				{
					if (*left_ptr != *right_ptr)
					{
						return false;
					}
				}

				return true;
			}();
		}

		constexpr bool operator!=(const guid& right) const noexcept
		{
			return !(*this == right);
		}
	};
}
