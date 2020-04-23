#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

/// <summary>
/// This is something for future use, that is designed for ABI-independent invocations across DLL boundaries.
/// </summary>
namespace glasssix
{
	namespace library
	{
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

			constexpr guid(std::uint32_t data1, std::uint16_t data2, std::uint16_t data3, const std::array<std::uint8_t, 8>& data4): data1{ data1 }, data2{ data2 }, data3{ data3 }, data4{ data4 }
			{
			}

			constexpr bool operator==(const guid& right) const noexcept
			{
				// Both std::array<>::operator== and std::equal are not constexpr functions.
				return data1 == right.data1 && data2 == right.data2 && data3 == right.data3 && [&]
				{
					if (data4.size() != right.data4.size())
					{
						return false;
					}

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

		/// <summary>
		/// Defines the common base.
		/// </summary>
		struct unknown_object
		{
			unknown_object(std::nullptr_t)
			{
			}
		};

		namespace impl
		{
			template<typename Object, typename Enable = void>
			struct abi
			{
				using type = Object;
			};

			template<typename Object>
			using abi_t = typename abi<Object>::type;

			/// <summary>
			/// Specialization for enum type.
			/// </summary>
			template<typename Enum> struct abi<Enum, std::enable_if<std::is_enum_v<Enum>>>
			{
				using type = std::underlying_type_t<Enum>;
			};

			/// <summary>
			/// Specialization for the common base.
			/// </summary>
			template<> struct abi<unknown_object>
			{
				struct type
				{
					virtual bool query_interface(const guid& id, void** object) noexcept = 0;
					virtual std::size_t add_ref() noexcept = 0;
					virtual std::size_t release() noexcept = 0;
				};
			};

			using unknown_object = abi_t<unknown_object>;
		}
	}
}
