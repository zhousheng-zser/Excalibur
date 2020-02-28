#pragma once

#include <array>
#include <cstdint>

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
			std::uint8_t data4[8];

			guid() noexcept = default;

			constexpr guid(std::uint32_t const data1, std::uint16_t const data2, std::uint16_t const data3, const std::array<std::uint8_t, 8>& data4) noexcept : data1{ data1 }, data2{ data2 }, data3{ data3 }, data4{ data4[0], data4[1], data4[2], data4[3], data4[4], data4[5], data4[6], data4[7] }
			{
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
			template <typename Object>
			struct abi
			{
				using type = Object;
			};

			template <typename Object>
			using abi_t = typename abi<Object>::type;

			/// <summary>
			/// Defines the common base.
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
