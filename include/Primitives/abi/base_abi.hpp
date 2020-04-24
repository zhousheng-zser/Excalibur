#pragma once

#include "base.hpp"

namespace glasssix::abi::impl
{
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
