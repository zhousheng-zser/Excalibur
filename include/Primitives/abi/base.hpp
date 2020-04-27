#pragma once

#include "guid.hpp"
#include "meta.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#ifdef _WIN32
#define G6_ABI_CALL __stdcall
#else
#define G6_ABI_CALL
#endif

#if defined(__has_cpp_attribute) &&  __has_cpp_attribute(no_unique_address)
#define G6_EBO [[no_unique_address]]
#elif defined(_MSC_VER)
#define G6_EBO __declspec(empty_bases)
#else
#define G6_EBO
#endif

#ifdef _MSC_VER
#define G6_NOVTABLE __declspec(novtable)
#else
#define G6_NOVTABLE
#endif

namespace glasssix::exposing::impl
{
	template<typename Derived>
	struct consume;

	template<typename Derived, typename Interface = Derived>
	using consume_t = typename consume<Interface>::template type<Derived>;

	template<typename T>
	struct delegate;

	template<typename T, typename Nested>
	using delegate_t = typename delegate<T>::template type<Nested>;

	template<typename T>
	struct category
	{
		using type = void;
	};

	template<typename T>
	using category_t = typename category<T>::type;

	/// <summary>
	/// Checks whether the type is contained by some category.
	/// </summary>
	template<typename T>
	inline constexpr bool has_category_v = !std::is_same_v<category_t<T>, void>;

	/// <summary>
	/// C++ basic types.
	/// </summary>
	struct basic_category;

	/// <summary>
	/// Public interfaces.
	/// </summary>
	struct interface_category;

	/// <summary>
	/// Delegates.
	/// </summary>
	struct delegate_category;

	/// <summary>
	/// Enumerations.
	/// </summary>
	struct enum_category;

	/// <summary>
	/// Classes.
	/// </summary>
	struct class_category;

	/// <summary>
	/// Generic public interfaces.
	/// </summary>
	template<typename... Args>
	struct generic_interface_category;

	/// <summary>
	/// The signature of a category.
	/// </summary>
	template <typename Category, typename T>
	struct category_signature;

	/// <summary>
	/// The signature of a type.
	/// </summary>
	template <typename T>
	struct signature
	{
		static constexpr auto value{ category_signature<typename category<T>::type, T>::value };
	};

	template<typename T>
	struct guid_storage
	{
		static_assert(bool{}, "Support for ordinary C++ types is disabled.");
	};

	template<typename, typename = void>
	struct is_implements : std::false_type {};

	template<typename T>
	struct is_implements<T, std::void_t<typename T::implements_type>> : std::true_type {};

	template<typename T>
	inline constexpr bool is_implements_v = is_implements<T>::value;

	template<typename Derived, typename Interface>
	struct require_one : consume_t<Derived, Interface>
	{
		operator Interface() const noexcept
		{
			return static_cast<const Derived*>(this)->template try_as<I>();
		}
	};

	template<typename Derived, typename... Interfaces>
	struct G6_EBO require : require_one<Derived, Interfaces>...
	{
	};

	template<typename T>
	struct name
	{
		static constexpr auto value{};
	};

	template <typename T>
	inline constexpr auto& name_v = name<T>::value;
}

namespace glasssix::exposing
{
	template<typename T>
	inline constexpr auto guid_of_v = impl::guid_storage<T>::value;
}
