#pragma once

#include "meta_utils.hpp"

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

/// <summary>
/// This is something for future use, that is designed for ABI-independent invocations across DLL boundaries.
/// </summary>
namespace glasssix::abi
{
	namespace impl
	{
		template<typename T, typename Enable = void>
		struct abi
		{
			using type = T;
		};

		template<typename T>
		using abi_t = typename abi<T>::type;

		template<typename T>
		struct consume;

		template<typename T, typename Nested = T>
		using consume_t = typename consume<Nested>::template type<T>;

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
			static_assert(false, "Support for ordinary C++ types is disabled.");
		};

		template<typename T>
		struct name
		{
			static constexpr auto value{};
		};

		template <typename T>
		inline constexpr auto& name_v = name<T>::value;
	}
}
