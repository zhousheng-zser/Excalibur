#pragma once

#include "meta.hpp"
#include "base.hpp"
#include "base_abi.hpp"
#include "param_string.hpp"
#include "g6_attributes.hpp"
#include "fundamental_semantics.hpp"

#include <new>
#include <tuple>
#include <array>
#include <atomic>
#include <memory>
#include <cstddef>
#include <utility>
#include <algorithm>
#include <type_traits>
#include <string_view>

#define META_STR(x) decltype(struct { static constexpr glasssix::exposing::utf8_string_view value = x; }{})::value

namespace glasssix::exposing::impl
{
	template<typename Derived, typename Interface>
	class implements_interface_vtable;

	template<typename Derived, typename Interface, typename = void>
	struct interface_vtable_base;

	namespace details
	{
		template<typename Derived, typename T>
		struct implements_interface_vtables_impl;

		template<typename Derived, typename... Interfaces>
		struct implements_interface_vtables_impl<Derived, std::tuple<Interfaces...>> : implements_interface_vtable<Derived, Interfaces>...
		{
		};

		template<typename Derived>
		struct pack_implemented_interfaces;

		template<template<typename, typename...> typename Implements, typename Derived, typename... Interfaces>
		struct pack_implemented_interfaces<Implements<Derived, Interfaces...>>
		{
			using type = meta::tuple_if_t<is_well_defined_interface, std::tuple<Interfaces...>>;
		};

		template<typename Derived>
		using pack_implemented_interfaces_t = typename pack_implemented_interfaces<Derived>::type;
	}

	template<typename T, typename = void>
	struct is_implements : std::false_type {};

	template<typename T>
	struct is_implements<T, std::void_t<typename T::implements_type>> : std::true_type {};

	/// <summary>
	/// Checks whether a type is an "implements" type.
	/// </summary>
	/// <typeparam name="T">The type</typeparam>
	template<typename T>
	inline constexpr bool is_implements_v = is_implements<T>::value;

	template<typename T, typename = void>
	struct has_unique_release : std::false_type {};

	template<typename T>
	struct has_unique_release < T, std::void_t<decltype(T::unique_release(std::unique_ptr<T>{})) >> : std::true_type{};

	/// <summary>
	/// Checks whether a type contains a static function named unique_release.
	/// </summary>
	/// <typeparam name="T">The type</typeparam>
	template<typename T>
	inline constexpr bool has_unique_release_v = has_unique_release<T>::value;

	template<typename T, typename = void>
	struct has_external_qualified_name : std::false_type {};

	template<typename T>
	struct has_external_qualified_name<T, std::void_t<decltype(T::external_qualified_name_type::value)>> : std::true_type {};

	/// <summary>
	/// Checks whether a type contains an external qualified name.
	/// </summary>
	/// <typeparam name="T">The type</typeparam>
	template<typename T>
	inline constexpr auto& has_external_qualified_name_v = has_external_qualified_name<T>::value;

	template<typename T, typename = void>
	struct get_external_qualified_name;

	template<typename T>
	struct get_external_qualified_name<T, std::enable_if_t<has_external_qualified_name_v<T>>>
	{
		static constexpr auto& value = T::external_qualified_name_type::value;
	};

	/// <summary>
	/// Retrieves the external qualified name of a type if exists.
	/// </summary>
	/// <typeparam name="T">The type</typeparam>
	template<typename T>
	inline constexpr auto& get_external_qualified_name_v = get_external_qualified_name<T>::value;

	/// <summary>
	/// The unknown_object vtable of an interface ABI, which forwards all calls to the derived type.
	/// </summary>
	template<typename Derived, typename Interface>
	struct interface_vtable_base<Derived, Interface, std::enable_if_t<is_well_defined_interface_v<Interface>>> : abi_t<Interface>, std::tuple<void*>
	{
		Derived& self() noexcept
		{
			// For C++ conformance, put workaround here to avoid reinterpret_cast between non-standard-layout objects.
			return static_cast<Derived&>(*static_cast<implements_interface_vtable<Derived, Interface>*>(std::get<void*>(*this)));
		}

		virtual std::int32_t G6_ABI_CALL query_interface(guid id, void** object) noexcept override
		{
			return self().query_interface(id, object);
		}

		virtual std::uint32_t G6_ABI_CALL add_ref() noexcept override
		{
			return self().add_ref();
		}

		virtual std::uint32_t G6_ABI_CALL release() noexcept override
		{
			return self().release();
		}
	};

	/// <summary>
	/// The vtable of an interface ABI, which forwards all calls to the derived type.
	/// </summary>
	template<typename Derived, typename Interface>
	struct interface_vtable : interface_vtable_base<Derived, Interface>
	{
	};

	/// <summary>
	/// A reference to a interfacial vtable.
	/// </summary>
	template<typename Interface>
	struct interface_vtable_ref : Interface
	{
		interface_vtable_ref(void* ptr) noexcept : Interface{ nullptr }
		{
			*put_abi(*this) = ptr;
		}

		~interface_vtable_ref() noexcept
		{
			detach_abi(*this);
		}

		interface_vtable_ref(const interface_vtable_ref&) noexcept = delete;
		interface_vtable_ref(interface_vtable_ref&&) noexcept = delete;
		interface_vtable_ref& operator=(const interface_vtable_ref&) noexcept = delete;
		interface_vtable_ref& operator=(interface_vtable_ref&&) noexcept = delete;
		void* operator new(std::size_t) = delete;
	};

	/// <summary>
	/// Implements a vtable for an interfacial ABI.
	/// </summary>
	template<typename Derived, typename Interface>
	class implements_interface_vtable
	{
	public:
		using vtable_type = interface_vtable<Derived, Interface>;
		friend vtable_type;

		template<typename Interface, typename Derived, typename>
		friend constexpr auto to_abi(Derived& derived) noexcept;

		implements_interface_vtable() noexcept
		{
			std::get<void*>(vtable_) = this;
		}

		/// <summary>
		/// Converts to one of the implemented interfaces.
		/// </summary>
		/// <returns>The interface</returns>
		operator interface_vtable_ref<Interface>() const noexcept
		{
			return interface_vtable_ref<Interface>{ const_cast<vtable_type*>(&vtable_) };
		}
	private:
		vtable_type vtable_;
	};

	/// <summary>
	/// Retrieves an interfacial ABI from a derived object.
	/// </summary>
	/// <typeparam name="Interface">The interfacial type</typeparam>
	/// <typeparam name="Derived">The derived type</typeparam>
	/// <param name="derived">The derived object</param>
	/// <returns>The ABI</returns>
	template<typename Interface, typename Derived, typename = std::enable_if_t<std::conjunction_v<is_implements<Derived>, is_well_defined_interface<Interface>>>>
	constexpr auto to_abi(Derived& derived) noexcept
	{
		return static_cast<abi_t<Interface>*>(&static_cast<implements_interface_vtable<Derived, Interface>&>(derived).vtable_);
	}

	/// <summary>
	/// Implements vtables for interfacial ABIs.
	/// </summary>
	template<typename Derived, typename... Interfaces>
	using implements_interface_vtables = details::implements_interface_vtables_impl<Derived, meta::tuple_if_t<is_well_defined_interface, std::tuple<Interfaces...>>>;

	template<typename Derived, typename = void>
	struct first_interface;

	template<typename Derived>
	struct first_interface<Derived, std::enable_if_t<is_implements_v<Derived>>>
	{
		using type = meta::tuple_first_t<details::pack_implemented_interfaces_t<typename Derived::implements_type>>;
	};

	/// <summary>
	/// Gets the first interface derived from glasssix::exposing::unknown_object.
	/// </summary>
	template<typename Derived>
	using first_interface_t = typename first_interface<Derived>::type;

	template<typename Derived, typename = void>
	class find_interface_by_guid;

	/// <summary>
	/// Finds an interface ABI by GUID.
	/// </summary>
	template<typename Derived>
	class find_interface_by_guid<Derived, std::enable_if_t<is_implements_v<Derived>>>
	{
	public:
		/// <summary>
		/// Gets the ABI of an interface by specified ID.
		/// </summary>
		/// <param name="derived">The derived object</param>
		/// <param name="id">The ID</param>
		/// <returns>The ABI</returns>
		static void* get(Derived& derived, const guid& id) noexcept
		{
			using packed_type = details::pack_implemented_interfaces_t<typename Derived::implements_type>;

			return get_impl<packed_type>(derived, id, std::make_index_sequence<std::tuple_size_v<packed_type>>{});
		}
	private:
		template<typename Packed, std::size_t... Indexes>
		static void* get_impl(Derived& derived, const guid& id, std::index_sequence<Indexes...>) noexcept
		{
			std::array<void*, std::tuple_size_v<Packed>> results =
			{
				[&]
				{
					using implements_type = typename Derived::implements_type;
					using interface_type = std::tuple_element_t<Indexes, Packed>;
					constexpr auto interface_id = guid_of_v<interface_type>;

					return interface_id == id ? to_abi<interface_type>(derived) : nullptr;
				}()...
			};

			auto iter = std::find_if(results.begin(), results.end(), [](void* inner) -> bool { return inner; });

			return iter != results.end() ? *iter : nullptr;
		}
	};

	template<typename Interface, typename = std::enable_if_t<has_abi_type_v<Interface>>>
	struct abi_adapter;

	/// <summary>
	/// The ABI adapter for an interface.
	/// </summary>
	template<typename Derived, typename Interface>
	using abi_adapter_t = typename abi_adapter<Interface>::template type<Derived>;

	/// <summary>
	/// A helper class that enables casting to the derived type of an interface.
	/// </summary>
	template<typename Interface, typename = std::enable_if_t<has_abi_type_v<Interface>>>
	struct enable_self_abi_awareness
	{
		decltype(auto) self_abi() const noexcept
		{
			return *static_cast<abi_t<Interface>*>(get_abi(static_cast<const Interface&>(*this)));
		}
	};

	/// <summary>
	/// Inherits a ABI adapter for an interface.
	/// </summary>
	template<typename Derived, typename Interface>
	struct inherits_abi_adapter : abi_adapter_t<Derived, Interface>
	{
		operator Interface() const noexcept
		{
			return static_cast<const Derived&>(*this).template try_as<Interface>();
		}
	};

	/// <summary>
	/// A class that implements the fundamental functions of glasssix::exposing::unknown_object.
	/// </summary>
	template<typename Derived>
	class G6_NOVTABLE unknown_object_impl
	{
	public:
		unknown_object_impl() noexcept : ref_count_{ 1 }
		{
			++get_module_ref_count();
		}

		virtual ~unknown_object_impl() noexcept
		{
			--get_module_ref_count();
		}

		std::int32_t G6_ABI_CALL query_interface(const guid& id, void** object) noexcept
		{
			if (object == nullptr)
			{
				return error_null_pointer;
			}

			if (*object = find_interface_by_guid<Derived>::get(static_cast<Derived&>(*this), id))
			{
				add_ref();

				return error_success;
			}

			// Provides the implementation of the first interface.
			if (is_guid_of<exposing::unknown_object>(id))
			{
				using first_interface_type = first_interface_t<Derived>;

				*object = static_cast<impl::abi_unknown_object*>(to_abi<first_interface_type>(static_cast<Derived&>(*this)));
				add_ref();

				return error_success;
			}

			return error_no_interface;
		}

		std::uint32_t G6_ABI_CALL add_ref() noexcept
		{
			return ++ref_count_;
		}

		std::uint32_t G6_ABI_CALL release() noexcept
		{
			std::uint32_t count = --ref_count_;

			if (count == 0)
			{
				// Call the unique_release routine for deferred operations if the derived type defines it.
				if constexpr (has_unique_release_v<Derived>)
				{
					ref_count_ = 1;
					Derived::unique_release(std::unique_ptr<Derived>{ static_cast<Derived*>(this) });
				}
				else
				{
					delete this;
				}
			}

			return count;
		}
	private:
		atomic_ref_count ref_count_;
	};
}

namespace glasssix::exposing
{
	/// <summary>
	/// A helper class to generate standard ABI implementations for a derived class.
	/// </summary>
	template<typename Derived, typename... Interfaces>
	struct implements : impl::implements_interface_vtables<Derived, Interfaces...>, impl::unknown_object_impl<Derived>
	{
		using implements_type = implements;
		using root_implements_type = impl::unknown_object_impl<Derived>;

		abi_result G6_ABI_CALL query_interface(const guid& id, void** object) noexcept
		{
			return root_implements_type::query_interface(id, object);
		}

		std::uint32_t G6_ABI_CALL add_ref() noexcept
		{
			return root_implements_type::add_ref();
		}

		std::uint32_t G6_ABI_CALL release() noexcept
		{
			return root_implements_type::release();
		}
	};

	/// <summary>
	/// A helper class to support implicitly casting to one or more interfaces.
	/// </summary>
	template<typename Derived, typename... Interfaces>
	struct inherits : unknown_object, impl::abi_adapter_t<Derived, Derived>, impl::inherits_abi_adapter<Derived, Interfaces>...
	{
		/// <summary>
		/// Creates an instance with nullptr.
		/// </summary>
		inherits(std::nullptr_t = nullptr) noexcept : unknown_object{ nullptr }
		{
		}

		/// <summary>
		/// Create an instance with an ABI from which ownership is taken.
		/// </summary>
		/// <param name="abi">The ABI</param>
		inherits(take_over_abi_from_void_ptr abi) noexcept : unknown_object{ abi }
		{
		}
	};

	/// <summary>
	/// A helper class that makes the external qualified name of an implementation.
	/// </summary>
	template<const utf8_string_view& name>
	struct make_external_qualified_name
	{
		struct external_qualified_name_type
		{
			static constexpr utf8_string_view value = name;
		};
	};

	/// <summary>
	/// Creates an in-process instance of an implementation type and returns the default interface.
	/// </summary>
	/// <typeparam name="T">The implementation type</typeparam>
	/// <typeparam name="Interface">The interfacial type</typeparam>
	/// <typeparam name="...Args">The types of arguments</typeparam>
	/// <param name="...args">The arguments</param>
	/// <returns>The instance</returns>
	template<typename T, typename Interface, typename... Args, typename = std::enable_if_t<std::conjunction_v<impl::is_implements<T>, impl::is_well_defined_interface<Interface>>>>
	auto make(Args&&... args)
	{
		return Interface{ take_over_abi_from_void_ptr{ impl::to_abi<Interface>(*new T(std::forward<Args>(args)...)) } };
	}

	/// <summary>
	/// Creates an in-process instance of an implementation type and returns the first interface.
	/// </summary>
	/// <typeparam name="T">The implementation type</typeparam>
	/// <typeparam name="...Args">The types of arguments</typeparam>
	/// <param name="...args">The arguments</param>
	/// <returns>The instance</returns>
	template<typename T, typename... Args, typename = std::enable_if_t<impl::is_implements_v<T>>>
	auto make_as_first(Args&&... args)
	{
		return make<T, impl::first_interface_t<T>>(std::forward<Args>(args)...);
	}

	template<typename Interface, typename = std::enable_if_t<impl::is_well_defined_interface_v<Interface>>>
	auto make_from_dll(utf8_string_view path)
	{

	}
}
