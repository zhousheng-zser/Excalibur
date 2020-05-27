#pragma once

#include "meta.hpp"
#include "base.hpp"
#include "base_abi.hpp"
#include "g6_attributes.hpp"

#include <tuple>
#include <array>
#include <atomic>
#include <cstddef>
#include <utility>
#include <algorithm>
#include <type_traits>

namespace glasssix::exposing::impl
{
	template<typename Derived, typename Interface>
	class implements_interface_vtable;

	template <typename Derived, typename Interface, typename = void>
	struct interface_vtable_base;

	/// <summary>
	/// The unknown_object vtable of an interface ABI, which forwards all calls to the derived type.
	/// </summary>
	template <typename Derived, typename Interface>
	struct interface_vtable_base<Derived, Interface, std::enable_if_t<is_derived_from_unknown_object_v<Interface>>> : abi_t<Interface>
	{
		Derived& self() noexcept
		{
			return static_cast<Derived&>(meta::get_standard_layout_from_first_member<implements_interface_vtable<Derived, Interface>>(*this));
		}

		virtual bool G6_ABI_CALL query_interface(const guid& id, void** object) noexcept override
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
	template <typename Derived, typename Interface>
	struct interface_vtable : interface_vtable_base<Derived, Interface>
	{
	};

	/// <summary>
	/// A reference to a interfacial vtable.
	/// </summary>
	template <typename Interface>
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

		interface_vtable_ref(const interface_vtable_ref&) = delete;
		interface_vtable_ref(interface_vtable_ref&&) = delete;
		interface_vtable_ref& operator=(const interface_vtable_ref&) = delete;
		interface_vtable_ref& operator=(interface_vtable_ref&&) = delete;
		void* operator new(std::size_t) = delete;
	};

	/// <summary>
	/// Implements a vtable for an interfacial ABI.
	/// </summary>
	template <typename Derived, typename Interface>
	class implements_interface_vtable
	{
	public:
		friend interface_vtable<Derived, Interface>;

		operator interface_vtable_ref<Interface>() const noexcept
		{
			return const_cast<interface_vtable<Derived, Interface>*>(&vtable_);
		}
	private:
		interface_vtable<Derived, Interface> vtable_;
	};

	namespace details
	{
		template <typename Derived, typename T>
		struct implements_interface_vtables_impl;

		template <typename Derived, typename... Interfaces>
		struct implements_interface_vtables_impl<Derived, std::tuple<Interfaces...>> : implements_interface_vtable<Derived, Interfaces>...
		{
		};

		template<typename Derived>
		struct pack_implemented_interfaces;

		template<template<typename, typename...> typename Implements, typename Derived, typename... Interfaces>
		struct pack_implemented_interfaces<Implements<Derived, Interfaces...>>
		{
			using type = meta::tuple_if_t<is_derived_from_unknown_object, std::tuple<Interfaces...>>;
		};
	}

	template<typename T, typename = void>
	struct is_implements : std::false_type {};

	template<typename T>
	struct is_implements<T, std::void_t<typename T::implements_type>> : std::true_type {};

	template<typename T>
	inline constexpr bool is_implements_v = is_implements<T>::value;

	/// <summary>
	/// Retrieves an interfacial ABI from a derived object.
	/// </summary>
	/// <typeparam name="Interface">The interface type</typeparam>
	/// <typeparam name="Derived">The derived type</typeparam>
	/// <param name="derived">The derived object</param>
	/// <returns>The ABI</returns>
	template<typename Interface, typename Derived, typename = std::enable_if_t<std::conjunction_v<is_implements<Derived>, is_derived_from_unknown_object<Interface>>>>
	constexpr auto to_abi(const Derived* derived) noexcept
	{
		return static_cast<abi_t<Interface>*>(get_abi(&derived));
	}

	/// <summary>
	/// Implements vtables for interfacial ABIs.
	/// </summary>
	template<typename Derived, typename... Interfaces>
	using implements_interface_vtables = details::implements_interface_vtables_impl<Derived, meta::tuple_if_t<is_derived_from_unknown_object, std::tuple<Interfaces...>>>;

	template<typename Derived, typename = void>
	struct first_interface;

	template<typename Derived>
	struct first_interface<Derived, std::enable_if_t<is_implements_v<Derived>>>
	{
		using type = meta::tuple_first_t<typename details::pack_implemented_interfaces<typename Derived::implements_type>::type>;
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
		static auto get(Derived& derived, const guid& id) noexcept
		{
			using packed_type = typename details::pack_implemented_interfaces<typename Derived::implements_type>::type;

			return get_impl(derived, id, std::make_index_sequence<std::tuple_size_v<packed_type>>);
		}
	private:
		template<typename Packed, std::size_t... Indexes>
		static auto get_impl(Derived& derived, const guid& id, std::index_sequence<Indexes...>) noexcept
		{
			std::array<void*, std::tuple_size_v<Packed>> results =
			{
				[&]
				{
					using implements_type = typename Derived::implements_type;
					using interface_type = std::tuple_element_t<Indexes, Packed>;
					constexpr auto interface_id = guid_of_v<interface_type>;

					return interface_id == id ? to_abi<interface_type>(&derived) : nullptr;
				}()...
			};

			auto iter = std::find_if(results.begin(), results.end(), [](exposing::unknown_object* inner) -> bool { return inner; });

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
	template<typename Derived, typename Interface, typename = std::enable_if_t<has_abi_type_v<Interface>>>
	struct enable_self_abi_awareness
	{
		decltype(auto) self_abi() const noexcept
		{
			return *static_cast<abi_t<Interface>*>(get_abi(static_cast<const Interface&>(static_cast<const Derived&>(*this))));
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
		unknown_object_impl() : reference_count_{ 1 }
		{
		}

		bool G6_ABI_CALL query_interface(const guid& id, void** object) noexcept
		{
			if (object == nullptr)
			{
				return false;
			}

 			if (*object = find_interface_by_guid<Derived>::get(*this, id))
			{
				add_ref();

				return true;
			}

			// Provides the implementation of the first interface.
			if (is_guid_of<exposing::unknown_object>(id))
			{
				using first_interface_type = first_interface_t<Derived>;

				*object = static_cast<impl::abi_unknown_object*>(to_abi<first_interface_type>(this));
				add_ref();

				return true;
			}
		}

		std::uint32_t G6_ABI_CALL add_ref() noexcept
		{
			// A relaxed memory order results in improved efficiency.
			return reference_count_.fetch_add(1, std::memory_order_relaxed) + 1;
		}

		std::uint32_t G6_ABI_CALL release() noexcept
		{
			if (std::uint32_t count = reference_count_.fetch_sub(1, std::memory_order_release) - 1; count == 0)
			{
				// Ensures serialized running.
				std::atomic_thread_fence(std::memory_order_acquire);
				delete this;
			}
		}
	private:
		std::atomic_uint32_t reference_count_;
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

		bool G6_ABI_CALL query_interface(const guid& id, void** object) noexcept
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
	};
}
