#pragma once

#include "meta.hpp"
#include "base.hpp"
#include "base_abi.hpp"

#include <tuple>
#include <array>
#include <atomic>
#include <cstddef>
#include <utility>
#include <algorithm>
#include <type_traits>

namespace glasssix::exposing::impl
{
	template <typename Derived, typename Interface>
	class producer;

	namespace details
	{
		/// <summary>
		/// Produces an implementation for an interface ABI.
		/// </summary>
		template <typename Derived, typename Interface, typename Enable = void>
		struct produce_for
		{
			static_assert(bool{}, "The interface must be derived from glasssix::exposing::unknown_object.");
		};

		/// <summary>
		/// Produces an implementation for an interfacial ABI, which forwards all calls to the derived type.
		/// </summary>
		template <typename Derived, typename Interface>
		struct produce_for<Derived, Interface, std::enable_if_t<is_derived_from_unknown_object_v<Interface>>> : abi_t<Interface>
		{
			Derived& self() noexcept
			{
				return static_cast<Derived&>(meta::get_standard_layout_from_first_member<producer<Derived, Interface>>(*this));
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
	}

	/// <summary>
	/// A reference to a procuded implementaion.
	/// </summary>
	template <typename Interface>
	struct produced_ref : Interface
	{
		produced_ref(void* ptr) noexcept : Interface{ nullptr }
		{
			*put_abi(*this) = ptr;
		}

		~produced_ref() noexcept
		{
			detach_abi(*this);
		}

		produced_ref(const produced_ref&) = delete;
		produced_ref(produced_ref&&) = delete;
		produced_ref& operator=(const produced_ref&) = delete;
		produced_ref& operator=(produced_ref&&) = delete;
		void* operator new(std::size_t) = delete;
	};

	/// <summary>
	/// A class that contains a produced implementation for an interfacial ABI.
	/// </summary>
	template <typename Derived, typename Interface>
	class producer
	{
	public:
		friend details::produce_for<Derived, Interface>;

		operator produced_ref<Interface> const() const noexcept
		{
			return const_cast<produce_for<Derived, Interface>*>(&vtable_);
		}
	private:
		details::produce_for<Derived, Interface> vtable_;
	};

	namespace details
	{
		template <typename Derived, typename T>
		struct producers_impl;

		template <typename Derived, typename... Interfaces>
		struct producers_impl<Derived, std::tuple<Interfaces...>> : producer<Derived, Interfaces>...
		{
		};

		template<typename Derived>
		struct pack_implemented_interfaces;

		template<template<typename, typename...> typename Implements, typename Derived, typename... Interfaces>
		struct pack_implemented_interfaces<Implements<Derived, Interfaces...>>
		{
			using type = meta::tuple_if_t<is_derived_from_unknown_object, Interfaces...>;
		};
	}

	template<typename, typename = void>
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
	template<typename Interface, typename Derived>
	inline constexpr auto to_abi(Derived&& derived) noexcept -> std::enable_if_t<std::conjunction_v<std::is_lvalue_reference<Derived>, is_implements<Derived>, is_derived_from_unknown_object<Interface>>, abi_t<Interface>*>
	{
		using producer_type = producer<Derived, Interface>;
		using producer_ref_type = std::conditional_t<std::is_const_v<std::decay_t<derived>>, std::add_const_t<producer_type>&, producer_type&>;
		
		return meta::get_standard_layout_first_member<abi_t<Interface>*>(static_cast<producer_ref_type>(derived));
	}

	/// <summary>
	/// A class that contains one or more procuded implementations for interfacial ABIs.
	/// </summary>
	template<typename Derived, typename... Interfaces>
	using producers = details::producers_impl<Derived, meta::tuple_if_t<is_derived_from_unknown_object, std::tuple<Interfaces...>>>;

	template<typename Derived, typename Enable = void>
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

	template<typename Derived, typename Enable = void>
	class find_interface_by_guid;

	/// <summary>
	/// Finds an interface ABI by GUID.
	/// </summary>
	template<typename Derived>
	class find_interface_by_guid<Derived, std::enable_if_t<is_implements_v<Derived>>>
	{
	public:
		static auto get(Derived& derived, const guid& id) const noexcept
		{
			using packed_type = typename details::pack_implemented_interfaces<typename Derived::implements_type>::type;

			return get_impl(derived, id, std::make_index_sequence<std::tuple_size_v<packed_type>>);
		}
	private:
		template<typename Packed, std::size_t... Indexes>
		static auto get_impl(Derived& derived, const guid& id, std::index_sequence<Indexes...>) const noexcept
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

			auto iter = std::find_if(results.begin(), results.end(), [](exposing::unknown_object* inner) -> bool { return inner; });

			return iter != results.end() ? *iter : nullptr;
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

				*object = static_cast<impl::unknown_object*>(to_abi<first_interface_type>(*this));
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
