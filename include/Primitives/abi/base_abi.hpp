#pragma once

#include "base.hpp"
#include "guid.hpp"
#include "meta.hpp"
#include "exceptions.hpp"

#include <cstddef>
#include <utility>
#include <type_traits>

namespace glasssix::exposing
{
	class unknown_object;

	template<typename T, typename>
	void* detach_abi(T&& object) noexcept;
	void* get_abi(const unknown_object& object) noexcept;
	void** put_abi(unknown_object& object) noexcept;
	void attach_abi(unknown_object& object, void* value) noexcept;
	void copy_from_abi(unknown_object& object, void* value) noexcept;
	void copy_to_abi(const unknown_object& object, void*& value) noexcept;
}

namespace glasssix::exposing::impl
{
	template<typename To, typename From, typename>
	To as(From* ptr);

	template<typename To, typename From, typename>
	To try_as(From* ptr);

	template<typename T, typename Enable = void>
	struct abi
	{
		using type = T;
	};

	template<typename T>
	using abi_t = typename abi<T>::type;

	/// <summary>
	/// Stores the GUID of a ABI.
	/// </summary>
	template<typename T>
	struct guid_storage<T, std::void_t<decltype(abi<T>::id)>>
	{
		static constexpr auto& value{ abi<T>::id };
	};

	/// <summary>
	/// Stores the GUID of a generic interface ABI.
	/// </summary>
	template<template<typename...> typename T, typename... Args>
	struct guid_storage<T<Args...>, std::void_t<decltype(abi<T<Args...>>::id)>>
	{
		static constexpr auto value{ create_guid_from_bytes(meta::concat_arrays(to_array(abi<T<Args...>>::id), type_signature_v<Args>...)) };
	};

	/// <summary>
	/// The type identity of a ABI.
	/// </summary>
	template<typename T>
	struct type_identity<T, std::void_t<typename abi<T>::identity_type>>
	{
		using type = typename abi<T>::identity_type;
	};

	/// <summary>
	/// The root interface ABI.
	/// </summary>
	template<> struct abi<exposing::unknown_object>
	{
		using identity_type = type_identity_interface;
		static constexpr guid id{ "00000000-0000-0000-C000-000000000046" };

		struct type
		{
			virtual bool G6_ABI_CALL query_interface(const guid& id, void** object) noexcept = 0;
			virtual std::uint32_t G6_ABI_CALL add_ref() noexcept = 0;
			virtual std::uint32_t G6_ABI_CALL release() noexcept = 0;
		};
	};

	using abi_unknown_object = abi_t<exposing::unknown_object>;

	/// <summary>
	/// Checks whether a type is an ABI interface.
	/// </summary>
	template<typename T>
	struct is_derived_from_unknown_object : std::bool_constant<std::conjunction_v<std::is_base_of<unknown_object, T>, std::is_standard_layout<T>>>
	{
	};

	template<typename T>
	inline constexpr bool is_derived_from_unknown_object_v = is_derived_from_unknown_object<T>::value;
}

namespace glasssix::exposing
{
	/// <summary>
	/// A fundamental wrapper for the underlying ABI.
	/// </summary>
	class unknown_object
	{
	public:
		/// <summary>
		///  Creates an instance.
		/// </summary>
		unknown_object() noexcept : abi_{}
		{
		}

		/// <summary>
		/// Create an instance with an ABI from which ownership is taken.
		/// </summary>
		/// <param name="abi">The ABI</param>
		unknown_object(void* abi) noexcept : abi_{ static_cast<impl::abi_unknown_object*>(abi) }
		{
		}

		/// <summary>
		/// Creates an instance with nullptr.
		/// </summary>
		unknown_object(std::nullptr_t) noexcept : unknown_object{}
		{
		}

		unknown_object(const unknown_object& other) noexcept : abi_{ other.abi_ }
		{
			add_ref();
		}

		unknown_object(unknown_object&& other) noexcept : abi_{ std::exchange(other.abi_, nullptr) }
		{
		}

		~unknown_object() noexcept
		{
		}

		unknown_object& operator=(const unknown_object& right) noexcept
		{
			if (this != &right)
			{
				release();
				abi_ = right.abi_;
				add_ref();
			}

			return *this;
		}

		unknown_object& operator=(unknown_object&& right) noexcept
		{
			if (this != &right)
			{
				release();
				abi_ = std::exchange(right.abi_, nullptr);
			}

			return *this;
		}

		unknown_object& operator=(std::nullptr_t) noexcept
		{
			release();

			return *this;
		}

		/// <summary>
		/// Disables allocations on the heap.
		/// </summary>
		void* operator new(std::size_t) = delete;

		/// <summary>
		/// Indicates whether the ABI is valid.
		/// </summary>
		/// <returns>True if the ABI is valid; otherwise false</returns>
		explicit operator bool() const noexcept
		{
			return abi_;
		}

		/// <summary>
		/// Converts to another implemented interface.
		/// </summary>
		/// <typeparam name="To">The destination type</typeparam>
		/// <returns>The result</returns>
		template<typename To>
		auto as() const
		{
			return impl::as<To>(abi_);
		}

		/// <summary>
		/// Converts to another implemented interface without any exceptions to be thrown.
		/// </summary>
		/// <typeparam name="To">The destination type</typeparam>
		/// <returns>The result</returns>
		template<typename To>
		auto try_as() const noexcept
		{
			return impl::try_as<To>(abi_);
		}

		/// <summary>
		/// Swaps the ABIs between two objects.
		/// </summary>
		/// <param name="left">The left value</param>
		/// <param name="right">The right value</param>
		friend void swap(unknown_object& left, unknown_object& right) noexcept
		{
			std::swap(left.abi_, right.abi_);
		}
	private:
		void add_ref() const noexcept
		{
			if (abi_)
			{
				abi_->add_ref();
			}
		}

		void release() noexcept
		{
			if (abi_)
			{
				std::exchange(abi_, nullptr)->release();
			}
		}

		impl::abi_unknown_object* abi_;
	};

	/// <summary>
	/// Gets the ABI of an object with type information erased.
	/// </summary>
	/// <param name="object">The object</param>
	/// <returns>The ABI</returns>
	inline void* get_abi(const unknown_object& object) noexcept
	{
		return meta::get_standard_layout_first_member<impl::abi_unknown_object*>(object);
	}

	/// <summary>
	/// Gets a pointer to the ABI of an object with type information erased.
	/// </summary>
	/// <param name="object">The object</param>
	/// <returns>The pointer to the ABI</returns>
	inline void** put_abi(unknown_object& object) noexcept
	{
		object = nullptr;

		return reinterpret_cast<void**>(&meta::get_standard_layout_first_member<impl::abi_unknown_object*>(object));
	}

	/// <summary>
	/// Attaches an ABI to an object.
	/// </summary>
	/// <param name="object">The object</param>
	/// <param name="abi">The ABI</param>
	inline void attach_abi(unknown_object& object, void* abi) noexcept
	{
		*put_abi(object) = abi;
	}

	/// <summary>
	/// Detaches the ABI from an object.
	/// </summary>
	/// <param name="object">The object</param>
	/// <returns>The ABI detached from the object</returns>
	template<typename T, typename = std::enable_if_t<std::disjunction_v<std::conjunction<meta::is_non_const_reference<T>, impl::is_derived_from_unknown_object<std::decay_t<T>>>, std::is_null_pointer<T>>>>
	void* detach_abi(T&& object) noexcept
	{
		if constexpr (std::is_null_pointer_v<T>)
		{
			return nullptr;
		}
		else
		{
			// When the object is a named rvalue reference, we just pass it to the put_abi function as a lvalue reference (without perfect forwarding).
			// Thus, the ABI of the object is capable of being exchanged.
			return std::exchange(*put_abi(object), nullptr);
		}
	}

	/// <summary>
	/// Duplicates an ABI and assignes it to an object with the reference count increased.
	/// </summary>
	/// <param name="object">The object</param>
	/// <param name="value">The ABI</param>
	inline void copy_from_abi(unknown_object& object, void* value) noexcept
	{
		if (value)
		{
			static_cast<impl::abi_unknown_object*>(value)->add_ref();
		}

		*put_abi(object) = value;
	}

	/// <summary>
	/// Copy the ABI of an object to another ABI with the reference count increased.
	/// </summary>
	/// <param name="object">The object</param>
	/// <param name="value">The ABI</param>
	inline void copy_to_abi(const unknown_object& object, void*& value) noexcept
	{
		if (value = get_abi(object))
		{
			static_cast<impl::abi_unknown_object*>(value)->add_ref();
		}
	}
}

namespace glasssix::exposing::impl
{
	namespace details
	{
		template<typename To, bool has_exception, typename From, typename = std::enable_if_t<std::conjunction_v<is_derived_from_unknown_object<From>, is_derived_from_unknown_object<To>>>>
		To as_impl(From* ptr)
		{
			To result{ nullptr };

			if (ptr && !ptr->query_interface(guid_of_v<To>, put_abi(result)))
			{
				if constexpr (has_exception)
				{
					throw glasssix_abi_no_interface{};
				}
			}

			return result;
		}
	}

	template<typename To, typename From>
	To as(From* ptr)
	{
		return details::as_impl<To, true>(ptr);
	}

	template<typename To, typename From>
	To try_as(From* ptr) noexcept
	{
		return details::as_impl<To, false>(ptr);
	}
}
