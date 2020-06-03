#pragma once

#include "base.hpp"
#include "guid.hpp"
#include "meta.hpp"
#include "exceptions.hpp"
#include "g6_attributes.hpp"

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
	template<typename To, typename From>
	To as(From* ptr);

	template<typename To, typename From>
	To try_as(From* ptr) noexcept;

	template<typename T, typename = void>
	struct abi;

	template<typename T>
	using abi_t = typename abi<T>::type;

	namespace details
	{
		template<typename T, typename = void>
		struct has_abi_type_top_level : std::false_type {};

		template<typename T>
		struct has_abi_type_top_level<T, std::void_t<abi_t<T>>> : std::true_type {};
	}

	template<typename T>
	struct has_abi_type : details::has_abi_type_top_level<T> {};

	template<template<typename...> typename T, typename... Args>
	struct has_abi_type<T<Args...>> : std::conjunction<details::has_abi_type_top_level<T<Args...>>, has_abi_type<Args>...> {};

	/// <summary>
	/// Checks whether a type contains a corresponding ABI type recursively.
	/// </summary>
	template<typename T>
	inline constexpr bool has_abi_type_v = has_abi_type<T>::value;

	/// <summary>
	/// The ABI is its own for a primitive type.
	/// </summary>
	template<typename T>
	struct abi<T, std::enable_if_t<is_primitive_v<T>>>
	{
		using type = T;
	};

	template<typename T>
	struct abi<T, std::enable_if_t<std::is_enum_v<T>>>
	{
		using type = std::underlying_type_t<T>;
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
	/// Checks whether a type is an ABI interface.
	/// </summary>
	template<typename T>
	struct is_well_defined_interface : std::conjunction<std::is_standard_layout<T>, has_abi_type<T>, std::is_base_of<unknown_object, T>> {};

	template<typename T>
	inline constexpr bool is_well_defined_interface_v = is_well_defined_interface<T>::value;

	template<typename T, typename = void>
	struct abi_in;

	/// <summary>
	/// An input argument of an ABI function.
	/// </summary>
	template<typename T>
	using abi_in_t = typename abi_in<T>::type;

	template<typename T>
	struct abi_in<T, std::enable_if_t<std::disjunction_v<is_primitive<T>, std::is_enum<T>>>>
	{
		using type = abi_t<T>;
	};

	template<typename T>
	struct abi_in<T, std::enable_if_t<is_well_defined_interface_v<T>>>
	{
		using type = void*;
	};

	template<typename T>
	struct abi_out
	{
		using type = abi_in_t<T>*;
	};

	/// <summary>
	/// An output argument of an ABI function.
	/// </summary>
	template<typename T>
	using abi_out_t = typename abi_out<T>::type;

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
	/// The root interface ABI.
	/// </summary>
	template<> struct abi<unknown_object>
	{
		using identity_type = type_identity_interface;
		static constexpr guid id{ "00000000-0000-0000-C000-000000000046" };

		struct type
		{
			virtual std::int32_t G6_ABI_CALL query_interface(guid id, void** object) noexcept = 0;
			virtual std::uint32_t G6_ABI_CALL add_ref() noexcept = 0;
			virtual std::uint32_t G6_ABI_CALL release() noexcept = 0;
		};
	};

	using abi_unknown_object = abi_t<unknown_object>;
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
	/// Gets the ABI of a primitive object.
	/// </summary>
	/// <typeparam name="T">The object type</typeparam>
	/// <param name="object">The object</param>
	/// <returns>The ABI</returns>
	template<typename T, typename = std::enable_if_t<impl::is_primitive_v<T>>>
	decltype(auto) get_abi(const T& object) noexcept
	{
		return meta::get_standard_layout_first_member<impl::abi_t<T>>(object);
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
	/// Gets a pointer to the ABI of a primitive object.
	/// </summary>
	/// <typeparam name="T">The object type</typeparam>
	/// <param name="object">The object</param>
	/// <returns>The pointer to the ABI</returns>
	template<typename T, typename = std::enable_if_t<impl::is_primitive_v<T>>>
	auto put_abi(T& object) noexcept
	{
		object = {};

		return &meta::get_standard_layout_first_member<impl::abi_t<T>>(object);
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
	template<typename T, typename = std::enable_if_t<std::disjunction_v<std::conjunction<meta::is_non_const_reference<T>, impl::is_well_defined_interface<std::decay_t<T>>>, std::is_null_pointer<T>>>>
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
	/// Detaches the ABI from a primitive object.
	/// </summary>
	/// <param name="object">The object</param>
	/// <returns>The ABI detached from the object</returns>
	template<typename T, typename = std::enable_if_t<impl::is_primitive_v<T>>>
	auto detach_abi(T&& object) noexcept
	{
		impl::abi_t<std::decay_t<T>> result{};

		return (meta::get_standard_layout_from_first_member<std::decay_t<T>>(result) = std::forward<T>(object), result);
	}

	/// <summary>
	/// Duplicates an ABI and assignes it to an object with the reference count increased.
	/// </summary>
	/// <param name="object">The object</param>
	/// <param name="abi">The ABI</param>
	inline void copy_from_abi(unknown_object& object, void* abi) noexcept
	{
		if (abi)
		{
			static_cast<impl::abi_unknown_object*>(abi)->add_ref();
		}

		*put_abi(object) = abi;
	}

	/// <summary>
	/// Duplicates an ABI and assignes it to a primitve object.
	/// </summary>
	/// <typeparam name="T">The object type</typeparam>
	/// <typeparam name="Abi">The ABI type</typeparam>
	/// <param name="object">The object</param>
	/// <param name="abi">The ABI</param>
	template<typename T, typename Abi, typename = std::enable_if_t<std::conjunction_v<impl::is_primitive<T>, std::is_same<impl::abi_t<T>, std::decay_t<Abi>>>>>
	void copy_from_abi(T& object, Abi&& abi) noexcept
	{
		*put_abi(object) = std::forward<Abi>(abi);
	}

	/// <summary>
	/// Copy the ABI of an object to another ABI with the reference count increased.
	/// </summary>
	/// <param name="object">The object</param>
	/// <param name="abi">The ABI</param>
	inline void copy_to_abi(const unknown_object& object, void*& abi) noexcept
	{
		if (abi = get_abi(object))
		{
			static_cast<impl::abi_unknown_object*>(abi)->add_ref();
		}
	}

	/// <summary>
	/// Copy the ABI of a primitive object to another ABI.
	/// </summary>
	/// <typeparam name="T">The object type</typeparam>
	/// <param name="object">The object</param>
	/// <param name="abi">The ABI</param>
	template<typename T, typename = std::enable_if_t<impl::is_primitive_v<T>>>
	void copy_to_abi(const unknown_object& object, impl::abi_t<T>& abi) noexcept
	{
		abi = get_abi(object);
	}

	/// <summary>
	/// Creates an interface from an ABI.
	/// </summary>
	/// <typeparam name="Interface">The interfacial type</typeparam>
	/// <param name="abi">The ABI</param>
	/// <returns>The interface</returns>
	template<typename Interface, typename = std::enable_if_t<impl::is_well_defined_interface_v<Interface>>>
	Interface create_from_abi(void* abi)
	{
		Interface result{};

		return (copy_from_abi(result, abi), result);
	}

	/// <summary>
	/// Creates a primitive type from an ABI.
	/// </summary>
	/// <typeparam name="Interface">The interfacial type</typeparam>
	/// <typeparam name="ABI">The ABI type</typeparam>
	/// <param name="abi">The ABI</param>
	/// <returns>The interface</returns>
	template<typename T, typename Abi, typename = std::enable_if_t<std::conjunction_v<impl::is_primitive<T>, std::is_same<impl::abi_t<T>, std::decay_t<Abi>>>>>
	T create_from_abi(Abi&& abi)
	{
		T result{};

		return (copy_from_abi(result, std::forward<Abi>(abi)), result);
	}
}

namespace glasssix::exposing::impl
{
	namespace details
	{
		template<typename To, bool has_exception, typename From, typename = std::enable_if_t<std::conjunction_v<is_well_defined_interface<From>, is_well_defined_interface<To>>>>
		To as_impl(From* ptr)
		{
			To result{ nullptr };

			if (ptr && !ptr->query_interface(guid_of_v<To>, put_abi(result)))
			{
				if constexpr (has_exception)
				{
					throw abi_no_interface{};
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
