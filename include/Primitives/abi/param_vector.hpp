#pragma once

#include "meta.hpp"
#include "base.hpp"
#include "base_abi.hpp"
#include "implements.hpp"

#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <iterator>
#include <type_traits>

namespace glasssix::exposing
{
	template<typename T>
	struct param_vector;
}

namespace glasssix::exposing::impl
{
	/// <summary>
	/// The ABI of a param_vector.
	/// </summary>
	template<typename T>
	struct abi<param_vector<T>>
	{
		using identity_type = type_identity_generic_interface;
		static constexpr guid id{ "DCB2A5A5-1D17-4E0A-83C2-640912AECD25" };

		struct type : abi_unknown_object
		{
			virtual void G6_ABI_CALL at(abi_in_t<std::uint64_t> index, abi_out_t<T> result) noexcept = 0;
			virtual void G6_ABI_CALL push_back(abi_in_t<T> item) noexcept = 0;
			virtual void G6_ABI_CALL remove_at(abi_in_t<std::uint64_t> index) noexcept = 0;
			virtual void G6_ABI_CALL insert_at(abi_in_t<std::uint64_t> index, abi_in_t<T> item) noexcept = 0;
			virtual void G6_ABI_CALL clear() noexcept = 0;
		};
	};
	
	/// <summary>
	/// The vtable of a param_vector.
	/// </summary>
	template<typename Derived, typename T>
	struct interface_vtable<Derived, param_vector<T>> : interface_vtable_base<Derived, param_vector<T>>
	{
		virtual void G6_ABI_CALL at(abi_in_t<std::uint64_t> index, abi_out_t<T> result) noexcept override try
		{
			*result = detach_abi(this->self().at(index));
		}
		catch (...)
		{

		}

		virtual void G6_ABI_CALL push_back(abi_in_t<T> item) noexcept override try
		{
			this->self().push_back(create_from_abi<T>(item));
		}
		catch (...)
		{
			
		}

		virtual void G6_ABI_CALL remove_at(abi_in_t<std::uint64_t> index) noexcept override try
		{
			this->self().remove_at(index);
		}
		catch (...)
		{

		}

		virtual void G6_ABI_CALL insert_at(abi_in_t<std::uint64_t> index, abi_in_t<T> item) noexcept override try
		{
			this->self().insert_at(index, create_from_abi<T>(item));
		}
		catch (...)
		{

		}

		virtual void G6_ABI_CALL clear() noexcept override try
		{
			this->self().clear();
		}
		catch (...)
		{

		}
	};

	/// <summary>
	/// The ABI adapter of a param_vector.
	/// </summary>
	template<typename T>
	struct abi_adapter<param_vector<T>>
	{
		template<typename Derived>
		struct type : enable_self_abi_awareness<Derived, param_vector<T>>
		{
			T at(std::uint64_t index)
			{
				T result{};
				
				return (this->self_abi().at(index, put_abi(result)), result);
			}

			void push_back(T item)
			{
				this->self_abi().push_back(get_abi(item));
			}

			void remove_at(std::uint64_t index)
			{
				this->self_abi().remove_at(get_abi(index));
			}

			void insert_at(std::uint64_t index, T element)
			{
				this->self_abi().insert_at(get_abi(index), get_abi(element));
			}

			void clear()
			{
				this->self_abi().clear();
			}
		};
	};
}

namespace glasssix::exposing
{
	/// <summary>
	/// A mutable vector that is capable of being parameters.
	/// </summary>
	template<typename T>
	struct param_vector : inherits<param_vector<T>>
	{
		param_vector(std::nullptr_t = nullptr) noexcept
		{
		}
	};
}

namespace glasssix::exposing::impl
{
	template<typename T>
	class param_vector_impl : implements<param_vector_impl<T>, param_vector<T>>
	{
	public:
		T at(std::uint64_t index)
		{
			return T{};
		}

		void push_back(T item)
		{
			buffer_.emplace_back(item);
		}

		void remove_at(std::uint64_t index)
		{
		}

		void insert_at(std::uint64_t index, T element)
		{
		}

		void clear()
		{
		}
	private:
		std::vector<T> buffer_;
	};
}
