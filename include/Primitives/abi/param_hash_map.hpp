#pragma once

#include "base.hpp"
#include "base_abi.hpp"
#include "implements.hpp"
#include "exceptions.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <iterator>
#include <type_traits>
#include <unordered_map>
#include <initializer_list>

namespace glasssix::exposing
{
	template<typename Key, typename Value>
	struct param_hash_map;
}

namespace glasssix::exposing::impl
{
	/// <summary>
	/// The ABI of a param_hash_map.
	/// </summary>
	template<typename Key, typename Value>
	struct abi<param_hash_map<Key, Value>>
	{
		using identity_type = type_identity_generic_interface;
		static constexpr guid id{ "5218106E-2AC9-438F-81CF-A1ED421878F6" };

		struct type : abi_unknown_object
		{
			virtual std::int32_t G6_ABI_CALL size(abi_out_t<std::uint64_t> result) noexcept = 0;
			virtual std::int32_t G6_ABI_CALL get_value(abi_in_t<Key> key, abi_out_t<Value> value) noexcept = 0;
			virtual std::int32_t G6_ABI_CALL try_get_value(abi_in_t<Key> key, abi_out_t<Value> value) noexcept = 0;
			virtual std::int32_t G6_ABI_CALL add_or_update(abi_in_t<Key> key, abi_in_t<Value> value) noexcept = 0;
			virtual std::int32_t G6_ABI_CALL contains(abi_in_t<Key> key) noexcept = 0;
			virtual std::int32_t G6_ABI_CALL remove(abi_in_t<Key> key) noexcept = 0;
			virtual std::int32_t G6_ABI_CALL clear() noexcept = 0;
		};
	};

	/// <summary>
	/// The vtable of a param_hash_map.
	/// </summary>
	template<typename Derived, typename Key, typename Value>
	struct interface_vtable<Derived, param_hash_map<Key, Value>> : interface_vtable_base<Derived, param_hash_map<Key, Value>>
	{
		virtual std::int32_t G6_ABI_CALL size(abi_out_t<std::uint64_t> result) noexcept override
		{
			return abi_safe_call([&] { *result = detach_abi(this->self().size()); });
		}

		virtual std::int32_t G6_ABI_CALL get_value(abi_in_t<Key> key, abi_out_t<Value> value) noexcept override
		{
			return abi_safe_call([&] { *value = detach_abi(this->self().get_value(create_from_abi<Key>(key))); });
		}

		virtual std::int32_t G6_ABI_CALL try_get_value(abi_in_t<Key> key, abi_out_t<Value> value) noexcept override
		{
			Value tmp{};
			abi_result result_code;
			
			return (result_code = abi_safe_call([&] { return to_abi_result(this->self().try_get_value(create_from_abi<Key>(key), tmp)); }), *value = detach_abi(tmp), result_code);
		}

		virtual std::int32_t G6_ABI_CALL add_or_update(abi_in_t<Key> key, abi_in_t<Value> value) noexcept override
		{
			return abi_safe_call([&] { this->self().add_or_update(create_from_abi<Key>(key), create_from_abi<Value>(value)); });
		}

		virtual std::int32_t G6_ABI_CALL contains(abi_in_t<Key> key) noexcept override
		{
			return abi_safe_call([&] { return to_abi_result(this->self().contains(create_from_abi<Key>(key))); });
		}

		virtual std::int32_t G6_ABI_CALL remove(abi_in_t<Key> key) noexcept override
		{
			return abi_safe_call([&] { this->self().remove(create_from_abi<Key>(key)); });
		}

		virtual std::int32_t G6_ABI_CALL clear() noexcept override
		{
			return abi_safe_call([&] { this->self().clear(); });
		}
	};

	/// <summary>
	/// The ABI adapter of a param_hash_map.
	/// </summary>
	template<typename Key, typename Value>
	struct abi_adapter<param_hash_map<Key, Value>>
	{
		template<typename Derived>
		struct type : enable_self_abi_awareness<param_hash_map<Key, Value>>
		{
			std::uint64_t size()
			{
				std::uint64_t result = 0;

				return (check_abi_result(this->self_abi().size(put_abi(result))), result);
			}

			Value get_value(const Key& key)
			{
				Value result{};

				return (check_abi_result(this->self_abi().get_value(get_abi(key), put_abi(result))), result);
			}

			bool try_get_value(const Key& key, Value& value) noexcept
			{
				return (value = {}, this->self_abi().try_get_value(get_abi(key), put_abi(value)) == error_success);
			}

			void add_or_update(const Key& key, const Value& value)
			{
				check_abi_result(this->self_abi().add_or_update(get_abi(key), get_abi(value)));
			}

			bool contains(const Key& key)
			{
				return this->self_abi().contains(get_abi(key)) == error_success;
			}

			void remove(const Key& key)
			{
				check_abi_result(this->self_abi().remove(get_abi(key)));
			}

			void clear()
			{
				check_abi_result(this->self_abi().clear());
			}
		};
	};
}

namespace glasssix::exposing
{
	/// <summary>
	/// A mutable vector that is capable of being parameters.
	/// </summary>
	template<typename Key, typename Value>
	struct param_hash_map : inherits<param_hash_map<Key, Value>>
	{
		using inherits<param_hash_map<Key, Value>>::inherits;
	};
}

namespace glasssix::exposing::impl
{
	template<typename Key, typename Value>
	class param_hash_map_impl : public implements<param_hash_map_impl<Key, Value>, param_hash_map<Key, Value>>
	{
	public:
		param_hash_map_impl()
		{
		}

		param_hash_map_impl(std::initializer_list<std::pair<const Key, Value>> list) : buffer_(std::move(list))
		{
		}

		std::uint64_t size()
		{
			return buffer_.size();
		}

		Value get_value(const Key& key)
		{
			auto iter = buffer_.find(key);

			return iter != buffer_.end() ? iter->second : throw abi_key_not_found{};
		}

		bool try_get_value(const Key& key, Value& value) noexcept
		{
			auto iter = buffer_.find(key);

			return iter != buffer_.end() ? (value = iter->second, true) : false;
		}

		void add_or_update(const Key& key, const Value& value)
		{
			buffer_.insert_or_assign(key, value);
		}

		bool contains(const Key& key)
		{
			return buffer_.find(key) != buffer_.end();
		}

		void remove(const Key& key)
		{
			if (auto iter = buffer_.find(key); iter != buffer_.end())
			{
				buffer_.erase(iter);
			}
		}

		void clear()
		{
			buffer_.clear();
		}
	private:
		std::unordered_map<Key, Value> buffer_;
	};
}

namespace glasssix::exposing
{
	template<typename Key, typename Value, typename = std::enable_if_t<std::conjunction_v<impl::has_abi_type<Key>, impl::has_abi_type<Value>>>>
	auto make_param_hash_map()
	{
		return make_as_first<impl::param_hash_map_impl<Key, Value>>();
	}

	template<typename Key, typename Value, typename = std::enable_if_t<std::conjunction_v<impl::has_abi_type<Key>, impl::has_abi_type<Value>>>>
	auto make_param_hash_map(std::initializer_list<std::pair<const Key, Value>> list)
	{
		return make_as_first<impl::param_hash_map_impl<Key, Value>>(list);
	}
}
