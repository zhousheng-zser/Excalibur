#pragma once

#include "dynamic_buffer.hpp"

#include <memory>
#include <cstddef>

namespace glasssix
{
	namespace irisviel
	{
		struct knn_mapping_data : dynamic_buffer, std::enable_shared_from_this<knn_mapping_data>
		{
			virtual bool is_active() const noexcept = 0;
			virtual void is_active(bool value) noexcept = 0;
			virtual char* key() noexcept = 0;
			virtual const char* key() const noexcept = 0;
			virtual void key(const char* value) noexcept = 0;
			virtual float* feature() noexcept = 0;
			virtual const float* feature() const noexcept = 0;
			virtual void feature(const float* value) noexcept = 0;
			virtual std::size_t feature_offset() const noexcept = 0;
			std::shared_ptr<knn_mapping_data> shared();

			static std::size_t struct_size(int dimension) noexcept;
			static std::size_t feature_offset(int dimension) noexcept;
			static std::shared_ptr<knn_mapping_data> create(int dimension);
			static std::shared_ptr<knn_mapping_data> create(int dimension, std::uint8_t* ptr);
			static std::shared_ptr<knn_mapping_data> create_ref(int dimension, std::uint8_t* ptr);
			static bool key_equals(const knn_mapping_data& left, const knn_mapping_data& right);
		};
	}
}
