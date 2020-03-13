#pragma once

#include <atomic>
#include <memory>
#include <cstddef>
#include <functional>

namespace glasssix
{
	namespace irisviel
	{
		/// <summary>
		/// Describes the mapping content.
		/// </summary>
		struct knn_mapping_header
		{
			int max_items;
			int current_position;
			int version;
			char index_file_name[128] = {};
		};

		struct knn_mapping_header_traits
		{
			static constexpr std::size_t header_size = sizeof(knn_mapping_header);
			static constexpr std::size_t current_position_offset = offsetof(knn_mapping_header, current_position);
		};
	}
}
