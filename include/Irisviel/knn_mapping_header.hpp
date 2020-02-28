#pragma once

#include <atomic>
#include <memory>
#include <cstdint>
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
			// Indicates the header size.
			static const size_t header_size;

			int max_items;
			int current_position;
			int version;
			char index_file_name[128] = {};

			std::function<void(size_t, int)> update_function;

			knn_mapping_header() = default;
			knn_mapping_header(const knn_mapping_header& other) = default;
			knn_mapping_header(knn_mapping_header&& other) = default;
			knn_mapping_header& operator=(const knn_mapping_header& other) = default;
			knn_mapping_header& operator=(knn_mapping_header&& other) = default;

			void update_current_position()
			{
				if (update_function)
				{
					update_function(offsetof(knn_mapping_header, current_position), current_position);
				}
			}
		};
	}
}
