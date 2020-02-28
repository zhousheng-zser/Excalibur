#pragma once

#include <cstring>

namespace glasssix
{
	namespace irisviel
	{
		struct knn_mapping_data
		{
			static const size_t feature_offset;
			static const size_t is_active_offset;

			float feature[128] = {};
			char key[33] = {};
			bool is_active;

			knn_mapping_data() = default;
			knn_mapping_data(const knn_mapping_data& right) = default;
			knn_mapping_data(knn_mapping_data&& right) = default;
			knn_mapping_data& operator=(const knn_mapping_data& right) = default;

			bool key_equals(const knn_mapping_data& other);
		};
	}
}
