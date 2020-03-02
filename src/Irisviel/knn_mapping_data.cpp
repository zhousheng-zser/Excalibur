#include "knn_mapping_data.hpp"

#include <cstddef>

namespace glasssix
{
	namespace irisviel
	{
		const size_t knn_mapping_data::feature_offset = offsetof(knn_mapping_data, feature);
		const size_t knn_mapping_data::is_active_offset = offsetof(knn_mapping_data, is_active);

		knn_mapping_data::knn_mapping_data() : feature{}, key{}, is_active{}
		{
		}

		bool knn_mapping_data::key_equals(const knn_mapping_data& other)
		{
			return !strncmp(key, other.key, sizeof(key));
		}
	}
}
