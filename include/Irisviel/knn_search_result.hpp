#pragma once

#include "knn_mapping_data.hpp"

namespace glasssix
{
	namespace irisviel
	{
		// Search result.
		struct knn_search_result
		{
			std::shared_ptr<knn_mapping_data> data;
			float distance_in_percentage;

			knn_search_result() = default;

			knn_search_result(const std::shared_ptr<knn_mapping_data>& data, float distance_in_percentage) : data{ data }, distance_in_percentage{ distance_in_percentage }
			{
			}

			knn_search_result(const knn_search_result& right)
			{
				data = right.data;
				distance_in_percentage = right.distance_in_percentage;
			}

			knn_search_result(knn_search_result&& right) noexcept
			{
				data = std::move(right.data);
				distance_in_percentage = std::move(right.distance_in_percentage);
			}

			knn_search_result& operator=(const knn_search_result& right)
			{
				data = right.data;
				distance_in_percentage = right.distance_in_percentage;

				return *this;
			}
		};
	}
}
