#ifndef DETECT_KNN_HARDWARE_LIBRARY_KNN_UTILS_H
#define DETECT_KNN_HARDWARE_LIBRARY_KNN_UTILS_H

#include "irisviel_search.hpp"

#include <vector>
#include <cstdint>


namespace glasssix
{
	namespace irisviel
	{
		class knn_utils final
		{
		public:
			static float search_single(const float* left, const float* right)
			{
				std::vector<const float*> adapter_left = { left };
				std::vector<const float*> adapter_right = { right };
				irisviel_search searcher{ adapter_left, dimensions_ };

				vector2d<uint32_t> tmp;
				vector2d<float> results;

				std::tie(tmp, results) = searcher.search_vector(adapter_right, 1);

				return results[0][0];
			}
		private:
			static constexpr int dimensions_ = 128;
		};
	}

}

#endif //DETECT_KNN_HARDWARE_LIBRARY_KNN_UTILS_H
