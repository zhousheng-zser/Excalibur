#pragma once

#include <vector>
#include <stdexcept>
#include <functional>

namespace glasssix
{
	namespace irisviel
	{
		// Provide access to features in a two-dimensional-array-like way.
		class knn_features
		{
		public:
			knn_features(const std::function<std::vector<const float*>()>& retriever, int dimension) : dimension_{ dimension }, retriever_{ retriever }
			{
				if (!retriever_)
				{
					throw std::runtime_error{ "The callback function cannot be null." };
				}
			}

			int dimension() const
			{
				return dimension_;
			}

			// Get all valid feature entries.
			std::vector<const float*> operator()() const
			{
				return retriever_();
			}
		private:
			int dimension_;
			std::function<std::vector<const float*>()> retriever_;
		};
	}
}
