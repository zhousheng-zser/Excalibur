#pragma once

#include <vector>
#include <functional>
#include <type_traits>

namespace glasssix
{
	namespace irisviel
	{
		class database_feature_observer
		{
		public:
			template<typename Retriever>
			database_feature_observer(Retriever&& retriever, int dimension) : retriever_{ std::forward<Retriever>(retriever) }, dimension_{ dimension }
			{
			}

			int dimension() const noexcept;
			std::vector<const float*> operator()() const;
		private:
			int dimension_;
			std::function<std::vector<const float*>()> retriever_;
		};
	}
}
