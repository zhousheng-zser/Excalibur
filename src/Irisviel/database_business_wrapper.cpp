#include "database_business_wrapper.hpp"
#include "irisviel_search.hpp"
#include "nsg_calculate_error.hpp"

#include <fstream>
#include <algorithm>

#include <glasssix/fmt/format.h>

namespace glasssix
{
	namespace irisviel
	{
		database_business_wrapper::database_business_wrapper(const std::shared_ptr<database_feature_observer>& observer, const std::string& map_file_path, const std::string& cache_path) : observer_{ observer }, map_file_path_{ map_file_path }, cache_path_{ cache_path }, cache_file_path_{ cache_path_ / fmt::format("{}.idx", map_file_path_.filename().replace_extension().string()) }, valid_state_{ false }
		{
		}

		std::string database_business_wrapper::cache_file_path() const
		{
			return cache_file_path_.string();
		}

		bool database_business_wrapper::build(bool rebuild)
		{
			current_data_ = (*observer_)();

			if (current_data_.size() < 1)
			{
				return false;
			}

			auto safe_handler = [&](auto&& handler)
			{
				valid_state_ = false;

				try
				{
					std::forward<decltype(handler)>(handler)();
					valid_state_ = true;

					return valid_state_;
				}
				catch (nsg_calculate_error&)
				{
					return false;
				}
				catch (std::bad_alloc&)
				{
					return false;
				}
			};

			auto dimension = observer_->dimension();

			// Build the data.
			// We must catch the exceptions of infinite numbers here.
			if (!safe_handler([&] { searcher_ = std::make_shared<irisviel_search>(current_data_, dimension); }))
			{
				return false;
			}

			if (current_data_.size() > 1)
			{
				auto cache_file_path = cache_file_path_.string();

				// rebuild only if the file does not exist.
				if (rebuild || !std::ifstream{ cache_file_path, std::ios::binary }.is_open())
				{

					if (!safe_handler([&] { searcher_->build_graph(); }))
					{
						return false;
					}

					searcher_->save_graph(cache_file_path.c_str());
				}

				// Load the searcher.
				try
				{
					searcher_->load_graph(cache_file_path.c_str());
				}
				// Note: we check the file very strictly to ensure that it is a valid file.
				// So there should almost be an exception if the file was created when the power shut down unexpectedly.
				catch (nsg_calculate_error&)
				{
					if (!safe_handler([&] { searcher_->build_graph(); }))
					{
						return false;
					}

					searcher_->save_graph(cache_file_path.c_str());
					searcher_->load_graph(cache_file_path.c_str());
				}

				if (!safe_handler([&] { searcher_->optimize_graph(); }))
				{
					return false;
				}

			}

			return true;
		}

		std::vector<std::vector<database_search_result>> database_business_wrapper::search(const float* feature, std::chrono::milliseconds& elapsed_time, int top) const
		{
			return search_many({ feature }, elapsed_time, top);
		}

		std::vector<std::vector<database_search_result>> database_business_wrapper::search_many(const std::vector<const float*>& features, std::chrono::milliseconds& elapsed_time, int top) const
		{
			int dimension = observer_->dimension();
			std::vector<std::vector<database_search_result>> result;

			// We only search the result when the current state is valid.
			if (!valid_state_)
			{
				return result;
			}

			try
			{
				if (current_data_.size() < 1 || !searcher_)
				{
					return std::vector<std::vector<database_search_result>>{};
				}

				top = std::min(static_cast<int>(current_data_.size()), top);

				// Start timing.
				auto start = std::chrono::high_resolution_clock::now();

				vector2d<uint32_t> ids;
				vector2d<float> distances;

				std::tie(ids, distances) = searcher_->search_vector(features, top);

				// Construct results.
				for (size_t i = 0; i < distances.size(); i++)
				{
					std::vector<database_search_result> inner;

					for (size_t j = 0; j < distances[i].size(); j++)
					{
						// Check if the index is out of range.
						auto index = ids[i][j];

						if (current_data_.size() == 1)
						{
							index = std::min(index, 0U);
						}

						if (index >= current_data_.size())
						{
							continue;
						}

						// Retrieve the orginal data in the mapping file.
						auto offset = reinterpret_cast<const std::uint8_t*>(current_data_[index]) - database_record::feature_offset(dimension);
						auto result = database_record::create(dimension, const_cast<std::uint8_t*>(offset));
						auto orginal_data = reinterpret_cast<const database_record*>(reinterpret_cast<const std::uint8_t*>(current_data_[index]) - database_record::feature_offset(dimension));

						inner.emplace_back(database_search_result{ result, distances[i][j] });
					}

					result.emplace_back(inner);
				}

				// End timing.
				auto end = std::chrono::high_resolution_clock::now();
				elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

				return result;
			}
			catch (...)
			{
			}

			return result;
		}
	}
}
