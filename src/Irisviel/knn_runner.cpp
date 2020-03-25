#include "knn_runner.hpp"
#include "irisviel_search.hpp"
#include "nsg_calculate_error.hpp"

#include <algorithm>

#define LOG_TAG "nsg-native-lib"
#define LOGD(...) 

#include <fstream>

#include <boost/format.hpp>

namespace glasssix
{
	namespace irisviel
	{
		knn_runner::knn_runner(const std::shared_ptr<knn_features>& features, const std::string& tmp_path) : features_{ features }, tmp_path_{ tmp_path }, valid_state_{ false }
		{
		}

		knn_runner::~knn_runner()
		{
		}

		std::string knn_runner::index_file_path() const
		{
			return index_file_path_;
		}

		bool knn_runner::build(bool rebuild)
		{
			return build(map_file_path_, rebuild);
		}

		bool knn_runner::build(const std::string& map_file_name, bool rebuild)
		{
			map_file_path_ = map_file_name;
			current_data_ = (*features_)();
			if (current_data_.size() < 1)
			{
				return false;
			}

			auto dimension = features_->dimension();
			// auto uuid = boost::uuids::random_generator{}();

			// auto path = "/storage/emulated/0/knntmp/" + boost::str(boost::format{ "glasssix_knn_%1%.idx" } % uuid);
			index_file_path_ = tmp_path_ + boost::str(boost::format{ "/%1%.idx" } % map_file_name.substr(map_file_name.find_last_of('/') + 1));

			LOGD("====== Before build ======");

			auto safe_handler = [&](auto&& handler)
			{
				valid_state_ = false;

				try
				{
					std::forward<decltype(handler)>(handler)();
					valid_state_ = true;

					return valid_state_;
				}
				catch (glasssix::irisviel::nsg_calculate_error&)
				{
					return false;
				}
				catch (std::bad_alloc&)
				{
					return false;
				}
			};

			// Build the data.
			// We must catch the exceptions of infinite numbers here.
			if (!safe_handler([&] { searcher_ = std::make_shared<glasssix::irisviel::irisviel_search>(current_data_, dimension); }))
			{
				return false;
			}

			if (current_data_.size() > 1)
			{
				// rebuild only if the file does not exist.
				if (rebuild || !std::ifstream{ index_file_path_, std::ios::in | std::ios::binary }.is_open())
				{

					if (!safe_handler([&] { searcher_->build_graph(); }))
					{
						return false;
					}

					searcher_->save_graph(index_file_path_.c_str());
					LOGD("Graph needs building...complete.");
				}

				// Load the searcher.
				try
				{
					searcher_->load_graph(index_file_path_.c_str());
				}
				// Note: we check the file very strictly to ensure that it is a valid file.
				// So there should almost be an exception if the file was created when the power shut down unexpectedly.
				catch (glasssix::irisviel::nsg_calculate_error&)
				{
					if (!safe_handler([&] { searcher_->build_graph(); }))
					{
						return false;
					}

					searcher_->save_graph(index_file_path_.c_str());
					LOGD("Some exception occurs. Rebuild the graph!!!");

					searcher_->load_graph(index_file_path_.c_str());
				}

				if (!safe_handler([&] { searcher_->optimize_graph(); }))
				{
					return false;
				}

			}

			LOGD("====== After build ======");

			return true;
		}

		std::vector<std::vector<knn_search_result>> knn_runner::search(const float* feature, std::chrono::milliseconds& elapsed_time, int top)
		{
			return search_many({ feature }, elapsed_time, top);
		}

		std::vector<std::vector<knn_search_result>> knn_runner::search_many(const std::vector<const float*>& features, std::chrono::milliseconds& elapsed_time, int top)
		{
			int dimension = features_->dimension();
			std::vector<std::vector<knn_search_result>> result;

			// We only search the result when the current state is valid.
			if (!valid_state_)
			{
				return result;
			}

			try
			{
				//auto data = (*features_)();
				if (current_data_.size() < 1 || !searcher_)
				{
					return std::vector<std::vector<knn_search_result>>{};
				}

				top = static_cast<int>(std::min(current_data_.size(), static_cast<size_t>(top)));

				// Start timing.
				auto start = std::chrono::high_resolution_clock::now();

				vector2d<uint32_t> ids;
				vector2d<float> distances;

				std::tie(ids, distances) = searcher_->search_vector(features, top);

				// Construct results.
				for (size_t i = 0; i < distances.size(); i++)
				{
					std::vector<knn_search_result> inner;
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
						auto offset = reinterpret_cast<const std::uint8_t*>(current_data_[index]) - knn_mapping_data::feature_offset(dimension);
						auto result = knn_mapping_data::create(dimension, const_cast<std::uint8_t*>(offset));

						auto orginal_data = reinterpret_cast<const knn_mapping_data*>(reinterpret_cast<const int8_t*>(current_data_[index]) - knn_mapping_data::feature_offset(dimension));

						inner.emplace_back(knn_search_result{ result, distances[i][j] });
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
