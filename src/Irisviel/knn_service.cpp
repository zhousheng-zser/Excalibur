#include "knn_service.hpp"

namespace glasssix
{
	namespace irisviel
	{
		knn_service::knn_service(int max_items, int dimension, const std::string& new_save_path, const std::string& tmp_path) : max_items_{ max_items }, dimension_{ dimension }, new_save_path_{ new_save_path }, tmp_path_{ tmp_path }
		{
		}

		std::string knn_service::save_path() const
		{
			return new_save_path_;
		}

		std::string knn_service::tmp_path() const
		{
			return tmp_path_;
		}

		void knn_service::build(const std::vector<std::string>& files)
		{
			for (auto& item : files)
			{
				build_single_core(item);
			}
		}

		std::vector<knn_search_result> knn_service::search(const float* feature, int top) const
		{
			std::chrono::milliseconds total_time;
			std::vector<knn_search_result> result;

			for (auto& item : cache_)
			{
				std::chrono::milliseconds elapsed_time;

				auto single_result = item.second->search(feature, elapsed_time, top)[0];

				std::copy(single_result.begin(), single_result.end(),
					std::back_inserter(result));

				total_time += elapsed_time;
			}

			std::sort(result.begin(), result.end(),
				[](const knn_search_result& left, knn_search_result& right) {
					return left.distance_in_percentage > right.distance_in_percentage;
				});

			result.resize(std::min(static_cast<size_t>(top), result.size()));

			return result;
		}

		void knn_service::remove_all()
		{
			cache_.clear();
		}

		std::vector<std::string> knn_service::delete_features(const std::vector<std::string> keys)
		{
			std::unordered_map<std::shared_ptr<knn_mapping_manager>, std::string> needs_delete;

			if (keys.size() == 1)
			{
				return delete_feature(keys.front());
			}

			for (auto& item : cache_)
			{
				bool is_built = false;
				std::for_each(keys.begin(), keys.end(), [&](const std::string& key) {
					if (item.first->delete_by_key(key))
					{
						is_built = true;
					}
					});
				if (is_built)
				{
					if (!item.second->build(true))
					{
						needs_delete[item.first] = item.first->file_path();
					}
					else
					{
						item.first->update_index_file(item.second->index_file_path());
						item.first->save_changes();
					}
				}
			}

			std::vector<std::string> files;

			LOGD("Here!!!!!!");
			for (auto& item : needs_delete)
			{
				auto file_path = item.second;
				cache_.erase(item.first);
				LOGD("--file_path %s", file_path.c_str());
				//force_delete_file(file_path);
				files.emplace_back(file_path);
			}

			return files;
		}

		std::vector<std::string> knn_service::delete_feature(const std::string& key)
		{
			std::unordered_map<std::shared_ptr<knn_mapping_manager>, std::string> needs_delete;
			for (auto& item : cache_)
			{
				if (item.first->delete_by_key(key))
				{
					if (!item.second->build(true))
					{
						needs_delete[item.first] = item.first->file_path();
					}
					else
					{
						item.first->update_index_file(item.second->index_file_path());
						item.first->save_changes();
					}
				}
			}

			std::vector<std::string> files;

			for (auto& item : needs_delete)
			{
				auto file_path = item.second;
				cache_.erase(item.first);
				// force_delete_file(file_path);
				files.emplace_back(file_path);
			}

			return files;
		}

		void knn_service::add_features(std::vector<std::shared_ptr<knn_mapping_data>>& data)
		{
			std::unordered_set<std::pair<std::shared_ptr<knn_mapping_manager>, std::shared_ptr<knn_runner>>> used_managers;
			for (int i = 0; i < data.size(); ++i)
			{
				auto item = std::find_if(cache_.begin(), cache_.end(),
					[&](const std::pair<std::shared_ptr<knn_mapping_manager>, std::shared_ptr<knn_runner>>& item) {
						return item.first->emplace_back(*data[i]);
					});
				if (item != cache_.end())
				{
					used_managers.emplace(*item);
					continue;
				}

				// If not found, create new.
				auto uuid = boost::uuids::to_string(boost::uuids::random_generator{}());
				std::string file_path{ new_save_path_ + "/append_" + uuid + ".map" };
				std::fstream{ file_path, std::ios::trunc | std::ios::out | std::ios::binary };
				build_single_core(file_path, data[i].get(), false);
			}

			for (auto& item : used_managers)
			{
				item.second->build(item.first->file_path(), true);
				item.first->update_index_file(item.second->index_file_path());
				item.first->save_changes();
			}
		}

		void knn_service::add_feature(knn_mapping_data& data)
		{
			auto item = std::find_if(cache_.begin(), cache_.end(),
				[&](const std::pair<std::shared_ptr<knn_mapping_manager>, std::shared_ptr<knn_runner>>& item) {
					return item.first->emplace_back(data);
				});

			// Success, rebuild it.
			if (item != cache_.end())
			{
				item->second->build(item->first->file_path(), true);
				item->first->save_changes();
			}
			else
			{
				// Fail, create new file.
				auto uuid = boost::uuids::to_string(boost::uuids::random_generator{}());
				std::string file_path{ new_save_path_ + "/append_" + uuid + ".map" };
				std::fstream{ file_path, std::ios::trunc | std::ios::out | std::ios::binary };

				build_single_core(file_path, &data);
			}
		}

		void knn_service::update(knn_mapping_data& data)
		{
			for (auto& item : cache_)
			{
				if (item.first->update(data))
				{
					item.second->build(true);
					item.first->update_index_file(item.second->index_file_path());
					item.first->save_changes();
				}
			}
		}

		void knn_service::update_more(const std::vector<std::shared_ptr<knn_mapping_data>>& data)
		{
			std::unordered_set<std::pair<std::shared_ptr<knn_mapping_manager>, std::shared_ptr<knn_runner>>> used_managers;

			for (auto& item : cache_)
			{
				bool is_used = false;

				for (auto& each_data : data)
				{
					if (item.first->update(*each_data))
					{
						is_used = true;
					}
				}
				if (is_used)
				{
					used_managers.emplace(item);
				}
			}

			for (auto& item : used_managers)
			{
				item.second->build(true);
				item.first->update_index_file(item.second->index_file_path());
				item.first->save_changes();
			}
		}

		void knn_service::build_single_core(const std::string& path, knn_mapping_data* data, bool needs_build)
		{
			auto manager = std::make_shared<knn_mapping_manager>(path, max_items_, dimension_);
			auto runner = std::make_shared<knn_runner>(manager->create_features_reference(),
				tmp_path_);
			//LOGD("All finished");

			if (data != nullptr)
			{
				manager->emplace_back(*data);
			}

			if (needs_build)
			{

				// Run the KNN algorithm.
				runner->build(path, data != nullptr);

				//LOGD("After knn build");
				manager->update_index_file(runner->index_file_path());
				manager->save_changes();
			}
			cache_[manager] = runner;
		}

		void knn_service::force_delete_file(const std::string& path)
		{
			system(("rm """ + path + """").c_str());
		}
	}
}
