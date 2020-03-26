#include "face_service.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <utility>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

#include <glasssix/fmt/format.h>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>

namespace std
{
	template<> struct hash<glasssix::irisviel::database_cache>
	{
		std::size_t operator()(const glasssix::irisviel::database_cache& obj) const noexcept
		{
			return std::hash<decltype(obj.manager)>{}(obj.manager);
		}
	};
}

namespace glasssix
{
	namespace irisviel
	{
		namespace
		{
			const fs::path database_extension{ ".map" };

			template<typename Predicate>
			void delete_if_core(std::vector<database_cache>& cache, Predicate&& predicate)
			{
				for (size_t i = 0; i < cache.size(); i++)
				{
					auto& item = cache[i];

					if (std::forward<Predicate>(predicate)(item))
					{
						item.commit();
					}

					// Deletes the database if it is empty.
					if (item.manager->empty())
					{
						cache.erase(cache.begin() + i);
					}

				}
			}
		}

		face_service::face_service(int max_items, int dimension, const std::string& database_path, const std::string& cache_path) : max_items_{ max_items }, dimension_{ dimension }, database_path_{ database_path }, cache_path_{ cache_path }
		{
		}

		std::string face_service::database_path() const
		{
			return database_path_.string();
		}

		std::string face_service::cache_path() const
		{
			return cache_path_.string();
		}

		void face_service::clear()
		{
			cache_.clear();
		}

		void face_service::load_databases()
		{
			for (auto& item : fs::directory_iterator{ database_path_, fs::directory_options::skip_permission_denied })
			{
				if (item.path().filename().extension() == database_extension)
				{
					auto pair = create_new_database_core(item.path().string());

					// Builds the existing database.
					pair.wrapper->build(false);
				}
			}
		}

		std::vector<database_search_result> face_service::search(const float* feature, int top) const
		{
			std::chrono::milliseconds total_time;
			std::vector<database_search_result> result;

			for (auto& item : cache_)
			{
				std::chrono::milliseconds elapsed_time;
				auto search_result = item.wrapper->search(feature, elapsed_time, top);

				if (!search_result.empty())
				{
					auto single_result = search_result.front();
					std::copy(single_result.begin(), single_result.end(), std::back_inserter(result));
				}
				
				total_time += elapsed_time;
			}

			std::sort(result.begin(), result.end(), [](const database_search_result& left, database_search_result& right) { return left.distance_in_percentage > right.distance_in_percentage; });
			result.resize(std::min(static_cast<size_t>(top), result.size()));

			return result;
		}

		void face_service::delete_features(const std::vector<std::string>& keys)
		{
			delete_if_core(cache_, [&](database_cache& item) { return std::count_if(keys.begin(), keys.end(), [&](const std::string& key) { return item.manager->remove(key) && !item.manager->empty(); }) > 0; });
		}

		void face_service::delete_feature(const std::string& key)
		{
			delete_if_core(cache_, [&](database_cache& item) { return item.manager->remove(key) && !item.manager->empty(); });
		}

		void face_service::add_features(const std::vector<std::shared_ptr<database_record>>& records)
		{
			std::unordered_set<database_cache> changed_databases;

			for (auto& record : records)
			{
				auto item = find_available_database_core(record->key());

				if (item && item.manager->add(*record))
				{
					changed_databases.emplace(std::move(item));
				}
			}

			// Builds the changed databases.
			for (auto& item : changed_databases)
			{
				item.commit();
			}
		}

		void face_service::add_feature(database_record& record)
		{
			auto item = find_available_database_core(record.key());

			if (item && item.manager->add(record))
			{
				item.commit();
			}
		}

		void face_service::update(database_record& record)
		{
			for (auto& item : cache_)
			{
				if (item.manager->update(record))
				{
					item.commit();
				}
			}
		}

		void face_service::update_more(const std::vector<std::shared_ptr<database_record>>& records)
		{
			for (auto& item : cache_)
			{
				std::ptrdiff_t count = std::count_if(records.begin(), records.end(), [&](const std::shared_ptr<database_record>& record) { return item.manager->update(*record); });

				if (count > 0)
				{
					item.commit();
				}
			}
		}

		database_cache face_service::find_available_database_core(const std::string& key)
		{
			// Finds a database that can accommodate at least one record and ensures there is no repeated key among the databases.
			bool already_contains = false;
			auto item = std::find_if(cache_.begin(), cache_.end(), [&](const database_cache& item) { return !(already_contains = item.manager->contains(key)) && !item.manager->full(); });

			if (item != cache_.end())
			{
				return *item;
			}
			// Creates a new database if all the databases are full.
			else if (!already_contains)
			{
				auto uuid = boost::uuids::to_string(boost::uuids::random_generator{}());
				auto file_path = database_path_ / fmt::format("{}{}", uuid, database_extension.string());
				std::ofstream{ file_path, std::ios::trunc | std::ios::binary };

				return create_new_database_core(file_path.string());
			}

			return database_cache{};
		}

		database_cache face_service::create_new_database_core(const std::string& path)
		{
			auto manager = std::make_shared<database_manager>(path, max_items_, dimension_);
			auto wrapper = std::make_shared<database_business_wrapper>(manager->create_feature_observer(), path, cache_path_.string());

			return cache_.emplace_back(std::move(manager), std::move(wrapper));
		}
	}
}
