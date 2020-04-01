#include "face_service.hpp"
#include "database_cache.hpp"
#include "filesystem_utils.hpp"

#include <list>
#include <chrono>
#include <fstream>
#include <cstddef>
#include <utility>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

#include <glasssix/fmt/format.h>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>

namespace glasssix
{
	namespace irisviel
	{
		namespace
		{
			const fs::path cache_folder{ "tmp" };
			const fs::path database_folder{ "data" };
			const fs::path database_extension{ ".map" };
		}

		class face_service::impl
		{
		public:
			impl(int single_database_capacity, int dimension, const fs::path& working_directory) : dimension_{ dimension }, single_database_capacity_{ single_database_capacity }, database_directory_{ working_directory / database_folder }, cache_directory_{ working_directory / cache_folder }
			{
				utils::safe_create_directories(database_directory_);
				utils::safe_create_directories(cache_directory_);
			}

			std::string database_diectory() const
			{
				return database_directory_.string();
			}

			std::string cache_directory() const
			{
				return cache_directory_.string();
			}

			void clear() noexcept
			{
				cache_.clear();
			}

			void remove_all()
			{
				std::for_each(cache_.begin(), cache_.end(), [](const std::shared_ptr<database_cache> item) { item->mark_for_deletion(); });
				cache_.clear();
			}

			void load_databases()
			{
				for (auto& item : fs::directory_iterator{ database_directory_, fs::directory_options::skip_permission_denied })
				{
					if (item.path().filename().extension() == database_extension)
					{
						auto cache = create_new_database_core(item.path().string());

						// Builds the existing database.
						cache->wrapper->build(false);
					}
				}
			}

			std::vector<database_search_result> search(const float* feature, int top) const
			{
				std::chrono::milliseconds total_time;
				std::vector<database_search_result> result;

				for (auto& item : cache_)
				{
					std::chrono::milliseconds elapsed_time;
					auto search_result = item->wrapper->search(feature, elapsed_time, top);

					if (!search_result.empty())
					{
						auto single_result = search_result.front();
						std::copy(single_result.begin(), single_result.end(), std::back_inserter(result));
					}

					total_time += elapsed_time;
				}

				std::sort(result.begin(), result.end(), [](const database_search_result& left, database_search_result& right) { return left.similarity > right.similarity; });
				result.resize(std::min(static_cast<size_t>(top), result.size()));

				return result;
			}

			void remove(const std::vector<std::string>& keys)
			{
				remove_if_core([&](database_cache& item) { return std::count_if(keys.begin(), keys.end(), [&](const std::string& key) { return item.manager->remove(key) && !item.manager->empty(); }) > 0; });
			}

			void remove(const std::string& key)
			{
				remove_if_core([&](database_cache& item) { return item.manager->remove(key) && !item.manager->empty(); });
			}

			void add(const std::vector<std::shared_ptr<database_record>>& records)
			{
				std::unordered_set<std::shared_ptr<database_cache>> changed_databases;

				for (auto& record : records)
				{
					auto item = find_available_database_core(record->key());

					if (item && item->manager->add(*record))
					{
						changed_databases.emplace(std::move(item));
					}
				}

				// Builds the changed databases.
				for (auto& item : changed_databases)
				{
					item->commit();
				}
			}

			void add(database_record& record)
			{
				auto item = find_available_database_core(record.key());

				if (item && item->manager->add(record))
				{
					item->commit();
				}
			}

			void update(database_record& record)
			{
				for (auto& item : cache_)
				{
					if (item->manager->update(record))
					{
						item->commit();
					}
				}
			}

			void update_more(const std::vector<std::shared_ptr<database_record>>& records)
			{
				for (auto& item : cache_)
				{
					std::ptrdiff_t count = std::count_if(records.begin(), records.end(), [&](const std::shared_ptr<database_record>& record) { return item->manager->update(*record); });

					if (count > 0)
					{
						item->commit();
					}
				}
			}

		private:
			template<typename Predicate>
			void remove_if_core(Predicate&& predicate)
			{
				for (auto iter = cache_.begin(); iter != cache_.end(); )
				{
					if (std::forward<Predicate>(predicate)(**iter))
					{
						(*iter)->commit();
					}

					// Deletes the database if it is empty.
					if ((*iter)->manager->empty())
					{
						iter = cache_.erase(iter);
					}
					else
					{
						iter++;
					}
				}
			}

			std::shared_ptr<database_cache> find_available_database_core(const std::string& key)
			{
				// Finds a database that can accommodate at least one record and ensures there is no repeated key among the databases.
				for (auto& item : cache_)
				{
					// Ensures the uniqueness of the key.
					if (item->manager->contains(key))
					{
						return nullptr;
					}

					if (!item->manager->full())
					{
						return item;
					}
				}

				// Generates a new empty database.
				auto uuid = boost::uuids::to_string(boost::uuids::random_generator{}());
				auto file_path = database_directory_ / fmt::format("{}{}", uuid, database_extension.string());
				std::ofstream{ file_path, std::ios::trunc | std::ios::binary };

				return create_new_database_core(file_path.string());
			}

			std::shared_ptr<database_cache> create_new_database_core(const std::string& path)
			{
				auto manager = std::make_shared<database_manager>(path, single_database_capacity_, dimension_);
				auto wrapper = std::make_shared<database_business_wrapper>(manager->create_feature_observer(), path, cache_directory_.string());

				return cache_.emplace_back(std::make_shared<database_cache>(std::move(manager), std::move(wrapper)));
			}

			int dimension_;
			int single_database_capacity_;
			fs::path cache_directory_;
			fs::path database_directory_;
			std::list<std::shared_ptr<database_cache>> cache_;
		};

		face_service::face_service(int single_database_capacity, int dimension, const std::string& working_directory) : impl_{ new impl{ single_database_capacity, dimension, working_directory } }
		{
		}

		face_service::~face_service()
		{
			if (impl_)
			{
				delete impl_;
				impl_ = nullptr;
			}
		}

		void face_service::clear() noexcept
		{
			impl_->clear();
		}

		void face_service::remove_all() noexcept
		{
			impl_->remove_all();
		}

		std::string face_service::database_directory() const
		{
			return impl_->database_diectory();
		}

		std::string face_service::cache_directory() const
		{
			return impl_->cache_directory();
		}

		void face_service::load_databases()
		{
			impl_->load_databases();
		}

		std::vector<database_search_result> face_service::search(const float* feature, int top) const
		{
			return impl_->search(feature, top);
		}

		void face_service::add(database_record& record)
		{
			impl_->add(record);
		}

		void face_service::add(const std::vector<std::shared_ptr<database_record>>& records)
		{
			impl_->add(records);
		}

		void face_service::remove(const std::string& key)
		{
			impl_->remove(key);
		}

		void face_service::remove(const std::vector<std::string>& keys)
		{
			impl_->remove(keys);
		}

		void face_service::update(database_record& record)
		{
			impl_->update(record);
		}

		void face_service::update(const std::vector<std::shared_ptr<database_record>>& records)
		{
			impl_->update_more(records);
		}
	}
}
