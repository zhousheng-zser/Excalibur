#pragma once

#include "knn_runner.hpp"
#include "knn_mapping_data.hpp"
#include "knn_mapping_manager.hpp"

#include <array>
#include <vector>
#include <fstream>
#include <cstddef>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
//#include <boost/filesystem.hpp>

#include <boost/format.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>

#define LOG_TAG "nsg-native-lib"
#define LOGD(...)

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include <string>

namespace std
{
	using glasssix_irisviel_mananger_pair_type = std::pair<std::shared_ptr<glasssix::irisviel::knn_mapping_manager>, std::shared_ptr<glasssix::irisviel::knn_runner>>;

	template<> struct hash<glasssix_irisviel_mananger_pair_type>
	{
		std::size_t operator()(const glasssix_irisviel_mananger_pair_type& obj) const noexcept
		{
			return std::hash<glasssix_irisviel_mananger_pair_type::first_type>{}(obj.first);
		}
	};
}

namespace glasssix
{
	namespace irisviel
	{
		class knn_service
		{
		public:
			knn_service(int max_items, int dimension, const std::string& new_save_path, const std::string& tmp_path);
			virtual ~knn_service() = default;
			std::string save_path() const;
			std::string tmp_path() const;
			void build(const std::vector<std::string>& files);
			std::vector<knn_search_result> search(const float* feature, int top) const;
			void remove_all();
			std::vector<std::string> delete_features(const std::vector<std::string> keys);
			std::vector<std::string> delete_feature(const std::string& key);
			void add_features(std::vector<std::shared_ptr<knn_mapping_data>>& data);
			void add_feature(knn_mapping_data& data);
			void update(knn_mapping_data& data);
			void update_more(const std::vector<std::shared_ptr<knn_mapping_data>>& data);
		private:
			void build_single_core(const std::string& path, knn_mapping_data* data = nullptr, bool needs_build = true);
			static void force_delete_file(const std::string& path);

			int max_items_;
			int dimension_;
			std::string tmp_path_;
			std::string new_save_path_;
			std::unordered_map<std::shared_ptr<knn_mapping_manager>, std::shared_ptr<knn_runner>> cache_;
		};
	}
}
