#include "../../include/Irisviel/irisviel_c.h"

#include "face_service.hpp"
#include "Primitives/memory.hpp"

using glasssix::memory::heap_alloc_objects;
using glasssix::memory::heap_alloc_elements;

glasssix::irisviel::face_service* irisviel_new_instance(int max_items, int dimension, const char* working_directory)
{
	return new glasssix::irisviel::face_service{ max_items, dimension, working_directory };
}

void irisviel_release_instance(glasssix::irisviel::face_service* instance)
{
	if (instance)
	{
		delete instance;
	}
}

char* irisviel_database_path(glasssix::irisviel::face_service* instance)
{
	std::string database_path = instance->database_path();
	size_t len = database_path.length();
	char* str = heap_alloc_elements<char>(len + 1);
	str[len] = '\0';
	std::copy(database_path.begin(), database_path.end(), str);
	return str;
}

char* irisviel_cache_path(glasssix::irisviel::face_service* instance)
{
	std::string cache_path = instance->cache_path();
	size_t len = cache_path.length();
	char* str = heap_alloc_elements<char>(len + 1);
	str[len] = '\0';
	std::copy(cache_path.begin(), cache_path.end(), str);
	return str;
}

void irisviel_load_databases(glasssix::irisviel::face_service* instance)
{
	instance->load_databases();
}

int irisviel_search(glasssix::irisviel::face_service* instance, glasssix::irisviel::database_search_result** result, float* feature, int top)
{
	std::vector<glasssix::irisviel::database_search_result> result_vec = instance->search(feature, top);

	size_t result_num = result_vec.size();

	*result = heap_alloc_objects<glasssix::irisviel::database_search_result>(result_num);
	std::copy(result_vec.begin(), result_vec.end(), *result);

	return result_num;
}

void irisviel_remove_all(glasssix::irisviel::face_service* instance)
{
	instance->remove_all();
}

void Irisviel_delete_features(glasssix::irisviel::face_service* instance, int keys_num, char* keys[])
{
	if (keys_num)
	{
		std::vector<std::string> keys_vec;
		for (int i = 0; i < keys_num; i++)
			keys_vec.emplace_back(keys[i]);

		instance->delete_features(keys_vec);
	}
}

void Irisviel_delete_feature(glasssix::irisviel::face_service* instance, char* key)
{
	instance->delete_feature(key);
}

void Irisviel_add_features(glasssix::irisviel::face_service* instance, int data_num, glasssix::irisviel::database_record* data)
{
	if (data_num)
	{
		std::vector<std::shared_ptr<glasssix::irisviel::database_record>> data_vec(data_num);

		for (size_t i = 0; i < data_num; i++)
		{
			data_vec.emplace_back(data[i].shared());
		}

		instance->add_features(data_vec);
	}
}

void Irisviel_add_feature(glasssix::irisviel::face_service* instance, glasssix::irisviel::database_record& data)
{
	instance->add_feature(data);
}

void Irisviel_update_feature(glasssix::irisviel::face_service* instance, glasssix::irisviel::database_record& data)
{
	instance->update(data);
}

void Irisviel_update_more(glasssix::irisviel::face_service* instance, int data_num, glasssix::irisviel::database_record* data)
{
	if (data_num)
	{
		std::vector<std::shared_ptr<glasssix::irisviel::database_record>> data_vec(data_num);

		for (size_t i = 0; i < data_num; i++)
		{
			data_vec.emplace_back(data[i].shared());
		}

		instance->update_more(data_vec);
	}
}
