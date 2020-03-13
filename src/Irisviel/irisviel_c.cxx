#include "../../include/Irisviel/irisviel_c.h"

#include "knn_service.hpp"
#include "Primitives/memory.hpp"

using glasssix::memory::heap_alloc_objects;
using glasssix::memory::heap_alloc_elements;

glasssix::irisviel::knn_service *Irisviel_NewInstance(int max_items, int dimension, char * new_save_path, char *tmp_path)
{
	return new glasssix::irisviel::knn_service(max_items, dimension, new_save_path, tmp_path);
}

void Irisviel_ReleaseInstance(glasssix::irisviel::knn_service *instance)
{
	delete instance;
}

char *Irisviel_save_path(glasssix::irisviel::knn_service *instance)
{
	std::string save_path = instance->save_path();
	size_t len = save_path.length();
	char *str = heap_alloc_elements<char>(len + 1);
	str[len] = '\0';
	std::copy(save_path.begin(), save_path.end(), str);
	return str;
}

char *Irisviel_tmp_path(glasssix::irisviel::knn_service *instance)
{
	std::string tmp_path = instance->tmp_path();
	size_t len = tmp_path.length();
	char *str = heap_alloc_elements<char>(len + 1);
	str[len] = '\0';
	std::copy(tmp_path.begin(), tmp_path.end(), str);
	return str;
}

void Irisviel_build(glasssix::irisviel::knn_service *instance, int n_files, char *str_files[])
{
	std::vector<std::string> files;
	for(int i = 0; i < n_files; i++)
		files.emplace_back(str_files[i]);
	
	instance->build(files);
}

int Irisviel_search(glasssix::irisviel::knn_service *instance, glasssix::irisviel::knn_search_result **result, float *feature, int top)
{
	std::vector<glasssix::irisviel::knn_search_result> result_vec = instance->search(feature, top);
	
	size_t result_num = result_vec.size();

	*result = heap_alloc_objects<glasssix::irisviel::knn_search_result>(result_num);
	std::copy(result_vec.begin(), result_vec.end(), *result);
	
	return result_num;
}

void Irisviel_remove_all(glasssix::irisviel::knn_service *instance)
{
	instance->remove_all();
}

void Irisviel_delete_features(glasssix::irisviel::knn_service *instance, int *needs_delete_files_num, char **needs_delete_files[], int keys_num, char *keys[])
{
	if(keys_num)
	{
		std::vector<std::string> keys_vec;
		for(int i = 0; i< keys_num; i++)
			keys_vec.emplace_back(keys[i]);
		
		std::vector<std::string> files = instance->delete_features(keys_vec);
		
		*needs_delete_files_num = files.size();
		if(*needs_delete_files_num)
		{
			*needs_delete_files = heap_alloc_elements<char*>(*needs_delete_files_num);
			for(int i = 0; i < *needs_delete_files_num; i++)
			{
				size_t file_path_len = files[i].size() + 1;
				(*needs_delete_files)[i] = heap_alloc_elements<char>(file_path_len + 1);
				(*needs_delete_files)[i][file_path_len] = '\0';
				std::copy(files[i].begin(), files[i].end(), (*needs_delete_files)[i]);
			}
		}
	}
}

void Irisviel_delete_feature(glasssix::irisviel::knn_service *instance, int *needs_delete_files_num, char **needs_delete_files[], char *key)
{
	std::vector<std::string> files = instance->delete_feature(key);
	
	*needs_delete_files_num = files.size();
	if(*needs_delete_files_num)
	{
		*needs_delete_files = heap_alloc_elements<char*>(*needs_delete_files_num);
		for(int i = 0; i < *needs_delete_files_num; i++)
		{
			size_t file_path_len = files[i].length();
			(*needs_delete_files)[i] = heap_alloc_elements<char>(file_path_len + 1);
			(*needs_delete_files)[i][file_path_len] = '\0';
			std::copy(files[i].begin(), files[i].end(), (*needs_delete_files)[i]);
		}
	}
}

void Irisviel_add_features(glasssix::irisviel::knn_service *instance, int data_num, glasssix::irisviel::knn_mapping_data *data)
{
	if(data_num)
	{
		std::vector <std::shared_ptr<glasssix::irisviel::knn_mapping_data>> data_vec(data_num);

		for (size_t i = 0; i < data_num; i++)
		{
			data_vec.emplace_back(data[i].shared());
		}

		instance->add_features(data_vec);
	}
}

void Irisviel_add_feature(glasssix::irisviel::knn_service *instance, glasssix::irisviel::knn_mapping_data &data)
{
	instance->add_feature(data);
}

void Irisviel_update_feature(glasssix::irisviel::knn_service *instance, glasssix::irisviel::knn_mapping_data &data)
{
	instance->update(data);
}

void Irisviel_update_more(glasssix::irisviel::knn_service *instance, int data_num, glasssix::irisviel::knn_mapping_data *data)
{
	if(data_num)
	{
		std::vector<std::shared_ptr<glasssix::irisviel::knn_mapping_data>> data_vec(data_num);

		for (size_t i = 0; i < data_num; i++)
		{
			data_vec.emplace_back(data[i].shared());
		}

		instance->update_more(data_vec);
	}
}
