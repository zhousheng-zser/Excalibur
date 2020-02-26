#include "knn_service.hpp"

#ifdef _MSC_VER
#define IRISVIEL_C_EXPORT __declspec(dllexport)
#else
#define IRISVIEL_C_EXPORT
#endif

extern "C" IRISVIEL_C_EXPORT glasssix::irisviel::knn_service *Irisviel_NewInstance(int max_items, char * new_save_path, char *tmp_path);

extern "C" IRISVIEL_C_EXPORT void Irisviel_ReleaseInstance(glasssix::irisviel::knn_service *instance);

extern "C" IRISVIEL_C_EXPORT char *Irisviel_save_path(glasssix::irisviel::knn_service *instance);

extern "C" IRISVIEL_C_EXPORT char *Irisviel_tmp_path(glasssix::irisviel::knn_service *instance);

extern "C" IRISVIEL_C_EXPORT void Irisviel_build(glasssix::irisviel::knn_service *instance, int n_files, char *str_files[]);

extern "C" IRISVIEL_C_EXPORT int Irisviel_search(glasssix::irisviel::knn_service *instance, glasssix::irisviel::knn_search_result **result, float *feature, int top);

extern "C" IRISVIEL_C_EXPORT void Irisviel_remove_all(glasssix::irisviel::knn_service *instance);

extern "C" IRISVIEL_C_EXPORT void Irisviel_delete_features(glasssix::irisviel::knn_service *instance, int *needs_delete_files_num, char **needs_delete_files[], int keys_num, char *keys[]);

extern "C" IRISVIEL_C_EXPORT void Irisviel_delete_feature(glasssix::irisviel::knn_service *instance, int *needs_delete_files_num, char **needs_delete_files[], char *key);

extern "C" IRISVIEL_C_EXPORT void Irisviel_add_features(glasssix::irisviel::knn_service *instance, int data_num, glasssix::irisviel::knn_mapping_data *data);

extern "C" IRISVIEL_C_EXPORT void Irisviel_add_feature(glasssix::irisviel::knn_service *instance, glasssix::irisviel::knn_mapping_data &data);

extern "C" IRISVIEL_C_EXPORT void Irisviel_update_feature(glasssix::irisviel::knn_service *instance, glasssix::irisviel::knn_mapping_data &data);

extern "C" IRISVIEL_C_EXPORT void Irisviel_update_more(glasssix::irisviel::knn_service *instance, int data_num, glasssix::irisviel::knn_mapping_data *data);