#ifdef _MSC_VER
#define IRISVIEL_C_EXPORT __declspec(dllexport)
#else
#define IRISVIEL_C_EXPORT
#endif

namespace glasssix
{
	namespace irisviel
	{
		class face_service;
		struct database_search_result;
		struct database_record;
	}
}

extern "C" IRISVIEL_C_EXPORT glasssix::irisviel::face_service *Irisviel_NewInstance(int max_items, int dimension, char * new_save_path, char *cache_path);

extern "C" IRISVIEL_C_EXPORT void Irisviel_ReleaseInstance(glasssix::irisviel::face_service *instance);

extern "C" IRISVIEL_C_EXPORT char *Irisviel_save_path(glasssix::irisviel::face_service *instance);

extern "C" IRISVIEL_C_EXPORT char *Irisviel_tmp_path(glasssix::irisviel::face_service *instance);

extern "C" IRISVIEL_C_EXPORT void Irisviel_load_databases(glasssix::irisviel::face_service *instance);

extern "C" IRISVIEL_C_EXPORT int Irisviel_search(glasssix::irisviel::face_service *instance, glasssix::irisviel::database_search_result **result, float *feature, int top);

extern "C" IRISVIEL_C_EXPORT void Irisviel_remove_all(glasssix::irisviel::face_service *instance);

extern "C" IRISVIEL_C_EXPORT void Irisviel_delete_features(glasssix::irisviel::face_service *instance, int *needs_delete_files_num, char **needs_delete_files[], int keys_num, char *keys[]);

extern "C" IRISVIEL_C_EXPORT void Irisviel_delete_feature(glasssix::irisviel::face_service *instance, int *needs_delete_files_num, char **needs_delete_files[], char *key);

extern "C" IRISVIEL_C_EXPORT void Irisviel_add_features(glasssix::irisviel::face_service *instance, int data_num, glasssix::irisviel::database_record *data);

extern "C" IRISVIEL_C_EXPORT void Irisviel_add_feature(glasssix::irisviel::face_service *instance, glasssix::irisviel::database_record &data);

extern "C" IRISVIEL_C_EXPORT void Irisviel_update_feature(glasssix::irisviel::face_service *instance, glasssix::irisviel::database_record &data);

extern "C" IRISVIEL_C_EXPORT void Irisviel_update_more(glasssix::irisviel::face_service *instance, int data_num, glasssix::irisviel::database_record *data);
