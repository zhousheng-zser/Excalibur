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

extern "C"
{
	IRISVIEL_C_EXPORT glasssix::irisviel::face_service * irisviel_new_instance(int max_items, int dimension, const char* working_directory);

	IRISVIEL_C_EXPORT void irisviel_release_instance(glasssix::irisviel::face_service* instance);

	IRISVIEL_C_EXPORT char* irisviel_database_path(glasssix::irisviel::face_service* instance);

	IRISVIEL_C_EXPORT char* irisviel_cache_path(glasssix::irisviel::face_service * instance);

	IRISVIEL_C_EXPORT void irisviel_load_databases(glasssix::irisviel::face_service * instance);

	IRISVIEL_C_EXPORT int irisviel_search(glasssix::irisviel::face_service * instance, glasssix::irisviel::database_search_result * *result, float* feature, int top);

	IRISVIEL_C_EXPORT void irisviel_remove_all(glasssix::irisviel::face_service * instance);

	IRISVIEL_C_EXPORT void Irisviel_delete_features(glasssix::irisviel::face_service * instance, int* needs_delete_files_num, char** needs_delete_files[], int keys_num, char* keys[]);

	IRISVIEL_C_EXPORT void Irisviel_delete_feature(glasssix::irisviel::face_service * instance, int* needs_delete_files_num, char** needs_delete_files[], char* key);

	IRISVIEL_C_EXPORT void Irisviel_add_features(glasssix::irisviel::face_service * instance, int data_num, glasssix::irisviel::database_record * data);

	IRISVIEL_C_EXPORT void Irisviel_add_feature(glasssix::irisviel::face_service * instance, glasssix::irisviel::database_record & data);

	IRISVIEL_C_EXPORT void Irisviel_update_feature(glasssix::irisviel::face_service * instance, glasssix::irisviel::database_record & data);

	IRISVIEL_C_EXPORT void Irisviel_update_more(glasssix::irisviel::face_service * instance, int data_num, glasssix::irisviel::database_record * data);
}
