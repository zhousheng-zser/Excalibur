#pragma once

namespace glasssix::jni
{
	enum class irisvika_class_key
	{
		face_service,
		database_record,
		database_search_result,

		// Exceptions
		null_pointer_exception,
		illegal_argument_exception,
		index_out_of_bounds_exception,
	};

	enum class irisvika_field_key
	{
		face_service_m_impl,
		database_record_m_key,
		database_record_m_feature
	};

	enum class irisvika_method_key
	{
		database_record_constructor,
		database_search_result_constructor
	};
}
