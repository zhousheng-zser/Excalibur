#pragma once

namespace glasssix::jni
{
	enum class tensor_class_key
	{
		face_service,

		// Essential
		object,

		// Numbers
		number,
		byte,
		integer,
		long_integer,
		short_integer,
		single_float,
		double_float,

		// Exceptions
		null_pointer_exception,
		illegal_argument_exception,
		index_out_of_bounds_exception,

		// Reflection
		clazz,
		type_variable
	};

	enum class tensor_field_key
	{
		face_service_m_impl,
	};

	enum class tensor_method_key
	{
		face_service_constructor,
		face_service_clear,
		face_service_remove_all,
		face_service_database_directory,
		face_service_cache_directory,
		face_service_load_databases,
		face_service_search,
		face_service_delete_1,
		face_service_delete_2,
		face_service_add

		// Essential
		object_get_class,

		// Numbers
		number_byte_value,
		number_int_value,
		number_short_value,
		number_long_value,
		number_float_value,
		number_double_value,

		// Reflection
		clazz_get_type_parameters
	};
}
