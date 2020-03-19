#pragma once

namespace glasssix::jni
{
	enum class tensor_class_key
	{
		tensor,

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
		tensor_m_impl,
	};

	enum class tensor_method_key
	{
		tensor_constructor,

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
