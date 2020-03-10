#pragma once

namespace glasssix::jni
{
	enum class tensor_class_key
	{
		tensor,
		number,
		byte,
		integer,
		long_integer,
		short_integer,
		single_float,
		double_float,
	};

	enum class tensor_field_key
	{
		tensor_m_impl,
	};

	enum class tensor_method_key
	{
		// java.lang.Number
		number_byte_value,
		number_int_value,
		number_short_value,
		number_long_value,
		number_float_value,
		number_double_value
	};
}
