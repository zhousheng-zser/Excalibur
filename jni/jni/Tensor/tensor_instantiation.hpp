#pragma once

#include <memory>
#include <optional>
#include <type_traits>

#include <jni.h>
#include <glasssix/tensor.hpp>

namespace glasssix::jni::utils
{
	struct tensor_instantiation
	{
		excalibur::tensor_base* (*create_1)(excalibur::orderType layout);
		excalibur::tensor_base* (*create_2)(const int shape, int device, excalibur::orderType layout);
		excalibur::tensor_base* (*create_3)(const std::vector<int>& shape, int device, excalibur::orderType layout);
	
		static void initialize();
		static std::optional<tensor_instantiation> get_by_primitive(jclass type);
	};
}
