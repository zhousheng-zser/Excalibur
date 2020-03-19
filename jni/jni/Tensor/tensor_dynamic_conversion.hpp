#pragma once

#include <typeindex>

#include <tensor_layout.hpp>
#include <glasssix/tensor.hpp>

namespace glasssix::utils
{
	excalibur::tensor_base* convert_tensor_image_layout(const excalibur::tensor_base& source, excalibur::tensor_layout layout);
}
