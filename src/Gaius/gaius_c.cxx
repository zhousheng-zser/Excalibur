#include "gaius_c.h"
#include "GaiusFeature.hpp"
#include "Primitives/memory.hpp"

#include <cstring>

using namespace glasssix::pure_c;
using namespace glasssix::memory;
using namespace glasssix::gaius;

GAIUS_C_EXPORT gaius_handle Gaius_NewInstance(int device)
{
	return to_handle<gaius_handle>(new GaiusFeature(device));
}

GAIUS_C_EXPORT void Gaius_ReleaseInstance(gaius_handle instance)
{
	delete instance;
}

GAIUS_C_EXPORT char* Gaius_getVersion()
{
	auto version = GaiusFeature::getVersion();
	std::size_t size = std::strlen(version) + 1;
	auto str = glasssix::memory::heap_alloc_elements<char>(size);

	std::memcpy(str, version, size);

	return str;
}

GAIUS_C_EXPORT float* Gaius_Forward(gaius_handle instance, const std::uint8_t* input_data, int num, int order, bool mask)
{
	if (num > 0)
	{
		auto feature = from_handle<GaiusFeature>(instance)->Forward(input_data, num, order, mask);
		auto result = heap_alloc_elements<float>(static_cast<std::size_t>(num) * 128);

		for (std::size_t i = 0; i < num; i++)
		{
			std::copy(feature[i].begin(), feature[i].end(), result + i * 128);
		}

		return result;
	}
	else
	{
		return nullptr;
	}
}
