#include "../../include/Gaius/gaius_c.h"
#include "Primitives/memory.hpp"

#include <cstring>

glasssix::gaius::GaiusFeature* Gaius_NewInstance(int device)
{
	return new glasssix::gaius::GaiusFeature(device);
}

void Gaius_ReleaseInstance(glasssix::gaius::GaiusFeature* instance)
{
	delete instance;
}

unsigned char* Gaius_getVersion()
{
	auto version = glasssix::gaius::GaiusFeature::getVersion();
	std::size_t size = std::strlen(version) + 1;
	auto str = glasssix::memory::heap_alloc_elements<unsigned char>(size);

	std::memcpy(str, version, size);

	return str;
}

float* Gaius_Forward(glasssix::gaius::GaiusFeature* instance, unsigned char* input_data, int num, int order, bool mask)
{
	if (num > 0)
	{
		auto feature = instance->Forward(input_data, num, order, mask);
		auto result = glasssix::memory::heap_alloc_elements<float>(static_cast<std::size_t>(num) * 128);

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
