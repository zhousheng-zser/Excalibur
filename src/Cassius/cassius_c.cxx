#include "../../include/Cassius/cassius_c.h"
#include "../../include/Primitives/memory.hpp"

#include <cstring>

glasssix::cassius::CassiusFeature* Cassius_NewInstance(int device)
{
	return new glasssix::cassius::CassiusFeature(device);
}

void Cassius_ReleaseInstance(glasssix::cassius::CassiusFeature* instance)
{
	delete instance;
}

unsigned char* Cassius_getVersion()
{
	auto version = glasssix::cassius::CassiusFeature::getVersion();
	std::size_t size = std::strlen(version) + 1;
	auto str = glasssix::memory::heap_alloc_elements<unsigned char>(size);

	std::memcpy(str, version, size);

	return str;
}

float* Cassius_Forward(glasssix::cassius::CassiusFeature* instance, unsigned char* input_data, int num, int order)
{
	if (num > 0)
	{
		std::vector<std::vector<float> > feature = instance->Forward(input_data, num, order);
		auto result = glasssix::memory::heap_alloc_elements<float>(static_cast<std::size_t>(num) * 512);

		for (std::size_t i = 0; i < num; i++)
		{
			std::copy(feature[i].begin(), feature[i].end(), result + i * 512);
		}

		return result;
	}
	else
	{
		return nullptr;
	}
}
