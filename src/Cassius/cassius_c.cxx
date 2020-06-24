#include "cassius_c.h"
#include "CassiusFeature.hpp"
#include "Primitives/memory.hpp"

#include <cstring>

using namespace glasssix::pure_c;
using namespace glasssix::memory;
using namespace glasssix::cassius;

cassius_handle Cassius_NewInstance(int device)
{
	return to_handle<cassius_handle>(new CassiusFeature(device));
}

void Cassius_ReleaseInstance(cassius_handle instance)
{
	delete instance;
}

unsigned char* Cassius_getVersion()
{
	auto version = CassiusFeature::getVersion();
	std::size_t size = std::strlen(version) + 1;
	auto str = glasssix::memory::heap_alloc_elements<unsigned char>(size);

	std::memcpy(str, version, size);

	return str;
}

float* Cassius_Forward(cassius_handle instance, unsigned char* input_data, int num, int order)
{
	if (num > 0)
	{
		auto feature = from_handle<CassiusFeature>(instance)->Forward(input_data, num, order);
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
