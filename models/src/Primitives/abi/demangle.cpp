#include "abi/demangle.hpp"

#include <memory>
#include <cstdlib>

#if defined(__GNUC__) && !defined(_WIN32)
#include <cxxabi.h>
#endif

namespace glasssix::exposing::allocations
{
	EXPORT_EXCALIBUR_PRIMITIVES void* G6_ABI_CALL demangle_cxx_abi_name_narrow(const char* name) noexcept
	{
		if (name == nullptr)
		{
			return detach_abi(param_string{});
		}

#ifdef __GUNC__
		std::unique_ptr<char, decltype(&std::free)> demangled_name{ abi::__cxa_demangle(name, nullptr, nullptr, nullptr), &std::free };

		return detach_abi(to_param_string(demangled_name ? demangled_name.get() : name);
#else
		return detach_abi(to_param_string(name));
#endif
	}
}
