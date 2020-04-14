#pragma once

#ifdef __cplusplus
#include <utility>

extern "C"
{
#endif

#define DEFINE_PURE_C_HANDLE(name) typedef struct tag_##name##_handle {} *name##_handle

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace glasssix
{
	namespace pure_c
	{
		template<typename Handle, typename T>
		constexpr auto to_handle(T* obj)
		{
			return reinterpret_cast<Handle>(obj);
		}

		template<typename T, typename Handle>
		constexpr auto from_handle(Handle&& handle)
		{
			return reinterpret_cast<T*>(std::forward<Handle>(handle));
		}
	}
}
#endif
