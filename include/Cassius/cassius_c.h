#include "Primitives/pure_c_handle_utils.h"

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>

#ifdef _MSC_VER
#define CASSIUS_C_EXPORT __declspec(dllexport)
#else
#define CASSIUS_C_EXPORT
#endif

	DEFINE_PURE_C_HANDLE(cassius);

	CASSIUS_C_EXPORT cassius_handle Cassius_NewInstance(int device);
	CASSIUS_C_EXPORT void Cassius_ReleaseInstance(cassius_handle instance);
	CASSIUS_C_EXPORT unsigned char* Cassius_getVersion();
	CASSIUS_C_EXPORT float* Cassius_Forward(cassius_handle instance, unsigned char* input_data, int num, int order);

#ifdef __cplusplus
}
#endif
