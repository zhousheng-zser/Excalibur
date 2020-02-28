#include "CassiusFeature.hpp"

/// <remarks>
/// IMPORTANT TIPS!!!
/// All return values typed pointers in this API shall always be freed
/// by "heap_free" that is one of the platform-independent functions
/// declared within "include/Primitives/memory.hpp".
/// 
/// IT IS DANGEROUS TO RELEASE THEM BY CALLING std::free OR operator delete[]
/// BECAUSE OF POSSIBLE DIFFERENCE C++ STANDARD VERSIONS ACROSS DLL BOUNDARIES.
/// </remarks>

#ifdef _MSC_VER
#define CASSIUS_C_EXPORT __declspec(dllexport)
#else
#define CASSIUS_C_EXPORT
#endif

extern "C" CASSIUS_C_EXPORT glasssix::cassius::CassiusFeature *Cassius_NewInstance(int device);

extern "C" CASSIUS_C_EXPORT void Cassius_ReleaseInstance(glasssix::cassius::CassiusFeature *instance);

extern "C" CASSIUS_C_EXPORT unsigned char *Cassius_getVersion();

extern "C" CASSIUS_C_EXPORT float *Cassius_Forward(glasssix::cassius::CassiusFeature *instance, unsigned char *input_data, int num, int order);
