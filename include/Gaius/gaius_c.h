#include "GaiusFeature.hpp"

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
#define GAIUS_C_EXPORT __declspec(dllexport)
#else
#define GAIUS_C_EXPORT
#endif

extern "C" GAIUS_C_EXPORT  glasssix::gaius::GaiusFeature *Gaius_NewInstance(int device);

extern "C" GAIUS_C_EXPORT  void Gaius_ReleaseInstance(glasssix::gaius::GaiusFeature *instance);

extern "C" GAIUS_C_EXPORT  unsigned char *Gaius_getVersion();

extern "C" GAIUS_C_EXPORT  float *Gaius_Forward(glasssix::gaius::GaiusFeature *instance, unsigned char *input_data, int num, int order, bool mask);