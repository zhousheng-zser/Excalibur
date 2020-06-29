#include "LonginusDetector.hpp"

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
#define LONGINUS_C_EXPORT __declspec(dllexport)
#else
#define LONGINUS_C_EXPORT
#endif

namespace glasssix
{
	namespace longinus
	{
		typedef struct Match_Retval_C {
			glasssix::longinus::face_rect_basic rect;
			char id[37];
			bool is_new;
		}Match_Retval_C;		
	}
}

extern "C" LONGINUS_C_EXPORT glasssix::longinus::LonginusDetector *Longinus_NewInstance(int device);

extern "C" LONGINUS_C_EXPORT void Longinus_ReleaseInstance(glasssix::longinus::LonginusDetector *instance);

extern "C" LONGINUS_C_EXPORT void Longinus_set(glasssix::longinus::LonginusDetector *instance, int type, int device);

#ifdef TRIAL
extern "C" LONGINUS_C_EXPORT int Longinus_detect(glasssix::longinus::LonginusDetector *instance, glasssix::longinus::face_rect_basic **ptr, const unsigned char *gray, int width, int height, int step, int minSize, float scale, int min_neighbors);

extern "C" LONGINUS_C_EXPORT int Longinus_detectWithInfo(glasssix::longinus::LonginusDetector *instance, glasssix::longinus::face_rect_with_face_info **ptr, const unsigned char *gray, int width, int height, int step, int minSize, float scale, int min_neighbors, int order);
#endif //!TRIAL

extern "C" LONGINUS_C_EXPORT int Longinus_match(glasssix::longinus::LonginusDetector *instance, glasssix::longinus::Match_Retval_C **ptr, glasssix::longinus::face_rect_basic *rects, int rect_num, int frame_extract_frequency, float distance_factor);

extern "C" LONGINUS_C_EXPORT int Longinus_matchWithInfo(glasssix::longinus::LonginusDetector* instance, glasssix::longinus::Match_Retval_C** ptr, glasssix::longinus::face_rect_with_face_info* rects, int rect_num, int frame_extract_frequency, float distance_factor);

extern "C" LONGINUS_C_EXPORT int Longinus_detectRetina(glasssix::longinus::LonginusDetector * instance, glasssix::longinus::face_rect_with_face_info * *ptr, const unsigned char* image, int min_win, int height, int width, int order, float threshold);

extern "C" LONGINUS_C_EXPORT int Longinus_detectEx(glasssix::longinus::LonginusDetector *instance, glasssix::longinus::face_rect_with_face_info **ptr, const unsigned char *image, int height, int width, int minSize, float *threshold, float factor, int stage, int order);

extern "C" LONGINUS_C_EXPORT int Longinus_detectEx_Mobile(glasssix::longinus::LonginusDetector *instance, glasssix::longinus::face_rect_with_face_info **ptr, const unsigned char *image, int height, int width, int minSize, float *threshold, float factor, int stage, int order);

extern "C" LONGINUS_C_EXPORT unsigned char *Longinus_alignFace(glasssix::longinus::LonginusDetector *instance, const unsigned char *ori_image, int n, int height, int width, int *bbox, int *landmarks);

extern "C" LONGINUS_C_EXPORT unsigned char *Longinus_alignFaceFromCropped(glasssix::longinus::LonginusDetector *instance, const unsigned char *ori_image, int n, int height, int width);

extern "C" LONGINUS_C_EXPORT int Longinus_detectEx_Mobile_nir(glasssix::longinus::LonginusDetector *instance, glasssix::longinus::face_rect_with_face_info **ptr, const unsigned char *image, int height, int width, int minSize, float *threshold, float factor, int stage, int order);

extern "C" LONGINUS_C_EXPORT void detectEx_mobile_pair(glasssix::longinus::LonginusDetector *instance, glasssix::longinus::face_rect_with_face_info **vsl_rect_ptr, int *vsl_rect_num, const unsigned char *vsl_image, int vsl_height, int vsl_width, int vsl_minSize, float *vsl_threshold, float vsl_factor, int vsl_stage, int vsl_order,
														glasssix::longinus::face_rect_with_face_info **nir_rect_ptr, int *nir_rect_num, const unsigned char *nir_image, int nir_height, int nir_width, int nir_minSize, float *nir_threshold, float nir_factor, int nir_stage, int nir_order);

extern "C" LONGINUS_C_EXPORT bool Longinus_blur_judge_vsl(glasssix::longinus::LonginusDetector *instance, const unsigned char *vsl_color_image, int height, int width, int n, int *bbox, int *landmarks, float *thresh, float **value, int order);

extern "C" LONGINUS_C_EXPORT bool Longinus_black_white_judge_vsl(glasssix::longinus::LonginusDetector *instance, const unsigned char *vsl_color_image, int height, int width, int n, int *bbox, int *landmarks, float *thresh, float **value, int order);

extern "C" LONGINUS_C_EXPORT bool Longinus_face_nose_judget_nir(glasssix::longinus::LonginusDetector *instance, const unsigned char *nir_color_image, int height, int width, int n, int *bbox, int *landmarks, float *thresh, float **value, int order);

extern "C" LONGINUS_C_EXPORT char *Longinus_getVersion();
