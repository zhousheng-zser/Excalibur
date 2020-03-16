#include "../../include/Longinus/longinus_c.h"
#include "Primitives/memory.hpp"

using glasssix::memory::heap_free;
using glasssix::memory::heap_alloc_elements;

glasssix::longinus::LonginusDetector* Longinus_NewInstance(int device)
{
	return new glasssix::longinus::LonginusDetector(device);
}

void Longinus_ReleaseInstance(glasssix::longinus::LonginusDetector* instance)
{
	delete instance;
}

void Longinus_set(glasssix::longinus::LonginusDetector* instance, int type, int device)
{
	switch (type)
	{
	case 0:
		instance->set(glasssix::longinus::longinus_detection_type::FRONTALVIEW, device);
	case 1:
		instance->set(glasssix::longinus::longinus_detection_type::FRONTALVIEW_REINFORCE, device);
	case 2:
		instance->set(glasssix::longinus::longinus_detection_type::MULTIVIEW, device);
	case 3:
		instance->set(glasssix::longinus::longinus_detection_type::MULTIVIEW_REINFORCE, device);
	default:
		instance->set(glasssix::longinus::longinus_detection_type::FRONTALVIEW, device);
	}
}

int Longinus_detect(glasssix::longinus::LonginusDetector* instance, glasssix::longinus::face_rect_basic** ptr, unsigned char* gray, int width, int height, int step, int minSize, float scale, int min_neighbors)
{
	std::vector<glasssix::longinus::face_rect_basic> vec = instance->detect(gray, width, height, step, minSize, scale, min_neighbors, false, false);
	int num = vec.size();
	if (num)
	{
		sort_descend(vec);
		*ptr = heap_alloc_elements<glasssix::longinus::face_rect_basic>(num);
		std::copy(vec.begin(), vec.end(), *ptr);
	}
	else
	{
		*ptr = nullptr;
	}

	return num;
}

int Longinus_detectWithInfo(glasssix::longinus::LonginusDetector* instance, glasssix::longinus::face_rect_with_face_info** ptr, unsigned char* gray, int width, int height, int step, int minSize, float scale, int min_neighbors, int order)
{
	std::vector<glasssix::longinus::face_rect_with_face_info> vec = instance->detect(gray, width, height, step, minSize, scale, min_neighbors, order, false, false);
	int num = vec.size();
	if (num)
	{
		sort_descend(vec);
		*ptr = heap_alloc_elements<glasssix::longinus::face_rect_with_face_info>(num);
		std::copy(vec.begin(), vec.end(), *ptr);
	}
	else
	{
		*ptr = nullptr;
	}

	return num;
}

int Longinus_match(glasssix::longinus::LonginusDetector* instance, glasssix::longinus::Match_Retval_C** ptr, glasssix::longinus::face_rect_basic* rects, int rect_num, int frame_extract_frequency, float distance_factor)
{
	std::vector<glasssix::longinus::face_rect_basic> rect_vec;
	for (int i = 0; i < rect_num; i++)
	{
		rect_vec.push_back(rects[i]);
	}

	std::vector<glasssix::longinus::Match_Retval> vec = instance->match(rect_vec, frame_extract_frequency, distance_factor);
	int num = vec.size();
	if (num)
	{
		*ptr = heap_alloc_elements<glasssix::longinus::Match_Retval_C>(num);
		for (int i = 0; i < num; i++)
		{
			(*ptr)[i].rect = vec[i].rect;
			(*ptr)[i].id[36] = '\0';
			std::copy(vec[i].id.begin(), vec[i].id.end(), (*ptr)[i].id);
			(*ptr)[i].is_new = vec[i].is_new;
		}
	}
	else
	{
		*ptr = nullptr;
	}

	return num;
}

int Longinus_matchWithInfo(glasssix::longinus::LonginusDetector* instance, glasssix::longinus::Match_Retval_C** ptr, glasssix::longinus::face_rect_with_face_info* rects, int rect_num, int frame_extract_frequency, float distance_factor)
{
	std::vector<glasssix::longinus::face_rect_with_face_info> rect_vec;
	for (int i = 0; i < rect_num; i++)
	{
		rect_vec.push_back(rects[i]);
	}

	std::vector<glasssix::longinus::Match_Retval> vec = instance->match(rect_vec, frame_extract_frequency, distance_factor);
	int num = vec.size();
	if (num)
	{
		*ptr = heap_alloc_elements<glasssix::longinus::Match_Retval_C>(num);
		for (int i = 0; i < num; i++)
		{
			(*ptr)[i].rect = vec[i].rect;
			(*ptr)[i].id[36] = '\0';
			std::copy(vec[i].id.begin(), vec[i].id.end(), (*ptr)[i].id);
			(*ptr)[i].is_new = vec[i].is_new;
		}
	}
	else
	{
		*ptr = nullptr;
	}

	return num;
}

int Longinus_detectRetina(glasssix::longinus::LonginusDetector* instance, glasssix::longinus::face_rect_with_face_info** ptr, unsigned char* image, int height, int width, int order, float threshold, float factor)
{
	std::vector<glasssix::longinus::face_rect_with_face_info> vec = instance->detectRetina(image, 3, height, width, order, threshold, factor);
	int num = vec.size();
	if (num)
	{
		sort_descend(vec);
		*ptr = heap_alloc_elements<glasssix::longinus::face_rect_with_face_info>(num);
		std::copy(vec.begin(), vec.end(), *ptr);
	}
	else
	{
		*ptr = nullptr;
	}

	return num;
}

int Longinus_detectEx(glasssix::longinus::LonginusDetector* instance, glasssix::longinus::face_rect_with_face_info** ptr, unsigned char* image, int height, int width, int minSize, float* threshold, float factor, int stage, int order)
{
	std::vector<glasssix::longinus::face_rect_with_face_info> vec = instance->detectEx(image, 3, height, width, minSize, threshold, 1.0f / factor, stage, order);
	int num = vec.size();
	if (num)
	{
		sort_descend(vec);
		*ptr = heap_alloc_elements<glasssix::longinus::face_rect_with_face_info>(num);
		std::copy(vec.begin(), vec.end(), *ptr);
	}
	else
	{
		*ptr = nullptr;
	}

	return num;
}

int Longinus_detectEx_Mobile(glasssix::longinus::LonginusDetector* instance, glasssix::longinus::face_rect_with_face_info** ptr, unsigned char* image, int height, int width, int minSize, float* threshold, float factor, int stage, int order)
{
	std::vector<glasssix::longinus::face_rect_with_face_info> vec = instance->detectEx_mobile(image, 3, height, width, minSize, threshold, 1.0f / factor, stage, order);
	int num = vec.size();
	if (num)
	{
		sort_descend(vec);
		*ptr = heap_alloc_elements<glasssix::longinus::face_rect_with_face_info>(num);
		std::copy(vec.begin(), vec.end(), *ptr);
	}
	else
	{
		*ptr = nullptr;
	}

	return num;
}

unsigned char* Longinus_alignFace(glasssix::longinus::LonginusDetector* instance, unsigned char* ori_image, int n, int height, int width, int* bbox, int* landmarks)
{
	if (n)
	{
		std::vector<std::vector<int> > bbox_vec;
		std::vector<std::vector<int> > landmarks_vec;
		for (int i = 0; i < n; i++)
		{
			bbox_vec.push_back({ bbox[i * 4], bbox[i * 4 + 1], bbox[i * 4 + 2], bbox[i * 4 + 3] });
			landmarks_vec.push_back({ landmarks[i * 10], landmarks[i * 10 + 1], landmarks[i * 10 + 2], landmarks[i * 10 + 3], landmarks[i * 10 + 4], landmarks[i * 10 + 5],
				landmarks[i * 10 + 6], landmarks[i * 10 + 7], landmarks[i * 10 + 8], landmarks[i * 10 + 9] });
		}
		std::vector<unsigned char> vec = instance->alignFace(ori_image, n, 1, height, width, bbox_vec, landmarks_vec);
		unsigned char* aligned = heap_alloc_elements<unsigned char>(vec.size());
		std::copy(vec.begin(), vec.end(), aligned);
		return aligned;
	}
	else
		return nullptr;
}

unsigned char* Longinus_alignFaceFromCropped(glasssix::longinus::LonginusDetector* instance, unsigned char* ori_image, int n, int height, int width)
{
	if (n)
	{
		std::vector<unsigned char> vec = instance->alignFace(ori_image, n, 1, height, width);
		unsigned char* aligned = heap_alloc_elements<unsigned char>(vec.size());
		std::copy(vec.begin(), vec.end(), aligned);
		return aligned;
	}
	else
		return nullptr;
}

int Longinus_detectEx_Mobile_nir(glasssix::longinus::LonginusDetector* instance, glasssix::longinus::face_rect_with_face_info** ptr, unsigned char* image, int height, int width, int minSize, float* threshold, float factor, int stage, int order)
{
	std::vector<glasssix::longinus::face_rect_with_face_info> vec = instance->detectEx_mobile_nir(image, 3, height, width, minSize, threshold, 1.0f / factor, stage, order);
	int num = vec.size();
	if (num)
	{
		sort_descend(vec);
		*ptr = heap_alloc_elements<glasssix::longinus::face_rect_with_face_info>(num);
		std::copy(vec.begin(), vec.end(), *ptr);
	}
	else
	{
		*ptr = nullptr;
	}

	return num;
}

extern "C" LONGINUS_C_EXPORT void detectEx_mobile_pair(glasssix::longinus::LonginusDetector * instance, glasssix::longinus::face_rect_with_face_info * *vsl_rect_ptr, int* vsl_rect_num, unsigned char* vsl_image, int vsl_height, int vsl_width, int vsl_minSize, float* vsl_threshold, float vsl_factor, int vsl_stage, int vsl_order,
	glasssix::longinus::face_rect_with_face_info * *nir_rect_ptr, int* nir_rect_num, unsigned char* nir_image, int nir_height, int nir_width, int nir_minSize, float* nir_threshold, float nir_factor, int nir_stage, int nir_order)
{
	std::vector<std::vector<glasssix::longinus::face_rect_with_face_info> > result = instance->detectEx_mobile_pair(vsl_image, 3, vsl_height, vsl_width, vsl_minSize, vsl_threshold, 1.0f / vsl_factor, vsl_stage, vsl_order,
		nir_image, 3, nir_height, nir_width, nir_minSize, nir_threshold, 1.0f / nir_factor, nir_stage, nir_order);
	int vsl_num = result[0].size();
	*vsl_rect_num = vsl_num;
	if (vsl_num)
	{
		sort_descend(result[0]);
		*vsl_rect_ptr = heap_alloc_elements<glasssix::longinus::face_rect_with_face_info>(vsl_num);
		std::copy(result[0].begin(), result[0].end(), *vsl_rect_ptr);
	}
	else
	{
		*vsl_rect_ptr = nullptr;
	}

	int nir_num = result[1].size();
	*nir_rect_num = nir_num;
	if (vsl_num)
	{
		sort_descend(result[1]);
		*nir_rect_ptr = heap_alloc_elements<glasssix::longinus::face_rect_with_face_info>(nir_num);
		std::copy(result[1].begin(), result[1].end(), *nir_rect_ptr);
	}
	else
	{
		*nir_rect_ptr = nullptr;
	}
}

bool Longinus_blur_judge_vsl(glasssix::longinus::LonginusDetector* instance, unsigned char* vsl_color_image, int height, int width, int n, int* bbox, int* landmarks, float* thresh, float** value, int order)
{
	std::vector<std::vector<int> > bbox_vec;
	std::vector<std::vector<int> > landmarks_vec;

	for (int i = 0; i < n; i++)
	{
		bbox_vec.push_back({ bbox[0 + i * 4], bbox[1 + i * 4], bbox[2 + i * 4], bbox[3 + i * 4] });
		landmarks_vec.push_back({ landmarks[0 + i * 10], landmarks[1 + i * 10], landmarks[2 + i * 10], landmarks[3 + i * 10], landmarks[4 + i * 10], landmarks[5 + i * 10], landmarks[6 + i * 10],
			landmarks[6 + i * 10], landmarks[7 + i * 10], landmarks[8 + i * 10], landmarks[9 + i * 10] });
	}

	*value = heap_alloc_elements<float>(2);
	bool ret = instance->blur_judge_vsl(vsl_color_image, height, width, bbox_vec, landmarks_vec, thresh, *value, order);
	if (value)
		return true;

	heap_free(*value);
	*value = nullptr;
	return false;
}

bool Longinus_black_white_judge_vsl(glasssix::longinus::LonginusDetector* instance, unsigned char* vsl_color_image, int height, int width, int n, int* bbox, int* landmarks, float* thresh, float** value, int order)
{
	std::vector<std::vector<int> > bbox_vec;
	std::vector<std::vector<int> > landmarks_vec;

	for (int i = 0; i < n; i++)
	{
		bbox_vec.push_back({ bbox[0 + i * 4], bbox[1 + i * 4], bbox[2 + i * 4], bbox[3 + i * 4] });
		landmarks_vec.push_back({ landmarks[0 + i * 10], landmarks[1 + i * 10], landmarks[2 + i * 10], landmarks[3 + i * 10], landmarks[4 + i * 10], landmarks[5 + i * 10], landmarks[6 + i * 10],
			landmarks[6 + i * 10], landmarks[7 + i * 10], landmarks[8 + i * 10], landmarks[9 + i * 10] });
	}

	*value = heap_alloc_elements<float>(2);
	bool ret = instance->blur_judge_vsl(vsl_color_image, height, width, bbox_vec, landmarks_vec, thresh, *value, order);
	if (value)
		return true;

	heap_free(*value);
	*value = nullptr;
	return false;
}

bool Longinus_face_nose_judget_nir(glasssix::longinus::LonginusDetector* instance, unsigned char* nir_color_image, int height, int width, int n, int* bbox, int* landmarks, float* thresh, float** value, int order)
{
	std::vector<std::vector<int> > bbox_vec;
	std::vector<std::vector<int> > landmarks_vec;

	for (int i = 0; i < n; i++)
	{
		bbox_vec.push_back({ bbox[0 + i * 4], bbox[1 + i * 4], bbox[2 + i * 4], bbox[3 + i * 4] });
		landmarks_vec.push_back({ landmarks[0 + i * 10], landmarks[1 + i * 10], landmarks[2 + i * 10], landmarks[3 + i * 10], landmarks[4 + i * 10], landmarks[5 + i * 10], landmarks[6 + i * 10],
			landmarks[6 + i * 10], landmarks[7 + i * 10], landmarks[8 + i * 10], landmarks[9 + i * 10] });
	}

	*value = heap_alloc_elements<float>(2);
	bool ret = instance->blur_judge_vsl(nir_color_image, height, width, bbox_vec, landmarks_vec, thresh, *value, order);
	if (value)
		return true;

	heap_free(*value);
	*value = nullptr;
	return false;
}

unsigned char* Longinus_getVersion()
{
	auto version = glasssix::longinus::LonginusDetector::getVersion();
	std::size_t size = std::strlen(version) + 1;
	auto str = glasssix::memory::heap_alloc_elements<unsigned char>(size);

	return str;
}
