#ifndef _ROMANCIADETECTOR_HPP_
#define _ROMANCIADETECTOR_HPP_

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include "BaseLonginusCascade.hpp"
#include "../Romancia/romancia.hpp"
#include "../Selene/selene.hpp"
#include "matcher.hpp"
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif
#ifndef TRIAL
#include "../Damocles/damocles.hpp"
#endif // !TRIAL

namespace glasssix
{
	namespace longinus
	{
		enum longinus_detection_type
		{
			FRONTALVIEW,
			FRONTALVIEW_REINFORCE,
			MULTIVIEW,
			MULTIVIEW_REINFORCE
		};

		struct face_rect_with_face_info : public face_rect_basic
		{
			Point2f pts[5];
			float yaw;
			float pitch;
			float roll;
			//float prob;
			float clarity;

			face_rect_with_face_info() : yaw{}, pitch{}, roll{}, clarity{}
			{
			}

			face_rect_with_face_info(const face_rect_basic& rect) : yaw{}, pitch{}, roll{}
			{
				*dynamic_cast<face_rect_basic*>(this) = rect;
				this->clarity = 0.0f;
			}

			face_rect_with_face_info& operator=(const face_rect_basic& rect)
			{
				*dynamic_cast<face_rect_basic*>(this) = rect;
				this->clarity = 0.0f;
				return *this;
			}
		};

#ifdef USE_OPENCV
		static std::vector<cv::Mat> encode2mats(std::vector<unsigned char> face_data, int face_count)
		{
			std::vector<cv::Mat> faces;
			assert(face_data.size() == face_count * 3 * 128 * 128);
			for (size_t i = 0; i < face_count; i++)
			{
				unsigned char* src_data = new unsigned char[3 * 128 * 128];
				memcpy(src_data, face_data.data() + i * 3 * 128 * 128, 3 * 128 * 128);
				cv::Mat dst = cv::Mat(128, 128, CV_8UC3);
				int src_offset = 128 * 128;
				int* c_src_offset = new int[3];
				for (int c = 0; c < 3; c++)
				{
					c_src_offset[c] = c * src_offset;
				}
				for (int h = 0; h < 128; h++)
				{
					unsigned char* dst_data = dst.ptr<unsigned char>(h);
					int src_sub_offset = h * 128;
					for (int w = 0; w < 128; w++)
					{
						for (int c = 0; c < 3; c++)
						{
							dst_data[w * 3 + c] = src_data[c_src_offset[c] + src_sub_offset + w];
						}
					}
				}
				delete c_src_offset;
				faces.push_back(dst);
				delete src_data;
			}
			return faces;
		}
#endif

		static void extract_faceinfo(std::vector<face_rect_with_face_info> face_info,
			std::vector<std::vector<int>>& bboxes, std::vector<std::vector<int>>& landmarks)
		{
			bboxes.clear();
			landmarks.clear();
			for (size_t i = 0; i < face_info.size(); i++)
			{
				bboxes.push_back({ face_info[i].x, face_info[i].y, face_info[i].width, face_info[i].height });
				landmarks.push_back({ static_cast<int>(face_info[i].pts[0].x), static_cast<int>(face_info[i].pts[0].y),
													  static_cast<int>(face_info[i].pts[1].x), static_cast<int>(face_info[i].pts[1].y),
													  static_cast<int>(face_info[i].pts[2].x), static_cast<int>(face_info[i].pts[2].y),
													  static_cast<int>(face_info[i].pts[3].x), static_cast<int>(face_info[i].pts[3].y),
													  static_cast<int>(face_info[i].pts[4].x), static_cast<int>(face_info[i].pts[4].y) });
			}
		}

		static int extract_biggest_faceinfo(std::vector<face_rect_with_face_info> face_info,
			std::vector<std::vector<int>>& bboxes, std::vector<std::vector<int>>& landmarks)
		{
			bboxes.clear();
			landmarks.clear();
			int index = 0;
			float area = 0.0f;
			for (size_t i = 0; i < face_info.size(); i++)
			{
				if (face_info[i].width * face_info[i].height > area)
				{
					index = i;
					area = face_info[i].width * face_info[i].height;
				}
			}
			if (area <= 0.0f)
			{
				return -1;
			}
			bboxes.push_back({ face_info[index].x, face_info[index].y, face_info[index].width, face_info[index].height });
			landmarks.push_back({ static_cast<int>(face_info[index].pts[0].x), static_cast<int>(face_info[index].pts[0].y),
												  static_cast<int>(face_info[index].pts[1].x), static_cast<int>(face_info[index].pts[1].y),
												  static_cast<int>(face_info[index].pts[2].x), static_cast<int>(face_info[index].pts[2].y),
												  static_cast<int>(face_info[index].pts[3].x), static_cast<int>(face_info[index].pts[3].y),
												  static_cast<int>(face_info[index].pts[4].x), static_cast<int>(face_info[index].pts[4].y) });

			return index;
		}

		static void sort_descend(std::vector<face_rect_basic>& rects)
		{
			std::sort(rects.begin(), rects.end());
		}

		static void sort_descend(std::vector<face_rect_with_face_info>& rects)
		{
			std::sort(rects.begin(), rects.end());
		}

		class EXPORT_LONGINUS LonginusDetector
		{
		public:
			class impl;

			LonginusDetector(int device = -1);
			virtual ~LonginusDetector();
			std::vector<face_rect_basic> detect(unsigned char* gray, int width, int height, int step, int minSize, float scale,
				int minNeighbors, bool useMultiThreads = false, bool doEarlyReject = false);
			std::vector<face_rect_with_face_info> detect(unsigned char* gray, int width, int height, int step, int minSize, float scale,
				int minNeighbors, int order = 0, bool useMultiThreads = false, bool doEarlyReject = false);

			std::vector<Match_Retval> match(std::vector<face_rect_basic>& faceRect, const int frame_extract_frequency, float distance_fractor = 1.0f) const;

			std::vector<Match_Retval> match(std::vector<face_rect_with_face_info>& faceRect, const int frame_extract_frequency, float distance_fractor = 1.0f) const;

			std::vector<unsigned char> alignFace(const unsigned char* ori_image, int n, int channels, int height, int width,
				std::vector<std::vector<int>> bbox, std::vector<std::vector<int> >landmarks) const;

			std::vector<unsigned char> alignFace(const unsigned char* ori_image, int n, int channels, int height, int width) const;

#ifndef TRIAL
			std::vector<face_rect_with_face_info> detectEx(const unsigned char* image, const int channels, const int height, const int width,
				const int minSize, const float* threshold, const float factor, const int stage, const int order = 1) const;

			std::vector<face_rect_with_face_info> detectEx_mobile(const unsigned char* image, const int channels, const int height, const int width,
				const int minSize, const float* threshold, const float factor, const int stage, const int order = 1) const;

			std::vector<face_rect_with_face_info> detectEx_mobile_nir(const unsigned char* image, const int channels, const int height, const int width,
				const int minSize, const float* threshold, const float factor, const int stage, const int order = 1) const;

			std::vector<std::vector<face_rect_with_face_info>> detectEx_mobile_pair(const unsigned char* vsl_image, const int vsl_channels, const int vsl_height, const int vsl_width,
				const int vsl_minSize, const float* vsl_threshold, const float vsl_factor, const int vsl_stage, const int vsl_order,
				const unsigned char* nir_image, int nir_channels = 0, int nir_height = 0, int nir_width = 0,
				int nir_minSize = 0, const float* nir_threshold = nullptr, float nir_factor = 0, int nir_stage = 0, int nir_order = 1) const;

			bool blur_judge_vsl(const unsigned char* vsl_color_image, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks, float thresh[2], float value[2], int order) const;

			bool black_white_judge_vsl(const unsigned char* vsl_color_image, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks, float thresh[2], float value[2], int order) const;

			bool face_nose_judge_nir(const unsigned char* nir_color_image, int height, int width, std::vector<std::vector<int>> bbox, std::vector<std::vector<int>> landmarks, float thresh[2], float value[2], int order) const;

#endif // !TRIAL

#ifndef RELEASE_SDK
			void load(std::vector<std::string> cascades, int device = -1);
#endif
			void set(longinus_detection_type detectionType, int device = -1);

			static const char* getVersion();
		private:
			impl* impl_;
		};
	}
}

#endif// !_ROMANCIADETECTOR_HPP_
