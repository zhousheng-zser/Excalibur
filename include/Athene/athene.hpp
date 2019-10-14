#pragma once
#ifndef _ATHENE_HPP_
#define _ATHENE_HPP_

#ifdef EXPORT_ATHENE
#undef EXPORT_ATHENE
#ifdef _MSC_VER // For Windows
#ifdef _WINDLL // Dynamic lib
#define EXPORT_ATHENE __declspec(dllexport)
#else // Static lib
#define EXPORT_ATHENE
#endif // !_WINDLL
#elif defined(__linux__) // For Linux
#define EXPORT_ATHENE
#endif
#else
#ifdef _MSC_VER
#define EXPORT_ATHENE __declspec(dllimport)
#elif defined(__linux__)
#define EXPORT_ATHENE
#endif
#endif

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"



struct BlobData;
namespace glasssix 
{
	class simple_window;

	namespace ozymandias
	{
		class video_array_renderer;
	}

	namespace athene
	{
		class EXPORT_ATHENE Athene
		{
		public:
			Athene() {};

			Athene(const char *stream, const char *deploy, const char *caffemodel, int base_height, int base_width,  int device = 0);
			
			//using video stream
			void Forward(unsigned char* image_data, int height, int width, std::vector<int> &lines, std::vector<int> &circles);

			void Forward();

			//using single image
			void Forward(cv::Mat &image);

			void Forward_cv();

			~Athene();

		private:
			int base_height_;
			int base_width_;
			const char *stream_;
			const char *deploy_;
			const char *caffemodel_;
			int device_;
			BlobData* nms_out_;
			BlobData* input_;
			cv::Size input_size_;
			cv::Size baseSize_;
			simple_window* window_;
			glasssix::ozymandias::video_array_renderer* renderer_;
		};
	}
}
#endif//!USE_OPENCV
#endif///!_ATHENE_HPP_