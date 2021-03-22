#pragma once
#ifndef _ULTRA_FACE_HPP_
#define _ULTRA_FACE_HPP_

#include <opencv2/opencv.hpp>
#include "../../include/Excalibur/pipeline.hpp"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#define num_featuremap 4
#define hard_nms 1
#define blending_nms 2 /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/

typedef struct FaceInfo {
	float x1;
	float y1;
	float x2;
	float y2;
	float score;

	float *landmarks;
} FaceInfo;

namespace glasssix
{
	class ultra_face
	{
	public:
		ultra_face(const std::string param_path, const std::string bin_path,
			int input_width, int input_length, int num_thread_ = 4, float score_threshold_ = 0.7, float iou_threshold_ = 0.3, int topk_ = -1);
		~ultra_face();

		std::vector<FaceInfo> detect(cv::Mat &img);

	private:
		void generateBBox(std::vector<FaceInfo> &bbox_collection, std::shared_ptr<memory::tensor<float>> scores,
			std::shared_ptr<memory::tensor<float>> boxes, float score_threshold, int num_anchors);

		void nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type = blending_nms);

		template <typename Dtype>
		void mat2tensor_cpu(const cv::Mat &srcu, std::shared_ptr<glasssix::memory::tensor<Dtype>>& dst,
			glasssix::memory::orderType order = glasssix::memory::NCHW, bool bgr2rgb = false)
		{
			if (srcu.data == NULL)
			{
				LOG(ERROR) << "No data.";
				return;
			}

			int channels = srcu.channels();
			int width = srcu.cols;
			int height = srcu.rows;
			cv::Mat src;
			srcu.convertTo(src, CV_32F);
			/*int type_id = src.type() % 8;
			auto type_name = std::string(typeid(Dtype).name());*/

			if (order == memory::NCHW)
			{
				dst.reset(new memory::tensor<Dtype>(std::vector<int>{1, channels, height, width}, -1, memory::NCHW));
				Dtype* dst_data = dst->mutable_cpu_data();
				int dst_offset = width * height;
				int* c_dst_offset = new int[channels];

				for (int c = 0; c < channels; c++)
				{
					if (bgr2rgb)
					{
						c_dst_offset[c] = (channels - 1 - c) * dst_offset;
					}
					else
					{
						c_dst_offset[c] = c * dst_offset;
					}
				}

				for (int c = 0; c < channels; c++)
				{
					for (int h = 0; h < height; h++)
					{
						const Dtype* src_data = src.ptr<Dtype>(h);
						int dst_sub_offset = h * width;

						for (int w = 0; w < width; w++)
						{
							dst_data[c_dst_offset[c] + dst_sub_offset + w] =
								src_data[w * channels + c];
						}
					}
				}

				delete[] c_dst_offset;
			}
			else if (order == memory::NHWC)
			{
				dst.reset(new memory::tensor<Dtype>(std::vector<int>{1, height, width, channels}, -1, memory::NHWC));
				Dtype* dst_data = dst->mutable_cpu_data();
				memcpy(dst_data, src.data, height * width * channels * sizeof(Dtype));
			}
			else
			{
				NOT_IMPLEMENTED;
			}
		}

		std::shared_ptr<glasssix::excalibur::pipeline<float>> pipe;
		int device_;

		int num_thread;
		int image_w;
		int image_h;

		int in_w;
		int in_h;
		int num_anchors;

		int topk;
		float score_threshold;
		float iou_threshold;


		const float mean_vals[3] = { 127, 127, 127 };
		const float norm_vals[3] = { 1.0 / 128, 1.0 / 128, 1.0 / 128 };

		const float center_variance = 0.1;
		const float size_variance = 0.2;
		const std::vector<std::vector<float>> min_boxes = {
				{10.0f,  16.0f,  24.0f},
				{32.0f,  48.0f},
				{64.0f,  96.0f},
				{128.0f, 192.0f, 256.0f} };
		const std::vector<float> strides = { 8.0, 16.0, 32.0, 64.0 };
		std::vector<std::vector<float>> featuremap_size;
		std::vector<std::vector<float>> shrinkage_size;
		std::vector<int> w_h_list;

		std::vector<std::vector<float>> priors = {};
	};
}

#endif // !_ULTRA_FACE_HPP_