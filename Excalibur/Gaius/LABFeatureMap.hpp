#pragma once
#ifndef _LABFEATUREMAP_HPP_
#define _LABFEATUREMAP_HPP_

#include "FeatureMap.hpp"
#include "math_helper.hpp"

namespace excalibur
{
	typedef struct LABFeature {
		int32_t x;
		int32_t y;
	} LABFeature;

	class LABFeatureMap: public FeatureMap
	{
	public:
		LABFeatureMap() : rect_width_(3), rect_height_(3), num_rect_(3)
		{
			device_ = -1;
		}

		LABFeatureMap(int device) :rect_width_(3), rect_height_(3), num_rect_(3), device_(device){};

		virtual ~LABFeatureMap() {}

		virtual void ComputeCPU(const uint8_t* input, int32_t width, int32_t height);

#ifdef USE_CUDA
		virtual void ComputeGPU(const uint8_t* input, int32_t width,
			int32_t height)
		{
			NOT_IMPLEMENTED;
		}
#endif

		uint8_t GetFeatureVal(int32_t offset_x, int32_t offset_y) const {
			return feat_map_data[(roi_.y + offset_y) * width_ + roi_.x + offset_x];
		}

		float GetStdDev() const;

	private:
		void Reshape(int32_t width, int32_t height);
		void ComputeIntegralImagesCPU(const uint8_t* input);
		void ComputeRectSumCPU();
		void ComputeFeatureMapCPU();

		template<typename Dtype>
		void IntegralCPU(std::shared_ptr<ImageTensor<Dtype>> data)
		{
			const Dtype* src = data->cpu_data();
			Dtype* dest = data->mutable_cpu_data();
			const Dtype* dest_above = dest;
			*dest = *(src++);
			for (int32_t c = 1; c < width_; c++, src++, dest++)
				*(dest + 1) = (*dest) + (*src);
			dest++;
			for (int32_t r = 1; r < height_; r++) {
				for (int32_t c = 0, s = 0; c < width_; c++, src++, dest++, dest_above++) {
					s += (*src);
					*dest = *dest_above + s;
				}
			}
		}

#ifdef USE_CUDA
		void ComputeIntegralImagesGPU(const unsigned char* input);
		void ComputeRectSumGPU();
		void ComputeFeatureMapGPU();
#endif

		const int32_t rect_width_;
		const int32_t rect_height_;
		const int32_t num_rect_;

		/*std::vector<uint8_t> feat_map_;
		std::vector<int32_t> rect_sum_;
		std::vector<int32_t> int_img_;
		std::vector<uint32_t> square_int_img_;*/
		std::shared_ptr<ImageTensor<unsigned char>> feat_map_;
		std::shared_ptr<ImageTensor<int>> rect_sum_;
		std::shared_ptr<ImageTensor<int>> int_img_;
		std::shared_ptr<ImageTensor<unsigned int>> square_int_img_;
		unsigned char* feat_map_data;
		const int* rect_sum_data;
		const int* int_img_data;
		const unsigned int* square_int_img_data;

		int device_;
	};
}

#endif // _LABFEATUREMAP_HPP_