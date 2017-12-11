#pragma once
#ifndef _IMAGE_TENSOR_HPP_
#define _IMAGE_TENSOR_HPP_

#include "../Excalibur/syncedmem.hpp"

#ifdef _OPENMP
#include <omp.h>
#define OMP_NUM_THREADS 8
#else
#define OMP_NUM_THREADS 1
#endif

namespace excalibur
{
	typedef struct Rect {
		int32_t x;
		int32_t y;
		int32_t width;
		int32_t height;
	} Rect;

	typedef struct FaceInfo {
		excalibur::Rect bbox;
		// 3 headpose
		float roll;
		float pitch;
		float yaw;
		/**< Larger score should mean higher confidence. */
		float score; 
	} FaceInfo;

	typedef struct {
		float x;
		float y;
	} FacialLandmark;

	template <typename Dtype>
	class ImageTensor
	{
		syncedmem* data_;
		std::vector<int> shape_;
		int count_;
		int device_;
	public:
		ImageTensor(const std::vector<int>& shape, int device);
		ImageTensor(int img_width, int img_height, int32_t img_num_channels = 1, int device = -1);
		~ImageTensor();

		const Dtype* cpu_data() const;
		const Dtype* gpu_data() const;
		Dtype* mutable_cpu_data() const;
		Dtype* mutable_gpu_data() const;
		void set_cpu_data(Dtype* data);
		void set_gpu_data(Dtype* data);

		int num() const { return shape_[0]; }
		int channels() const { return shape_[1]; }
		int height() const { return shape_[2]; }
		int width() const { return shape_[3]; }
	};
}


#endif // _IMAGE_TENSOR_HPP_