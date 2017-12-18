#include "LABFeatureMap.hpp"

namespace excalibur
{
	void LABFeatureMap::ComputeCPU(const uint8_t* input, int32_t width,
		int32_t height) {
		if (input == nullptr || width <= 0 || height <= 0) {
			return;  // @todo handle the errors!!!
		}

		Reshape(width, height);
		ComputeIntegralImagesCPU(input);
		ComputeRectSumCPU();
		
		rect_sum_data = rect_sum_->cpu_data();
		feat_map_data = feat_map_->mutable_cpu_data();
		ComputeFeatureMapCPU();

		int_img_data = int_img_->cpu_data();
		square_int_img_data = square_int_img_->cpu_data();
		feat_map_cpu_data = feat_map_->cpu_data();
	}

	template<typename Dtype>
	void LABFeatureMap::IntegralCPU(std::shared_ptr<ImageTensor<Dtype>> data)
	{
		const Dtype* src = data->cpu_data();
		Dtype* dest = data->mutable_cpu_data();
		const Dtype* dest_above = dest;
		*dest = *(src++);

		for (int32_t c = 1; c < width_; c++, src++, dest++)
		{
			*(dest + 1) = (*dest) + (*src);
		}
		dest++;
		for (int32_t r = 1; r < height_; r++) {
			for (int32_t c = 0, s = 0; c < width_; c++, src++, dest++, dest_above++) {
				s += (*src);
				*dest = *dest_above + s;
			}
		}
	}

	template
	void LABFeatureMap::IntegralCPU(std::shared_ptr<ImageTensor<int>> data);

	template
	void LABFeatureMap::IntegralCPU(std::shared_ptr<ImageTensor<unsigned int>> data);
	

	float LABFeatureMap::GetStdDev() const {
		double mean;
		double m2;
		double area = roi_.width * roi_.height;

		int32_t top_left;
		int32_t top_right;
		int32_t bottom_left;
		int32_t bottom_right;

		if (roi_.x != 0) {
			if (roi_.y != 0) {
				top_left = (roi_.y - 1) * width_ + roi_.x - 1;
				top_right = top_left + roi_.width;
				bottom_left = top_left + roi_.height * width_;
				bottom_right = bottom_left + roi_.width;

				mean = (int_img_data[bottom_right] - int_img_data[bottom_left] +
					int_img_data[top_left] - int_img_data[top_right]) / area;
				m2 = (square_int_img_data[bottom_right] - square_int_img_data[bottom_left] +
					square_int_img_data[top_left] - square_int_img_data[top_right]) / area;
			}
			else {
				bottom_left = (roi_.height - 1) * width_ + roi_.x - 1;
				bottom_right = bottom_left + roi_.width;

				mean = (int_img_data[bottom_right] - int_img_data[bottom_left]) / area;
				m2 = (square_int_img_data[bottom_right] - square_int_img_data[bottom_left]) / area;
			}
		}
		else {
			if (roi_.y != 0) {
				top_right = (roi_.y - 1) * width_ + roi_.width - 1;
				bottom_right = top_right + roi_.height * width_;

				mean = (int_img_data[bottom_right] - int_img_data[top_right]) / area;
				m2 = (square_int_img_data[bottom_right] - square_int_img_data[top_right]) / area;
			}
			else {
				bottom_right = (roi_.height - 1) * width_ + roi_.width - 1;
				mean = int_img_data[bottom_right] / area;
				m2 = square_int_img_data[bottom_right] / area;
			}
		}

		return static_cast<float>(std::sqrt(m2 - mean * mean));
	}

	void LABFeatureMap::Reshape(int32_t width, int32_t height) {
		width_ = width;
		height_ = height;

		int32_t len = width_ * height_;
		feat_map_.reset(new ImageTensor<unsigned char>(1, 1, len, device_));
		rect_sum_.reset(new ImageTensor<int>(1, 1, len, device_));
		int_img_.reset(new ImageTensor<int>(1, 1, len, device_));
		square_int_img_.reset(new ImageTensor<unsigned int>(1, 1, len, device_));
	}

	void LABFeatureMap::ComputeIntegralImagesCPU(const uint8_t* input) {
		int32_t len = width_ * height_;

		MathHelper::UInt8ToInt32CPU(input, int_img_->mutable_cpu_data(), len);
		MathHelper::SquareCPU(int_img_->cpu_data(), square_int_img_->mutable_cpu_data(), len);
		IntegralCPU(int_img_);
		IntegralCPU(square_int_img_);
	}

	void LABFeatureMap::ComputeRectSumCPU() {
		int32_t width = width_ - rect_width_;
		int32_t height = height_ - rect_height_;
		const int32_t* int_img = int_img_->cpu_data();
		int32_t* rect_sum = rect_sum_->mutable_cpu_data();
		*rect_sum = *(int_img + (rect_height_ - 1) * width_ + rect_width_ - 1);
		MathHelper::VectorSubCPU(int_img + (rect_height_ - 1) * width_ +
			rect_width_, int_img + (rect_height_ - 1) * width_, rect_sum + 1, width);
		
#ifdef _OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
		{
#pragma omp for nowait
#endif
			for (int32_t i = 1; i <= height; i++) {
				const int32_t* top_left = int_img + (i - 1) * width_;
				const int32_t* top_right = top_left + rect_width_ - 1;
				const int32_t* bottom_left = top_left + rect_height_ * width_;
				const int32_t* bottom_right = bottom_left + rect_width_ - 1;
				int32_t* dest = rect_sum + i * width_;

				*(dest++) = (*bottom_right) - (*top_right);
				MathHelper::VectorSubCPU(bottom_right + 1, top_right + 1, dest, width);
				MathHelper::VectorSubCPU(dest, bottom_left, dest, width);
				MathHelper::VectorAddCPU(dest, top_left, dest, width);
			}
#ifdef _OPENMP
		}
#endif
		
	}

	void LABFeatureMap::ComputeFeatureMapCPU() {
		int32_t width = width_ - rect_width_ * num_rect_;
		int32_t height = height_ - rect_height_ * num_rect_;
		int32_t offset = width_ * rect_height_;

#ifdef _OPENMP
#pragma omp parallel num_threads(OMP_NUM_THREADS)
		{
#pragma omp for nowait
#endif
			for (int32_t r = 0; r <= height; r++) {
				for (int32_t c = 0; c <= width; c++) {
					uint8_t* dest = feat_map_data + r * width_ + c;
					*dest = 0;

					int32_t white_rect_sum = rect_sum_data[(r + rect_height_) * width_ + c + rect_width_];
					int32_t black_rect_idx = r * width_ + c;
					*dest |= (white_rect_sum >= rect_sum_data[black_rect_idx] ? 0x80 : 0x0);
					black_rect_idx += rect_width_;
					*dest |= (white_rect_sum >= rect_sum_data[black_rect_idx] ? 0x40 : 0x0);
					black_rect_idx += rect_width_;
					*dest |= (white_rect_sum >= rect_sum_data[black_rect_idx] ? 0x20 : 0x0);
					black_rect_idx += offset;
					*dest |= (white_rect_sum >= rect_sum_data[black_rect_idx] ? 0x08 : 0x0);
					black_rect_idx += offset;
					*dest |= (white_rect_sum >= rect_sum_data[black_rect_idx] ? 0x01 : 0x0);
					black_rect_idx -= rect_width_;
					*dest |= (white_rect_sum >= rect_sum_data[black_rect_idx] ? 0x02 : 0x0);
					black_rect_idx -= rect_width_;
					*dest |= (white_rect_sum >= rect_sum_data[black_rect_idx] ? 0x04 : 0x0);
					black_rect_idx -= offset;
					*dest |= (white_rect_sum >= rect_sum_data[black_rect_idx] ? 0x10 : 0x0);
				}
			}
#ifdef _OPENMP
		}
#endif
	}

#ifdef USE_CUDA
	void LABFeatureMap::ComputeGPU(const uint8_t* input, int32_t width, int32_t height)
	{
		if (input == nullptr || width <= 0 || height <= 0) {
			return; 
		}
		
		Reshape(width, height);
		ComputeIntegralImagesGPU(input);
		ComputeRectSumGPU();
		
		rect_sum_data = rect_sum_->gpu_data();
		feat_map_data = feat_map_->mutable_gpu_data();
		ComputeFeatureMapGPU();
		
		//std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();

		int_img_data = int_img_->cpu_data();
		square_int_img_data = square_int_img_->cpu_data();
		feat_map_cpu_data = feat_map_->cpu_data();
		/*std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
		std::cout << "total detection time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << std::endl << std::endl;*/
	}


	void LABFeatureMap::ComputeIntegralImagesGPU(const unsigned char* input)
	{
		int32_t len = width_ * height_;
		/*std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();*/
		MathHelper::UInt8ToInt32GPU(input, int_img_->mutable_gpu_data(), len);
		MathHelper::SquareGPU(int_img_->gpu_data(), square_int_img_->mutable_gpu_data(), len);
		IntegralGPU(int_img_);
		IntegralGPU(square_int_img_);
		/*std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
		std::cout << "total detection time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << std::endl << std::endl;*/
	}
#endif
}