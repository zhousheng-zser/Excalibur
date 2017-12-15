#include <map>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include "LABBoostModelReader.hpp"
#include "FUSTDetector.hpp"
#include "utils.hpp"
#include "SURFMLPModelReader.hpp"
#include "SURFMLP.hpp"


namespace excalibur
{
	bool FUSTDtector::LoadModel(const std::string& model_path)
	{
		std::ifstream model_file(model_path, std::ifstream::binary);
		bool is_loaded = true;

		if (!model_file.is_open()) {
			is_loaded = false;
		}
		else {
			hierarchy_size_.clear();
			num_stage_.clear();
			wnd_src_id_.clear();

			int32_t hierarchy_size;
			int32_t num_stage;
			int32_t num_wnd_src;
			int32_t type_id;
			int32_t feat_map_index = 0;
			std::shared_ptr<ModelReader> reader;
			std::shared_ptr<Classifier> classifier;
			ClassifierType classifier_type;

			model_file.read(reinterpret_cast<char*>(&num_hierarchy_), sizeof(int32_t));
			for (int32_t i = 0; is_loaded && i < num_hierarchy_; i++) {
				model_file.read(reinterpret_cast<char*>(&hierarchy_size),
					sizeof(int32_t));
				hierarchy_size_.push_back(hierarchy_size);

				for (int32_t j = 0; is_loaded && j < hierarchy_size; j++) {
					model_file.read(reinterpret_cast<char*>(&num_stage), sizeof(int32_t));
					num_stage_.push_back(num_stage);

					for (int32_t k = 0; is_loaded && k < num_stage; k++) {
						model_file.read(reinterpret_cast<char*>(&type_id), sizeof(int32_t));
						classifier_type = static_cast<ClassifierType>(type_id);
						reader = CreateModelReader(classifier_type);
						classifier = CreateClassifier(classifier_type);

						is_loaded = !model_file.fail() &&
							reader->Read(&model_file, classifier.get());
						if (is_loaded) {
							model_.push_back(classifier);
							std::shared_ptr<FeatureMap> feat_map;
							if (cls2feat_idx_.count(classifier_type) == 0) {
								feat_map_.push_back(CreateFeatureMap(classifier_type));
								cls2feat_idx_.insert(
									std::map<ClassifierType, int32_t>::value_type(
										classifier_type, feat_map_index++));
							}
							feat_map = feat_map_[cls2feat_idx_.at(classifier_type)];
							model_.back()->SetFeatureMap(feat_map.get());
						}
					}

					wnd_src_id_.push_back(std::vector<int32_t>());
					model_file.read(reinterpret_cast<char*>(&num_wnd_src), sizeof(int32_t));
					if (num_wnd_src > 0) {
						wnd_src_id_.back().resize(num_wnd_src);
						for (int32_t k = 0; k < num_wnd_src; k++) {
							model_file.read(reinterpret_cast<char*>(&(wnd_src_id_.back()[k])),
								sizeof(int32_t));
						}
					}
				}
			}

			model_file.close();
		}
		return is_loaded;
	}

	std::vector<FaceInfo> FUSTDtector::Detect(std::shared_ptr<ImagePyramid> img_pyramid)
	{
		float score;
		FaceInfo wnd_info;
		Rect wnd;
		float scale_factor = 0.0;
		std::shared_ptr<ImageTensor<unsigned char>> img_scaled =
			img_pyramid->GetNextScaleImage(&scale_factor);

		wnd.height = wnd.width = wnd_size_;

		// Sliding window

		std::vector<std::vector<FaceInfo> > proposals(hierarchy_size_[0]);
		std::shared_ptr<FeatureMap> & feat_map_1 =
			feat_map_[cls2feat_idx_[model_[0]->type()]];

		while (img_scaled != nullptr) {
			feat_map_1->ComputeCPU(img_scaled->cpu_data(), img_scaled->width(),
				img_scaled->height());

			wnd_info.bbox.width = static_cast<int32_t>(wnd_size_ / scale_factor + 0.5);
			wnd_info.bbox.height = wnd_info.bbox.width;

			int32_t max_x = img_scaled->width() - wnd_size_;
			int32_t max_y = img_scaled->height() - wnd_size_;
			for (int32_t y = 0; y <= max_y; y += slide_wnd_step_y_) {
				wnd.y = y;
				for (int32_t x = 0; x <= max_x; x += slide_wnd_step_x_) {
					wnd.x = x;
					feat_map_1->SetROI(wnd);

					wnd_info.bbox.x = static_cast<int32_t>(x / scale_factor + 0.5);
					wnd_info.bbox.y = static_cast<int32_t>(y / scale_factor + 0.5);

					for (int32_t i = 0; i < hierarchy_size_[0]; i++) {
						if (model_[i]->Classify(&score)) {
							wnd_info.score = static_cast<double>(score);
							proposals[i].push_back(wnd_info);
						}
					}
				}
			}

			img_scaled = img_pyramid->GetNextScaleImage(&scale_factor);
		}

		std::vector<std::vector<FaceInfo> > proposals_nms(hierarchy_size_[0]);
		for (int32_t i = 0; i < hierarchy_size_[0]; i++) {
			utils::NonMaximumSuppression(&(proposals[i]),
				&(proposals_nms[i]), 0.8f);
			proposals[i].clear();
		}
		return proposals_nms[0];
	}

	void FUSTDtector::GetWindowData(const ImageTensor<unsigned char>& img, const Rect& wnd)
	{
		int32_t pad_left;
		int32_t pad_right;
		int32_t pad_top;
		int32_t pad_bottom;
		Rect roi = wnd;
		int width = img.width();
		int height = img.height();
		const unsigned char * data = img.cpu_data();

		pad_left = pad_right = pad_top = pad_bottom = 0;
		if (roi.x + roi.width > width)
			pad_right = roi.x + roi.width - width;
		if (roi.x < 0) {
			pad_left = -roi.x;
			roi.x = 0;
		}
		if (roi.y + roi.height > height)
			pad_bottom = roi.y + roi.height - height;
		if (roi.y < 0) {
			pad_top = -roi.y;
			roi.y = 0;
		}

		wnd_data_buf_.resize(roi.width * roi.height);
		const uint8_t* src = data + roi.y * width + roi.x;
		uint8_t* dest = wnd_data_buf_.data();
		int32_t len = sizeof(uint8_t) * roi.width;
		int32_t len2 = sizeof(uint8_t) * (roi.width - pad_left - pad_right);

		if (pad_top > 0) {
			std::memset(dest, 0, len * pad_top);
			dest += (roi.width * pad_top);
		}
		if (pad_left == 0) {
			if (pad_right == 0) {
				for (int32_t y = pad_top; y < roi.height - pad_bottom; y++) {
					std::memcpy(dest, src, len);
					src += width;
					dest += roi.width;
				}
			}
			else {
				for (int32_t y = pad_top; y < roi.height - pad_bottom; y++) {
					std::memcpy(dest, src, len2);
					src += width;
					dest += roi.width;
					std::memset(dest - pad_right, 0, sizeof(uint8_t) * pad_right);
				}
			}
		}
		else {
			if (pad_right == 0) {
				for (int32_t y = pad_top; y < roi.height - pad_bottom; y++) {
					std::memset(dest, 0, sizeof(uint8_t)* pad_left);
					std::memcpy(dest + pad_left, src, len2);
					src += width;
					dest += roi.width;
				}
			}
			else {
				for (int32_t y = pad_top; y < roi.height - pad_bottom; y++) {
					std::memset(dest, 0, sizeof(uint8_t) * pad_left);
					std::memcpy(dest + pad_left, src, len2);
					src += width;
					dest += roi.width;
					std::memset(dest - pad_right, 0, sizeof(uint8_t) * pad_right);
				}
			}
		}
		if (pad_bottom > 0)
			std::memset(dest, 0, len * pad_bottom);

		/*ImageData src_img(roi.width, roi.height);
		ImageData dest_img(wnd_size_, wnd_size_);
		src_img.data = wnd_data_buf_.data();
		dest_img.data = wnd_data_.data();*/
		std::shared_ptr<ImageTensor<unsigned char>> src_img = std::make_shared<ImageTensor<unsigned char>>(roi.width, roi.height, 1, -1);
		std::shared_ptr<ImageTensor<unsigned char>> dest_img = std::make_shared<ImageTensor<unsigned char>>(wnd_size_, wnd_size_, 1, -1);
		src_img->set_cpu_data(wnd_data_buf_.data());
		dest_img->set_cpu_data(wnd_data_.data());
		ImagePyramid::ResizeImageCPU(src_img, dest_img);
	}

	std::shared_ptr<ModelReader> FUSTDtector::CreateModelReader(ClassifierType type)
	{
		std::shared_ptr<ModelReader> reader;
		switch (type) {
		case ClassifierType::LAB_Boosted_Classifier:
			reader.reset(new LABBoostModelReader());
			break;
		case ClassifierType::SURF_MLP:
			reader.reset(new SURFMLPModelReader());
			break;
		default:
			break;
		}
		return reader;
	}

	std::shared_ptr<Classifier> FUSTDtector::CreateClassifier(ClassifierType type)
	{
		std::shared_ptr<Classifier> classifier;
		switch (type) {
		case ClassifierType::LAB_Boosted_Classifier:
			classifier.reset(new LABBoostedClassifier());
			break;
		case ClassifierType::SURF_MLP:
			classifier.reset(new SURFMLP());
			break;
		default:
			break;
		}
		return classifier;
	}

	std::shared_ptr<FeatureMap> FUSTDtector::CreateFeatureMap(ClassifierType type)
	{
		std::shared_ptr<FeatureMap> feat_map;
		switch (type) {
		case ClassifierType::LAB_Boosted_Classifier:
			feat_map.reset(new LABFeatureMap());
			break;
		case ClassifierType::SURF_MLP:
			feat_map.reset(new SURFFeatureMap());
			break;
		default:
			break;
		}
		return feat_map;
	}

}