#include "mtcnn.hpp"

using namespace std;
using namespace cv;

namespace glasssix
{

	bool CompareBBox(const FaceInfoX & a, const FaceInfoX & b) 
	{
		return a.bbox.score > b.bbox.score;
	}

	MTCNN::MTCNN(int device_id) {
		device_id_ = device_id;
		omp_set_num_threads(threads_num);
		PNet_ = new mtcnn_pnet(device_id_);
		RNet_ = new mtcnn_rnet(device_id_);
		ONet_ = new mtcnn_onet(device_id_);
	}

	float MTCNN::IoU(float xmin, float ymin, float xmax, float ymax,
		float xmin_, float ymin_, float xmax_, float ymax_, bool is_iom) 
	{
		float iw = std::min(xmax, xmax_) - std::max(xmin, xmin_) + 1;
		float ih = std::min(ymax, ymax_) - std::max(ymin, ymin_) + 1;
		if (iw <= 0 || ih <= 0)
			return 0;
		float s = iw*ih;
		if (is_iom) {
			float ov = s / min((xmax - xmin + 1)*(ymax - ymin + 1), (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1));
			return ov;
		}
		else {
			float ov = s / ((xmax - xmin + 1)*(ymax - ymin + 1) + (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1) - s);
			return ov;
		}
	}

	std::vector<FaceInfoX> MTCNN::NMS(std::vector<FaceInfoX>& bboxes,
		float thresh, char methodType) {
		std::vector<FaceInfoX> bboxes_nms;
		if (bboxes.size() == 0) {
			return bboxes_nms;
		}
		std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

		int32_t select_idx = 0;
		int32_t num_bbox = static_cast<int32_t>(bboxes.size());
		std::vector<int32_t> mask_merged(num_bbox, 0);
		bool all_merged = false;

		while (!all_merged) {
			while (select_idx < num_bbox && mask_merged[select_idx] == 1)
				select_idx++;
			if (select_idx == num_bbox) {
				all_merged = true;
				continue;
			}

			bboxes_nms.push_back(bboxes[select_idx]);
			mask_merged[select_idx] = 1;

			FaceBox select_bbox = bboxes[select_idx].bbox;
			float area1 = static_cast<float>((select_bbox.xmax - select_bbox.xmin + 1) * (select_bbox.ymax - select_bbox.ymin + 1));
			float x1 = static_cast<float>(select_bbox.xmin);
			float y1 = static_cast<float>(select_bbox.ymin);
			float x2 = static_cast<float>(select_bbox.xmax);
			float y2 = static_cast<float>(select_bbox.ymax);

			select_idx++;
#pragma omp parallel for num_threads(threads_num)
			for (int32_t i = select_idx; i < num_bbox; i++) {
				if (mask_merged[i] == 1)
					continue;

				FaceBox & bbox_i = bboxes[i].bbox;
				float x = std::max<float>(x1, static_cast<float>(bbox_i.xmin));
				float y = std::max<float>(y1, static_cast<float>(bbox_i.ymin));
				float w = std::min<float>(x2, static_cast<float>(bbox_i.xmax)) - x + 1;
				float h = std::min<float>(y2, static_cast<float>(bbox_i.ymax)) - y + 1;
				if (w <= 0 || h <= 0)
					continue;

				float area2 = static_cast<float>((bbox_i.xmax - bbox_i.xmin + 1) * (bbox_i.ymax - bbox_i.ymin + 1));
				float area_intersect = w * h;

				switch (methodType) {
				case 'u':
					if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > thresh)
						mask_merged[i] = 1;
					break;
				case 'm':
					if (static_cast<float>(area_intersect) / std::min(area1, area2) > thresh)
						mask_merged[i] = 1;
					break;
				default:
					break;
				}
			}
		}
		return bboxes_nms;
	}

	void MTCNN::BBoxRegression(std::vector<FaceInfoX>& bboxes) 
	{
#pragma omp parallel for num_threads(threads_num)
		for (int i = 0; i < bboxes.size(); ++i) {
			FaceBox &bbox = bboxes[i].bbox;
			float *bbox_reg = bboxes[i].bbox_reg;
			float w = bbox.xmax - bbox.xmin + 1;
			float h = bbox.ymax - bbox.ymin + 1;
			bbox.xmin += bbox_reg[0] * w;
			bbox.ymin += bbox_reg[1] * h;
			bbox.xmax += bbox_reg[2] * w;
			bbox.ymax += bbox_reg[3] * h;
		}
	}

	void MTCNN::BBoxPad(std::vector<FaceInfoX>& bboxes, int width, int height) 
	{
#pragma omp parallel for num_threads(threads_num)
		for (int i = 0; i < bboxes.size(); ++i) {
			FaceBox &bbox = bboxes[i].bbox;
			bbox.xmin = round(std::max(bbox.xmin, 0.f));
			bbox.ymin = round(std::max(bbox.ymin, 0.f));
			bbox.xmax = round(std::min(bbox.xmax, width - 1.f));
			bbox.ymax = round(std::min(bbox.ymax, height - 1.f));
		}
	}

	void MTCNN::BBoxPadSquare(std::vector<FaceInfoX>& bboxes, int width, int height) 
	{
#pragma omp parallel for num_threads(threads_num)
		for (int i = 0; i < bboxes.size(); ++i) {
			FaceBox &bbox = bboxes[i].bbox;
			float w = bbox.xmax - bbox.xmin + 1;
			float h = bbox.ymax - bbox.ymin + 1;
			float side = h>w ? h : w;
			bbox.xmin = round(max(bbox.xmin + (w - side)*0.5f, 0.f));

			bbox.ymin = round(max(bbox.ymin + (h - side)*0.5f, 0.f));
			bbox.xmax = round(min(bbox.xmin + side - 1, width - 1.f));
			bbox.ymax = round(min(bbox.ymin + side - 1, height - 1.f));
		}
	}

	void MTCNN::GenerateBBox(std::shared_ptr<tensor<float>> confidence, std::shared_ptr<tensor<float>> reg_box,
		float scale, float thresh) 
	{
		int feature_map_w_ = confidence->width();
		int feature_map_h_ = confidence->height();
		int spatical_size = feature_map_w_*feature_map_h_;
		const float* confidence_data = confidence->cpu_data() + spatical_size;
		const float* reg_data = reg_box->cpu_data();
		candidate_boxes_.clear();
		for (int i = 0; i<spatical_size; i++) {
			if (confidence_data[i] >= thresh) {

				int y = i / feature_map_w_;
				int x = i - feature_map_w_ * y;
				FaceInfoX FaceInfoX;
				FaceBox &faceBox = FaceInfoX.bbox;

				faceBox.xmin = (float)(x * pnet_stride) / scale;
				faceBox.ymin = (float)(y * pnet_stride) / scale;
				faceBox.xmax = (float)(x * pnet_stride + pnet_cell_size - 1.f) / scale;
				faceBox.ymax = (float)(y * pnet_stride + pnet_cell_size - 1.f) / scale;

				FaceInfoX.bbox_reg[0] = reg_data[i];
				FaceInfoX.bbox_reg[1] = reg_data[i + spatical_size];
				FaceInfoX.bbox_reg[2] = reg_data[i + 2 * spatical_size];
				FaceInfoX.bbox_reg[3] = reg_data[i + 3 * spatical_size];

				faceBox.score = confidence_data[i];
				candidate_boxes_.push_back(FaceInfoX);
			}
		}
	}

	vector<FaceInfoX> MTCNN::ProposalNet(const cv::Mat& img, int minSize, float threshold, float factor) {
		cv::Mat  resized;
		int width = img.cols;
		int height = img.rows;
		float scale = 12.f / minSize;
		float minWH = std::min(height, width) *scale;
		std::vector<float> scales;
		while (minWH >= 12) {
			scales.push_back(scale);
			minWH *= factor;
			scale *= factor;
		}
		total_boxes_.clear();
		std::shared_ptr<tensor<float>> input_layer = nullptr;
		for (int i = 0; i < scales.size(); i++) {
			int ws = (int)std::ceil(width*scales[i]);
			int hs = (int)std::ceil(height*scales[i]);
			cv::resize(img, resized, cv::Size(ws, hs), 0, 0, cv::INTER_LINEAR);
			input_layer.reset(new tensor<float>(std::vector<int>{ 1, 3, hs, ws }, device_id_));
			float * input_data = input_layer->mutable_cpu_data();
			cv::Vec3b * img_data = (cv::Vec3b *)resized.data;
			int spatial_size = ws* hs;
			for (int k = 0; k < spatial_size; ++k) {
				input_data[k] = float((img_data[k][0] - mean_val)* std_val);
				input_data[k + spatial_size] = float((img_data[k][1] - mean_val) * std_val);
				input_data[k + 2 * spatial_size] = float((img_data[k][2] - mean_val) * std_val);
			}
			PNet_->Forward(input_layer);

			std::shared_ptr<tensor<float>> confidence = PNet_->get_prob1();
			std::shared_ptr<tensor<float>> reg = PNet_->get_conv4_2();
			GenerateBBox(confidence, reg, scales[i], threshold);
			std::vector<FaceInfoX> bboxes_nms = NMS(candidate_boxes_, 0.5, 'u');
			if (bboxes_nms.size()>0) {
				total_boxes_.insert(total_boxes_.end(), bboxes_nms.begin(), bboxes_nms.end());
			}
		}
		int num_box = (int)total_boxes_.size();
		vector<FaceInfoX> res_boxes;
		if (num_box != 0) {
			res_boxes = NMS(total_boxes_, 0.7f, 'u');
			BBoxRegression(res_boxes);
			BBoxPadSquare(res_boxes, width, height);
		}
		return res_boxes;
	}

	vector<FaceInfoX> MTCNN::NextStage(const cv::Mat& image, vector<FaceInfoX> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold) 
	{
		vector<FaceInfoX> res;
		int batch_size = (int)pre_stage_res.size();
		if (batch_size == 0)
			return res;
		std::shared_ptr<tensor<float>> input_layer = nullptr;
		std::shared_ptr<tensor<float>> confidence = nullptr;
		std::shared_ptr<tensor<float>> reg_box = nullptr;
		std::shared_ptr<tensor<float>> reg_landmark = nullptr;

		switch (stage_num) {
		case 2: {
			input_layer.reset(new tensor<float>(std::vector<int>{batch_size, 3, input_h, input_w}, device_id_));
		}break;
		case 3: {
			input_layer.reset(new tensor<float>(std::vector<int>{batch_size, 3, input_h, input_w}, device_id_));
		}break;
		default:
			return res;
			break;
		}
		float * input_data = input_layer->mutable_cpu_data();
		int spatial_size = input_h*input_w;

#pragma omp parallel for num_threads(threads_num)
		for (int n = 0; n < batch_size; ++n) {
			FaceBox &box = pre_stage_res[n].bbox;
			cv::Mat roi = image(Rect(Point((int)box.xmin, (int)box.ymin), Point((int)box.xmax, (int)box.ymax))).clone();
			resize(roi, roi, Size(input_w, input_h));
			float *input_data_n = input_data + input_layer->offset(n);
			Vec3b *roi_data = (Vec3b *)roi.data;
			CHECK_EQ(roi.isContinuous(), true);
			for (int k = 0; k < spatial_size; ++k) {
				input_data_n[k] = float((roi_data[k][0] - mean_val)*std_val);
				input_data_n[k + spatial_size] = float((roi_data[k][1] - mean_val)*std_val);
				input_data_n[k + 2 * spatial_size] = float((roi_data[k][2] - mean_val)*std_val);
			}
		}
		switch (stage_num) {
		case 2: {
			RNet_->Forward(input_layer);
			confidence = RNet_->get_prob1();
			reg_box = RNet_->get_conv5_2();
		}break;
		case 3: {
			ONet_->Forward(input_layer);
			confidence = ONet_->get_prob1();
			reg_box = ONet_->get_conv6_2();
			reg_landmark = ONet_->get_conv6_3();
		}break;
		}
		const float* confidence_data = confidence->cpu_data();
		const float* reg_data = reg_box->cpu_data();
		const float* landmark_data = nullptr;
		if (reg_landmark) {
			landmark_data = reg_landmark->cpu_data();
		}
		for (int k = 0; k < batch_size; ++k) {
			if (confidence_data[2 * k + 1] >= threshold) {
				FaceInfoX info;
				info.bbox.score = confidence_data[2 * k + 1];
				info.bbox.xmin = pre_stage_res[k].bbox.xmin;
				info.bbox.ymin = pre_stage_res[k].bbox.ymin;
				info.bbox.xmax = pre_stage_res[k].bbox.xmax;
				info.bbox.ymax = pre_stage_res[k].bbox.ymax;
				for (int i = 0; i < 4; ++i) {
					info.bbox_reg[i] = reg_data[4 * k + i];
				}
				if (reg_landmark) {
					float w = info.bbox.xmax - info.bbox.xmin + 1.f;
					float h = info.bbox.ymax - info.bbox.ymin + 1.f;
					for (int i = 0; i < 5; ++i) {
						info.landmark[2 * i] = landmark_data[10 * k + 2 * i] * w + info.bbox.xmin;
						info.landmark[2 * i + 1] = landmark_data[10 * k + 2 * i + 1] * h + info.bbox.ymin;
					}
				}
				res.push_back(info);
			}
		}
		return res;
	}

	vector<FaceInfoX> MTCNN::Detect(const cv::Mat& image, const int minSize, const float* threshold, const float factor, const int stage) 
	{
		vector<FaceInfoX> pnet_res;
		vector<FaceInfoX> rnet_res;
		vector<FaceInfoX> onet_res;
		if (stage >= 1) 
		{
			pnet_res = ProposalNet(image, minSize, threshold[0], factor);
		}
		if (stage >= 2 && pnet_res.size()>0) {
			if (pnet_max_detect_num < (int)pnet_res.size()) {
				pnet_res.resize(pnet_max_detect_num);
			}
			int num = (int)pnet_res.size();
			int size = (int)ceil(1.f*num / step_size);
			for (int iter = 0; iter < size; ++iter) {
				int start = iter*step_size;
				int end = min(start + step_size, num);
				vector<FaceInfoX> input(pnet_res.begin() + start, pnet_res.begin() + end);
				vector<FaceInfoX> res = NextStage(image, input, 24, 24, 2, threshold[1]);
				rnet_res.insert(rnet_res.end(), res.begin(), res.end());
			}
			rnet_res = NMS(rnet_res, 0.7f, 'u');
			BBoxRegression(rnet_res);
			BBoxPadSquare(rnet_res, image.cols, image.rows);

		}
		if (stage >= 3 && rnet_res.size()>0) {
			int num = (int)rnet_res.size();
			int size = (int)ceil(1.f*num / step_size);
			for (int iter = 0; iter < size; ++iter) {
				int start = iter*step_size;
				int end = min(start + step_size, num);
				vector<FaceInfoX> input(rnet_res.begin() + start, rnet_res.begin() + end);
				vector<FaceInfoX> res = NextStage(image, input, 48, 48, 3, threshold[2]);
				onet_res.insert(onet_res.end(), res.begin(), res.end());
			}
			BBoxRegression(onet_res);
			onet_res = NMS(onet_res, 0.7f, 'm');
			BBoxPad(onet_res, image.cols, image.rows);

		}
		if (stage == 1) {
			return pnet_res;
		}
		else if (stage == 2) {
			return rnet_res;
		}
		else if (stage == 3) {
			return onet_res;
		}
		else {
			return onet_res;
		}
	}

	MTCNN::~MTCNN()
	{
		delete PNet_;
		delete RNet_;
		delete ONet_;
	}
}
