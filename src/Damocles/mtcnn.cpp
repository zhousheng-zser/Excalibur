#include "mtcnn.hpp"
#include "../Excalibur/tensor_operation_cpu.hpp"
#include "../Excalibur/tensor_operation_gpu.hpp"
#include <algorithm>


namespace glasssix
{
	namespace longinus
	{
		bool CompareBBox(const FaceInfomation & a, const FaceInfomation & b)
		{
			return a.bbox.score > b.bbox.score;
		}

		MTCNN::MTCNN(int device_id) {
			device_id_ = device_id;
#ifdef _OPENMP
			omp_set_num_threads(threads_num);
#endif
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
				float ov = s / std::min((xmax - xmin + 1)*(ymax - ymin + 1), (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1));
				return ov;
			}
			else {
				float ov = s / ((xmax - xmin + 1)*(ymax - ymin + 1) + (xmax_ - xmin_ + 1)*(ymax_ - ymin_ + 1) - s);
				return ov;
			}
		}

		std::vector<FaceInfomation> MTCNN::NMS(std::vector<FaceInfomation>& bboxes,
			float thresh, char methodType) {
			std::vector<FaceInfomation> bboxes_nms;
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

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_num)
#endif
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

		void MTCNN::BBoxRegression(std::vector<FaceInfomation>& bboxes)
		{
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_num)
#endif // _OPENMP
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

		void MTCNN::BBoxPad(std::vector<FaceInfomation>& bboxes, int width, int height)
		{
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_num)
#endif
			for (int i = 0; i < bboxes.size(); ++i) {
				FaceBox &bbox = bboxes[i].bbox;
				bbox.xmin = round(std::max(bbox.xmin, 0.f));
				bbox.ymin = round(std::max(bbox.ymin, 0.f));
				bbox.xmax = round(std::min(bbox.xmax, width - 1.f));
				bbox.ymax = round(std::min(bbox.ymax, height - 1.f));
			}
		}

		void MTCNN::BBoxPadSquare(std::vector<FaceInfomation>& bboxes, int width, int height)
		{
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_num)
#endif
			for (int i = 0; i < bboxes.size(); ++i) {
				FaceBox &bbox = bboxes[i].bbox;
				float w = bbox.xmax - bbox.xmin + 1;
				float h = bbox.ymax - bbox.ymin + 1;
				float side = h>w ? h : w;
				bbox.xmin = round(std::max(bbox.xmin + (w - side)*0.5f, 0.f));

				bbox.ymin = round(std::max(bbox.ymin + (h - side)*0.5f, 0.f));
				bbox.xmax = round(std::min(bbox.xmin + side - 1, width - 1.f));
				bbox.ymax = round(std::min(bbox.ymin + side - 1, height - 1.f));
			}
		}

		void MTCNN::GenerateBBox(const std::shared_ptr<tensor<float>> &confidence, const std::shared_ptr<tensor<float>> &reg_box,
			float scale, float thresh)
		{
			int feature_map_w = confidence->width();
			int feature_map_h = confidence->height();
			int spatical_size = feature_map_w * feature_map_h;
			const float* confidence_data = confidence->cpu_data() + spatical_size;
			const float* reg_data = reg_box->cpu_data();

			candidate_boxes_.clear();
			for (int i = 0; i<spatical_size; i++) {
				if (confidence_data[i] >= thresh) {

					int y = i / feature_map_w;
					int x = i - feature_map_w * y;
					FaceInfomation FaceInfomation;
					FaceBox &faceBox = FaceInfomation.bbox;

					faceBox.xmin = (float)(x * pnet_stride) / scale;
					faceBox.ymin = (float)(y * pnet_stride) / scale;
					faceBox.xmax = (float)(x * pnet_stride + pnet_cell_size - 1.f) / scale;
					faceBox.ymax = (float)(y * pnet_stride + pnet_cell_size - 1.f) / scale;

					FaceInfomation.bbox_reg[0] = reg_data[i];
					FaceInfomation.bbox_reg[1] = reg_data[i + spatical_size];
					FaceInfomation.bbox_reg[2] = reg_data[i + 2 * spatical_size];
					FaceInfomation.bbox_reg[3] = reg_data[i + 3 * spatical_size];

					faceBox.score = confidence_data[i];
					candidate_boxes_.push_back(FaceInfomation);
				}
			}
		}

		std::vector<FaceInfomation> MTCNN::ProposalNet(const unsigned char* image, const int channels, const int height, const int width, 
			int minSize, float threshold, float factor, orderType order) 
		{
			std::shared_ptr<tensor<unsigned char>> src_tensor, resized_tensor;
			if (order == NHWC)
			{
				std::shared_ptr<tensor<unsigned char>> src_nhwc_tensor;
				src_nhwc_tensor.reset(new tensor<unsigned char>(std::vector<int>{1, height, width, channels}, device_id_, NHWC));
				if (device_id_ < 0)
				{
					memcpy(src_nhwc_tensor->mutable_cpu_data(), image, channels * height * width * sizeof(unsigned char));
					tensor_operation_cpu::nhwc2nchw_cpu(src_nhwc_tensor, src_tensor);
				}
				else
				{
#ifdef USE_CUDA
					cudaMemcpy(src_nhwc_tensor->mutable_gpu_data(), image, channels * height * width * sizeof(unsigned char), cudaMemcpyDefault);
					tensor_operation_gpu::nhwc2nchw_gpu(src_nhwc_tensor, src_tensor);
#else
					NO_GPU;
#endif // USE_CUDA
				}
			}
			else if (order == NCHW)
			{
				src_tensor.reset(new tensor<unsigned char>(std::vector<int>{1, channels, height, width}, device_id_, NCHW));
				if (device_id_ < 0)
				{
					memcpy(src_tensor->mutable_cpu_data(), image, channels * height * width * sizeof(unsigned char));
				}
				else
				{
#ifdef USE_CUDA
					cudaMemcpy(src_tensor->mutable_gpu_data(), image, channels * height * width * sizeof(unsigned char), cudaMemcpyDefault);
#else
					NO_GPU;
#endif // USE_CUDA
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}
			

			float scale = 12.f / minSize;
			float minWH = std::min(height, width) *scale;
			std::vector<float> scales;
			while (minWH >= 12) {
				scales.push_back(scale);
				minWH *= factor;
				scale *= factor;
			}
			total_boxes_.clear();
			std::shared_ptr<tensor<float>> input_layer;
			for (int i = 0; i < scales.size(); i++) {
				int ws = (int)std::ceil(width*scales[i]);
				int hs = (int)std::ceil(height*scales[i]);
				input_layer.reset(new tensor<float>(std::vector<int>{ 1, channels, hs, ws }, device_id_));
				if (device_id_ < 0)
				{
					tensor_operation_cpu::resize_cpu(src_tensor, resized_tensor, hs, ws);
					tensor_operation_cpu::preprocess_tensors_cpu(resized_tensor, input_layer);
				}
				else
				{
#ifdef USE_CUDA
					tensor_operation_gpu::resize_gpu(src_tensor, resized_tensor, hs, ws);
					tensor_operation_gpu::preprocess_tensors_gpu(resized_tensor, input_layer);
#else
					NO_GPU;
#endif // USE_CUDA
				}

				PNet_->Forward(input_layer);
				std::shared_ptr<tensor<float>> confidence = PNet_->get_prob1();
				std::shared_ptr<tensor<float>> reg = PNet_->get_conv4_2();
				GenerateBBox(confidence, reg, scales[i], threshold);
				std::vector<FaceInfomation> bboxes_nms = NMS(candidate_boxes_, 0.5, 'u');
				if (bboxes_nms.size()>0) {
					total_boxes_.insert(total_boxes_.end(), bboxes_nms.begin(), bboxes_nms.end());
				}
			}
			int num_box = (int)total_boxes_.size();
			std::vector<FaceInfomation> res_boxes;
			if (num_box != 0) {
				res_boxes = NMS(total_boxes_, 0.7f, 'u');
				BBoxRegression(res_boxes);
				BBoxPadSquare(res_boxes, width, height);
			}
			return res_boxes;
		}

		std::vector<FaceInfomation> MTCNN::NextStage(const unsigned char* image, const int channels, const int height, const int width, 
			std::vector<FaceInfomation> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold, orderType order)
		{
			std::vector<FaceInfomation> res;
			int batch_size = (int)pre_stage_res.size();
			if (batch_size == 0)
				return res;

			std::shared_ptr<tensor<unsigned char>> src_tensor;
			if (order == NHWC)
			{
				std::shared_ptr<tensor<unsigned char>> src_nhwc_tensor;
				src_nhwc_tensor.reset(new tensor<unsigned char>(std::vector<int>{1, height, width, channels}, device_id_, NHWC));
				if (device_id_ < 0)
				{
					memcpy(src_nhwc_tensor->mutable_cpu_data(), image, channels * height * width * sizeof(unsigned char));
					tensor_operation_cpu::nhwc2nchw_cpu(src_nhwc_tensor, src_tensor);
				}
				else
				{
#ifdef USE_CUDA
					cudaMemcpy(src_nhwc_tensor->mutable_gpu_data(), image, channels * height * width * sizeof(unsigned char), cudaMemcpyDefault);
					tensor_operation_gpu::nhwc2nchw_gpu(src_nhwc_tensor, src_tensor);
#else
					NO_GPU;
#endif // USE_CUDA
				}
			}
			else if (order == NCHW)
			{
				src_tensor.reset(new tensor<unsigned char>(std::vector<int>{1, channels, height, width}, device_id_, NCHW));
				if (device_id_ < 0)
				{
					memcpy(src_tensor->mutable_cpu_data(), image, channels * height * width * sizeof(unsigned char));
				}
				else
				{
#ifdef USE_CUDA
					cudaMemcpy(src_tensor->mutable_gpu_data(), image, channels * height * width * sizeof(unsigned char), cudaMemcpyDefault);
#else
					NO_GPU;
#endif // USE_CUDA
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			std::shared_ptr<tensor<float>> input_layer;
			std::shared_ptr<tensor<float>> confidence;
			std::shared_ptr<tensor<float>> reg_box;
			std::shared_ptr<tensor<float>> reg_landmark;

			switch (stage_num) {
			case 2: {
				input_layer.reset(new tensor<float>(std::vector<int>{batch_size, channels, input_h, input_w}, device_id_));
			}break;
			case 3: {
				input_layer.reset(new tensor<float>(std::vector<int>{batch_size, channels, input_h, input_w}, device_id_));
			}break;
			default:
				return res;
				break;
			}

			float *input_data;
			if (device_id_ < 0)
			{
				input_data = input_layer->mutable_cpu_data();
			}
			else
			{
				input_data = input_layer->mutable_gpu_data();
			}

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_num)
#endif
			for (int n = 0; n < batch_size; ++n)
			{
				std::shared_ptr<tensor<unsigned char>> roi_tensor, roi_resized_tensor;
				std::shared_ptr<tensor<float>> roi_resized_float_tensor;
				roi_resized_float_tensor.reset(new tensor<float>(std::vector<int>{1, channels, input_h, input_w}, device_id_));

				FaceBox &box = pre_stage_res[n].bbox;
				int rect_h = (int)box.ymax - (int)box.ymin + 1;
				int rect_w = (int)box.xmax - (int)box.xmin + 1;
				glasssix::excalibur::rectangle<int> roi_rect((int)box.xmin, (int)box.ymin, rect_h, rect_w);
				
				if (device_id_ < 0)
				{
					float *input_data_n = input_data + input_layer->offset(n);
					if (rect_h > 0 && rect_w > 0)
					{
						tensor_operation_cpu::safty_cut_cpu(src_tensor, roi_tensor, &roi_rect);
						tensor_operation_cpu::resize_cpu(roi_tensor, roi_resized_tensor, input_h, input_w);
						tensor_operation_cpu::preprocess_tensors_cpu(roi_resized_tensor, roi_resized_float_tensor);
						memcpy(input_data_n, roi_resized_float_tensor->cpu_data(), channels * input_h * input_w * sizeof(float));
					}
					else
					{
						memset(input_data_n, 0, channels * input_h * input_w * sizeof(float));
					}
					
				}
				else
				{
#ifdef USE_CUDA
					float *input_data_n = input_data + input_layer->offset(n);
					if (rect_h > 0 && rect_w > 0)
					{
						tensor_operation_gpu::safty_cut_gpu(src_tensor, roi_tensor, &roi_rect);
						tensor_operation_gpu::resize_gpu(roi_tensor, roi_resized_tensor, input_h, input_w);
						tensor_operation_gpu::preprocess_tensors_gpu(roi_resized_tensor, roi_resized_float_tensor);
						cudaMemcpy(input_data_n, roi_resized_float_tensor->gpu_data(), channels * input_h * input_w * sizeof(float), cudaMemcpyDefault);
					}
					else
					{
						cudaMemset(input_data_n, 0, channels * input_h * input_w * sizeof(float));
					}
#else
					NO_GPU;
#endif // USE_CUDA
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
					FaceInfomation info;
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

		std::vector<FaceInfomation> MTCNN::Detect(const unsigned char* image, const int channels, const int height, const int width, 
			const int minSize, const float* threshold, const float factor, const int stage, int order)
		{
			std::vector<FaceInfomation> pnet_res;
			std::vector<FaceInfomation> rnet_res;
			std::vector<FaceInfomation> onet_res; 
			if (order != 0 && order != 1)
			{
				return onet_res;
			}
			orderType order_ = (orderType)order;
			if (stage >= 1)
			{
				pnet_res = ProposalNet(image, channels, height, width, minSize, threshold[0], factor, order_);
			}

			if (stage >= 2 && pnet_res.size()>0) {
				if (pnet_max_detect_num < (int)pnet_res.size()) {
					pnet_res.resize(pnet_max_detect_num);
				}
				int num = (int)pnet_res.size();
				int size = (int)ceil(1.f*num / step_size);
				for (int iter = 0; iter < size; ++iter) {
					int start = iter*step_size;
					int end = std::min(start + step_size, num);
					std::vector<FaceInfomation> input(pnet_res.begin() + start, pnet_res.begin() + end);
					std::vector<FaceInfomation> res = NextStage(image, channels, height, width, input, 24, 24, 2, threshold[1], order_);
					rnet_res.insert(rnet_res.end(), res.begin(), res.end());
				}
				rnet_res = NMS(rnet_res, 0.7f, 'u');
				BBoxRegression(rnet_res);
				BBoxPadSquare(rnet_res, width, height);
			}
			
			if (stage >= 3 && rnet_res.size()>0) {
				int num = (int)rnet_res.size();
				int size = (int)ceil(1.f*num / step_size);
				for (int iter = 0; iter < size; ++iter) {
					int start = iter*step_size;
					int end = std::min(start + step_size, num);
					std::vector<FaceInfomation> input(rnet_res.begin() + start, rnet_res.begin() + end);
					std::vector<FaceInfomation> res = NextStage(image, channels, height, width, input, 48, 48, 3, threshold[2], order_);
					onet_res.insert(onet_res.end(), res.begin(), res.end());
				}
				BBoxRegression(onet_res);
				onet_res = NMS(onet_res, 0.7f, 'm');
				BBoxPad(onet_res, width, height);
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
}
