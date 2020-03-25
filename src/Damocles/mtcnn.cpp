#include "mtcnn.hpp"
#include "../Excalibur/tensor_operation_cpu.hpp"
#include "../Excalibur/tensor_operation_gpu.hpp"
#include <algorithm>
#include <fstream>
using namespace std;

namespace glasssix
{
	namespace longinus
	{
		/// <summary>
        /// sort bboxes by score
        /// </summary>
        /// <param name="a">first box</param>
        /// <param name="b">second box</param>
		bool CompareBBox(const FaceInfomation & a, const FaceInfomation & b)
		{
			return a.bbox.score < b.bbox.score;
		}


		MTCNN::MTCNN(int device_id) {
			device_id_ = device_id;
			PNet_ = new mtcnn_pnet(device_id_);
			RNet_ = new mtcnn_rnet(device_id_);
			ONet_ = new mtcnn_onet(device_id_);
		}


		/// <summary>
		/// non-maximum suppression
		/// </summary>
		/// <param name="bboxes">input boxes</param>
		/// <param name="thresh">threshold value decides whether suppress or not</param>
		/// <param name="methodType">'u': intersection/union_box, 'm':intersection/max_box</param>
		std::vector<FaceInfomation> MTCNN::NMS(std::vector<FaceInfomation>& bboxes,
			float thresh, char methodType) {
			std::vector<FaceInfomation> bboxes_nms;
			if (bboxes.size() == 0) {
				return bboxes_nms;
			}
			std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

			float IOU = 0;
			float maxX = 0;
			float maxY = 0;
			float minX = 0;
			float minY = 0;
			std::vector<int> vPick;
			int nPick = 0;
			std::multimap<float, int> vScores;
			const int num_boxes = bboxes.size();
			vPick.resize(num_boxes);
			for (int i = 0; i < num_boxes; ++i) {
				vScores.insert(std::pair<float, int>(bboxes[i].bbox.score, i));
			}

			while (vScores.size() > 0) {
				int last = vScores.rbegin()->second;
				vPick[nPick] = last;
				nPick += 1;
				for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();) {
					int it_idx = it->second;
					maxX = std::max(bboxes.at(it_idx).bbox.xmin, bboxes.at(last).bbox.xmin);
					maxY = std::max(bboxes.at(it_idx).bbox.ymin, bboxes.at(last).bbox.ymin);
					minX = std::min(bboxes.at(it_idx).bbox.xmax, bboxes.at(last).bbox.xmax);
					minY = std::min(bboxes.at(it_idx).bbox.ymax, bboxes.at(last).bbox.ymax);
					//maxX1 and maxY1 reuse 
					maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
					maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
					//IOU reuse for the area of two bbox
					IOU = maxX * maxY;

					float area_it_idx = static_cast<float>((bboxes.at(it_idx).bbox.xmax - bboxes.at(it_idx).bbox.xmin) * (bboxes.at(it_idx).bbox.ymax - bboxes.at(it_idx).bbox.ymin));
					float area_last = static_cast<float>((bboxes.at(last).bbox.xmax - bboxes.at(last).bbox.xmin) * (bboxes.at(last).bbox.ymax - bboxes.at(last).bbox.ymin));
					switch (methodType) {
					case 'u':
						IOU = IOU / (area_it_idx + area_last - IOU);
						break;
					case 'm':
						IOU = IOU / ((area_it_idx < area_last) ? area_it_idx : area_last);
						break;
					default:
						break;
					}

					if (IOU > thresh) {
						it = vScores.erase(it);
					}
					else {
						it++;
					}
				}
			}

			vPick.resize(nPick);
			bboxes_nms.resize(nPick);
			for (int i = 0; i < nPick; i++) {
				bboxes_nms[i] = bboxes[vPick[i]];
			}

			return bboxes_nms;
		}


		/// <summary>
		/// refine bbox coordinates according to bbox_reg
		/// </summary>
		/// <param name="vecFaceInfomation">input boxes</param>
		/// <param name="height">image height</param>
		/// <param name="width">image width</param>
		/// <param name="square">output square boxes</param>
		void MTCNN::refine(std::vector<FaceInfomation> &vecFaceInfomation, const int &height, const int &width, bool square) {
			if (vecFaceInfomation.empty()) {
				//cout << "FaceInfomation is empty!!" << endl;
				return;
			}
			float bbw = 0, bbh = 0, maxSide = 0;
			float h = 0, w = 0;
			float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
			for (vector<FaceInfomation>::iterator it = vecFaceInfomation.begin(); it != vecFaceInfomation.end(); it++) {
				bbw = (*it).bbox.xmax - (*it).bbox.xmin/* + 1*/;
				bbh = (*it).bbox.ymax - (*it).bbox.ymin/* + 1*/;
				x1 = (*it).bbox.xmin + (*it).bbox_reg[0] * bbw;
				y1 = (*it).bbox.ymin + (*it).bbox_reg[1] * bbh;
				x2 = (*it).bbox.xmax + (*it).bbox_reg[2] * bbw;
				y2 = (*it).bbox.ymax + (*it).bbox_reg[3] * bbh;



				if (square) {
					w = x2 - x1/* + 1*/;
					h = y2 - y1/* + 1*/;
					maxSide = (h > w) ? h : w;
					x1 = x1 + w * 0.5 - maxSide * 0.5;
					y1 = y1 + h * 0.5 - maxSide * 0.5;
					(*it).bbox.xmax = round(x1 + maxSide/* - 1*/);
					(*it).bbox.ymax = round(y1 + maxSide/* - 1*/);
					(*it).bbox.xmin = round(x1);
					(*it).bbox.ymin = round(y1);
				}

				//boundary check
				if ((*it).bbox.xmin < 0)(*it).bbox.xmin = 0;
				if ((*it).bbox.ymin < 0)(*it).bbox.ymin = 0;
				if ((*it).bbox.xmax > width)(*it).bbox.xmax = width - 1;
				if ((*it).bbox.ymax > height)(*it).bbox.ymax = height - 1;
			}
		}


		/// <summary>
        /// generate proposal bboxes according to P-NET result
        /// </summary>
        /// <param name="confidence">P-NET confidence</param>
        /// <param name="reg_box">P-NET reg_box</param>
        /// <param name="scale">scale factor</param>
        /// <param name="thresh">confidence exceed threshold value will be proposed</param>
		void MTCNN::GenerateBBox(const std::shared_ptr<tensor<float>> &confidence, const std::shared_ptr<tensor<float>> &reg_box,
			float scale, float thresh)
		{
			int feature_map_w = confidence->width();
			int feature_map_h = confidence->height();
			int spatical_size = feature_map_w * feature_map_h;
			int m2_spatical_size = 2 * spatical_size;
			int m3_spatical_size = 3 * spatical_size;
			const float* confidence_data = confidence->cpu_data() + spatical_size;
			const float* reg_data = reg_box->cpu_data();
			
			candidate_boxes_.clear();
			for (int i = 0; i < spatical_size; i++) {
				if (confidence_data[i] >= thresh) {

					int y = i / feature_map_w;
					int x = i % feature_map_w;
					FaceInfomation FaceInfomation;
					FaceBox &faceBox = FaceInfomation.bbox;

					int x_pnet_stride = x * pnet_stride + 1;
					int y_pnet_stride = y * pnet_stride + 1;
					faceBox.xmin = (int)(x_pnet_stride / scale + 0.5f);
					faceBox.ymin = (int)(y_pnet_stride / scale + 0.5f);
					faceBox.xmax = (int)((x_pnet_stride + pnet_cell_size) / scale + 0.5f);
					faceBox.ymax = (int)((y_pnet_stride + pnet_cell_size) / scale + 0.5f);
					
					FaceInfomation.bbox_reg[0] = reg_data[i];
					FaceInfomation.bbox_reg[1] = reg_data[i + spatical_size];
					FaceInfomation.bbox_reg[2] = reg_data[i + m2_spatical_size];
					FaceInfomation.bbox_reg[3] = reg_data[i + m3_spatical_size];

					faceBox.score = confidence_data[i];
					candidate_boxes_.push_back(FaceInfomation);
				}
			}
		}


		/// <summary>
        /// propose bboxes
        /// </summary>
        /// <param name="image">tensor contains image data</param>
        /// <param name="minSize">minimum size image can be scaled</param>
        /// <param name="threshold">confidence exceed threshold value will be proposed</param>
        /// <param name="factor">scale factor between two near images</param>
		/// <param name="order">order type of image tensor: NCHW(0) / NHWC(1)</param>
		std::vector<FaceInfomation> MTCNN::ProposalNet(const std::shared_ptr<tensor<float>> &image, int minSize, float threshold, float factor, orderType order) 
		{
			int channels = image->channels();
			int height = image->height();
			int width = image->width();
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
				float coef = scales[i];
				int ws = static_cast<int>(width * coef + 0.5);
				int hs = static_cast<int>(height * coef + 0.5);
				input_layer.reset(new tensor<float>(std::vector<int>{ 1, channels, hs, ws }, device_id_));

				if (device_id_ < 0)
				{
					tensor_operation_cpu::resize_cpu(image, input_layer, hs, ws);
				}
				else
				{
#ifdef USE_CUDA
					tensor_operation_gpu::resize_gpu(image, input_layer, hs, ws);
#else
					NO_GPU;
#endif // USE_CUDA
				}

				PNet_->Forward(input_layer);
				std::shared_ptr<tensor<float>> confidence = PNet_->get_prob1();
				std::shared_ptr<tensor<float>> reg = PNet_->get_conv4_2();

				GenerateBBox(confidence, reg, scales[i], threshold);

				if (candidate_boxes_.size() > 0) {
					total_boxes_.insert(total_boxes_.end(), candidate_boxes_.begin(), candidate_boxes_.end());
				}
			}

			return total_boxes_;
		}


		/// <summary>
        /// R-NET and O-NET
        /// </summary>
        /// <param name="image">tensor contains image data</param>
        /// <param name="pre_stage_res">bboxes generate by P-NET or R-NET</param>
        /// <param name="input_w">R-NET: 24, O-NET: 48</param>
        /// <param name="input_h">R-NET: 24, O-NET: 48</param>
        /// <param name="stage_num">R-NET: 2, O-NET: 3</param>
		/// <param name="threshold">confidence exceed threshold value will be retained</param>
		/// <param name="order">order type of image tensor: NCHW(0) / NHWC(1)</param>
		std::vector<FaceInfomation> MTCNN::NextStage(const std::shared_ptr<tensor<float>> &image, std::vector<FaceInfomation> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold, orderType order)
		{
			std::vector<FaceInfomation> res;
			int batch_size = (int)pre_stage_res.size();
			if (batch_size == 0)
				return res;

			int channels = image->channels();
			int height = image->height();
			int width = image->width();

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

			float *input_data = nullptr;
			if (device_id_ < 0)
			{
				input_data = input_layer->mutable_cpu_data();
			}
			else
			{
#ifdef USE_CUDA
				input_data = input_layer->mutable_gpu_data();
#else
				NO_GPU;
#endif
			}

#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int n = 0; n < batch_size; ++n)
			{
				std::shared_ptr<tensor<float>> roi_tensor, roi_resized_tensor;
				
				FaceBox &box = pre_stage_res[n].bbox;
				int rect_h = (int)box.ymax - (int)box.ymin;
				int rect_w = (int)box.xmax - (int)box.xmin;
				glasssix::excalibur::rectangle<int> roi_rect((int)box.xmin, (int)box.ymin, rect_h, rect_w);
				
				if (device_id_ < 0)
				{
					float *input_data_n = input_data + input_layer->offset(n);
					if (rect_h > 0 && rect_w > 0)
					{
						tensor_operation_cpu::safty_cut_cpu(image, roi_tensor, &roi_rect);
						tensor_operation_cpu::resize_cpu(roi_tensor, roi_resized_tensor, input_h, input_w);
						memcpy(input_data_n, roi_resized_tensor->cpu_data(), channels * input_h * input_w * sizeof(float));
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
						tensor_operation_gpu::safty_cut_gpu(image, roi_tensor, &roi_rect);
						tensor_operation_gpu::resize_gpu(roi_tensor, roi_resized_tensor, input_h, input_w);
						CUDA_CHECK(cudaMemcpy(input_data_n, roi_resized_tensor->gpu_data(), channels * input_h * input_w * sizeof(float), cudaMemcpyDefault));
					}
					else
					{
						CUDA_CHECK(cudaMemset(input_data_n, 0, channels * input_h * input_w * sizeof(float)));
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
						float w = info.bbox.xmax - info.bbox.xmin;
						float h = info.bbox.ymax - info.bbox.ymin;
						for (int i = 0; i < 5; ++i) {
							info.landmark[2 * i] = landmark_data[10 * k + i] * w + info.bbox.xmin;
							info.landmark[2 * i + 1] = landmark_data[10 * k + i + 5] * h + info.bbox.ymin;
						}
					}
					res.push_back(info);
				}
			}
			return res;
		}


		/// <summary>
        /// detect humanface in an image
        /// </summary>
        /// <param name="image">image data</param>
        /// <param name="channels">image channel</param>
        /// <param name="height">image height</param>
        /// <param name="width">image width</param>
        /// <param name="minSize">minumum bbox window size</param>
        /// <param name="threshold">threshold values of P-NET/R-NET/O-NET</param>
        /// <param name="factor">scale factor between two near images</param>
		/// <param name="stage">1:P-NET, 2:P-NET/R-NET, 3:P-NET/R-NET/O-NET</param>
		/// <param name="order">order type of image tensor: NCHW(0) / NHWC(1)</param>
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

			std::shared_ptr<tensor<unsigned char>> src_tensor;
			std::shared_ptr<tensor<float>> src_float_tensor;
			float means[3] = { 127.5f, 127.5f, 127.5f };
			float var = 0.0078125;
			if (order == NHWC)
			{
				std::shared_ptr<tensor<unsigned char>> src_nhwc_tensor;
				src_nhwc_tensor.reset(new tensor<unsigned char>(std::vector<int>{1, height, width, channels}, device_id_, NHWC));
				if (device_id_ < 0)
				{
					memcpy(src_nhwc_tensor->mutable_cpu_data(), image, channels * height * width * sizeof(unsigned char));
					tensor_operation_cpu::nhwc2nchw_cpu(src_nhwc_tensor, src_tensor);
					tensor_operation_cpu::preprocess_tensors_cpu(src_tensor, src_float_tensor, means, var);
				}
				else
				{
#ifdef USE_CUDA
					CUDA_CHECK(cudaMemcpy(src_nhwc_tensor->mutable_gpu_data(), image, channels * height * width * sizeof(unsigned char), cudaMemcpyDefault));
					tensor_operation_gpu::nhwc2nchw_gpu(src_nhwc_tensor, src_tensor);
					tensor_operation_gpu::preprocess_tensors_gpu(src_tensor, src_float_tensor, means, var);
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
					tensor_operation_cpu::preprocess_tensors_cpu(src_tensor, src_float_tensor, means, var);
				}
				else
				{
#ifdef USE_CUDA
					CUDA_CHECK(cudaMemcpy(src_tensor->mutable_gpu_data(), image, channels * height * width * sizeof(unsigned char), cudaMemcpyDefault));
					tensor_operation_gpu::preprocess_tensors_gpu(src_tensor, src_float_tensor, means, var);
#else
					NO_GPU;
#endif // USE_CUDA
				}
			}
			else
			{
				NOT_IMPLEMENTED;
			}

			if (stage >= 1)
			{
				pnet_res = ProposalNet(src_float_tensor, minSize, threshold[0], factor, order_);
				pnet_res = NMS(pnet_res, 0.5, 'u');
				refine(pnet_res, height, width, true);
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
					std::vector<FaceInfomation> res = NextStage(src_float_tensor, input, 24, 24, 2, threshold[1], order_);
					rnet_res.insert(rnet_res.end(), res.begin(), res.end());
				}
				rnet_res = NMS(rnet_res, 0.7f, 'u');
				refine(rnet_res, height, width, true);
			}
			
			if (stage >= 3 && rnet_res.size()>0) {
				int num = (int)rnet_res.size();
				int size = (int)ceil(1.f*num / step_size);
				for (int iter = 0; iter < size; ++iter) {
					int start = iter*step_size;
					int end = std::min(start + step_size, num);
					std::vector<FaceInfomation> input(rnet_res.begin() + start, rnet_res.begin() + end);
					std::vector<FaceInfomation> res = NextStage(src_float_tensor, input, 48, 48, 3, threshold[2], order_);
					onet_res.insert(onet_res.end(), res.begin(), res.end());
				}

				onet_res = NMS(onet_res, 0.7f, 'm');
				refine(onet_res, height, width, true);				
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
