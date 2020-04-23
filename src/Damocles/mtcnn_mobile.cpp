#include "mtcnn_mobile.hpp"
#include "Excalibur/tensor_operation_cpu.hpp"

#ifdef USE_CUDA
#include "Excalibur/tensor_operation_gpu.hpp"
#endif

#include <algorithm>

using namespace glasssix::memory;
using namespace glasssix::longinus;

glasssix::longinus::mtcnn_mobile::mtcnn_mobile(int device_id, bool handle_big_face) : vDamocles(device_id)
{
	device_id_ = device_id;
#ifdef _OPENMP
	omp_set_num_threads(threads_num);
#endif
	PNet_ = new pnet_mobile(device_id_);
	RNet_ = new rnet_mobile(device_id_);
	ONet_ = new onet_mobile(device_id_);

	handle_big_face_ = handle_big_face;
}


glasssix::longinus::mtcnn_mobile::~mtcnn_mobile()
{
	delete PNet_;
	delete RNet_;
	delete ONet_;
}


/// <summary>
/// detect humanface in an image
/// </summary>
/// <param name="img">image data</param>
/// <param name="channels">image channel</param>
/// <param name="height">image height</param>
/// <param name="width">image width</param>
/// <param name="minSize">minumum bbox window size</param>
/// <param name="threshold">threshold values of P-NET/R-NET/O-NET</param>
/// <param name="factor">scale factor between two near images</param>
/// <param name="stage">1:P-NET, 2:P-NET/R-NET, 3:P-NET/R-NET/O-NET</param>
/// <param name="order">order type of image tensor: NCHW(0) / NHWC(1)</param>
std::vector<FaceInfomation> glasssix::longinus::mtcnn_mobile::Detect(const unsigned char* img, const int channels, const int height, const int width,
	const int minSize, const float* threshold, const float factor, const int stage, int order)
{
	std::shared_ptr<tensor<unsigned char> > src_tensor;
	if (order == NHWC)
	{
		std::shared_ptr<tensor<unsigned char> > src_nhwc_tensor;
		src_nhwc_tensor = std::shared_ptr<tensor<unsigned char> > (new tensor<unsigned char>(std::vector<int>{1, height, width, channels}, device_id_, NHWC));
		src_nhwc_tensor->copy_from(img, channels * height * width);
		if (device_id_ < 0)
		{
			tensor_operation_cpu::nhwc2nchw_cpu(src_nhwc_tensor, src_tensor);
		}
		else
		{
#ifdef USE_CUDA
			tensor_operation_gpu::nhwc2nchw_gpu(src_nhwc_tensor, src_tensor);
#else
			NO_GPU;
#endif
		}
	}
	else if (order == NCHW)
	{
		src_tensor = std::shared_ptr<tensor<unsigned char> >(new tensor <unsigned char>(std::vector<int>{1, channels, height, width}, device_id_, NCHW));
		src_tensor->copy_from(img, channels * height * width);
	}
	else
	{
		NOT_IMPLEMENTED;
	}

	const int pnet_overlap_thresh_count = 3;
	float nms_thresh_per_scale = 0.45;
	float nms_thresh[3] = { 0.4, 0.5, 0.5 };
	float minWH = std::min(height, width);

	int minSize_ = std::max(pnet_size, minSize);
	float m = (float)pnet_size / minSize_;
	minWH *= m;
	std::vector<float> scales;
	while (minWH > pnet_size)
	{
		scales.push_back(m);
		minWH *= factor;
		m *= factor;
	}
	minWH = std::min(width, height);

	int count = scales.size();

	for (int i = count - 1; i >= 0; i--)
	{
		if (std::ceil(scales[i] * minWH) <= pnet_size)
			count--;
	}

	if (handle_big_face_)
	{
		if (count > 2)
			count--;

		scales.resize(count);
		if (count > 0)
		{
			float last_size = ceil(scales[count - 1] * minWH);
			for (int tmp_size = last_size - 1; tmp_size >= pnet_size + 1; tmp_size -= 2)
			{
				scales.push_back(tmp_size * 1.0 / minWH);
				count++;
			}
		}

		scales.push_back(pnet_size * 1.0 / minWH);
		count++;
	}
	else
	{
		scales.push_back(pnet_size * 1.0 / minWH);
		count++;
	}

	std::vector<Longinus_CNN_BBox> result_bbox;
	
	std::vector<Longinus_CNN_BBox> pnet_bbox;

	if(stage >= 1)
		PNet_Process(src_tensor, threshold[0], nms_thresh[0], scales, pnet_bbox);
	
	std::vector<Longinus_CNN_BBox> rnet_bbox;
	if(stage >= 2)
		RNet_Process(pnet_bbox, rnet_bbox, src_tensor, minSize_, threshold[1], nms_thresh[1]);

	std::vector<Longinus_CNN_BBox> onet_bbox;
	if(stage >= 3)
		ONet_Process(rnet_bbox, onet_bbox, src_tensor, minSize_, threshold[2], nms_thresh[2], true);
	
	if(stage >= 3)
		result_bbox = onet_bbox;
	else if(stage >= 2)
		result_bbox = rnet_bbox;
	else if(stage >= 1)
		result_bbox = pnet_bbox;

	std::vector<FaceInfomation> faceInfo;
	for (int i = 0; i < result_bbox.size(); i++)
	{
		FaceInfomation f;
		f.bbox.score = result_bbox[i].score;
		f.bbox.xmin = result_bbox[i].col1;
		f.bbox.ymin = result_bbox[i].row1;
		f.bbox.xmax = result_bbox[i].col2;
		f.bbox.ymax = result_bbox[i].row2;

		std::memcpy(f.landmark, result_bbox[i].ppoint, 10 * sizeof(float));
		std::memcpy(f.headpose, result_bbox[i].headpose, 3 * sizeof(float));
		std::memcpy(f.bbox_reg, result_bbox[i].regreCoord, 4 * sizeof(float));
		
		faceInfo.push_back(f);
	}

	return faceInfo;
}


/// <summary>
/// propose bboxes
/// </summary>
/// <param name="bgr_8uc3">tensor contains image data</param>
/// <param name="thresh">score threshhold value</param>
/// <param name="nms_thresh">nms threshold value</param>
/// <param name="scales">scale factors</param>
/// <param name="pnet_result">proposed bboxes</param>
bool glasssix::longinus::mtcnn_mobile::PNet_Process(std::shared_ptr<tensor<unsigned char> > &bgr_8uc3, float thresh, float nms_thresh, std::vector<float> scales, std::vector<Longinus_CNN_BBox>& pnet_result)
{
	pnet_result.clear();
	std::vector<std::vector<float> > maps;
	std::vector<int> mapH;
	std::vector<int> mapW;

	int width = bgr_8uc3->width();
	int height = bgr_8uc3->height();

	int scale_num = 0;
	std::vector<int> changedH, changedW;
	for (int i = 0; i < scales.size(); i++)
	{
		int changedh = (int)ceil(height*scales[i]);
		int changedw = (int)ceil(width*scales[i]);
		if (changedh < pnet_size || changedw < pnet_size)
			continue;
		scale_num++;
		changedH.push_back(changedh);
		changedW.push_back(changedw);
		mapH.push_back((changedh - pnet_size) / pnet_stride + 1);
		mapW.push_back((changedw - pnet_size) / pnet_stride + 1);
	}

	maps.resize(scale_num);
	for (int i = 0; i < scale_num; i++)
	{
		maps[i].resize(mapH[i] * mapW[i]);
	}

	float means[3] = { 104.0f, 117.0f, 124.0f };
	float var = 0.0078125;
	for (int i = 0; i < scale_num; i++)
	{
		int changedh = changedH[i];
		int changedw = changedW[i];
		float cur_scale_x = (float)width / changedw;
		float cur_scale_y = (float)height / changedh;

		std::shared_ptr<tensor<unsigned char> > resized;
		std::shared_ptr<tensor<float> > bgr_32fc3 = std::shared_ptr<tensor<float> >(new tensor<float>(bgr_8uc3->data_shape(), device_id_, bgr_8uc3->order()));
		if (device_id_ < 0)
		{
			tensor_operation_cpu::resize_cpu(bgr_8uc3, resized, changedh, changedw);
			tensor_operation_cpu::preprocess_tensors_cpu(resized, bgr_32fc3, means, var);
		}
		else
		{
#ifdef USE_CUDA
			tensor_operation_gpu::resize_gpu(bgr_8uc3, resized, changedh, changedw);
			tensor_operation_gpu::preprocess_tensors_gpu(resized, bgr_32fc3, means, var);
#else
			NO_GPU;
#endif
		}

		PNet_->Forward(bgr_32fc3);
		std::shared_ptr<memory::tensor<float>> confidence = PNet_->get_cls_prob();
		const float *confidence_data = confidence->cpu_data();

		int confidenceH = confidence->height();
		int confidenceW = confidence->width();

		int result_num = confidenceH * confidenceW;

		confidence_data += result_num;

		for (int row = 0; row < confidenceH; row++)
		{
			for (int col = 0; col < confidenceW; col++)
			{
				if (row < mapH[i] && col < mapW[i])
				{
					maps[i][row*mapW[i] + col] = *confidence_data;
				}
				confidence_data++;
			}
		}
	}

	return GenerateBoundingBox(pnet_result, maps, scales, mapH, mapW, thresh, nms_thresh, pnet_stride, pnet_size, width, height);
}


/// <summary>
/// sort bboxes by score
/// </summary>
/// <param name="lsh">first box</param>
/// <param name="rsh">second box</param>
static bool _cmp_score(const Longinus_CNN_OrderScore& lsh, const Longinus_CNN_OrderScore& rsh)
{
	return lsh.score < rsh.score;
}


/// <summary>
/// non-maximum suppression
/// </summary>
/// <param name="boundingBox">input boxes</param>
/// <param name="bboxScore">input scores</param>
/// <param name="overlap_threshold">threshold value decides whether suppress or not</param>
/// <param name="modelname">'Union': intersection/union_box, 'Min':intersection/min_box</param>
/// <param name="overlap_count_thresh">overlap_count_thresh</param>
/// <param name="thread_num">number of thread on-using</param>
static void _nms(std::vector<Longinus_CNN_BBox> &boundingBox, std::vector<Longinus_CNN_OrderScore> &bboxScore, const float overlap_threshold,
	const std::string& modelname = "Union", int overlap_count_thresh = 0, int thread_num = 1)
{
	if (boundingBox.empty() || overlap_threshold >= 1.0)
	{
		return;
	}
	std::vector<int> heros;
	std::vector<int> overlap_num;
	//sort the score
	sort(bboxScore.begin(), bboxScore.end(), _cmp_score);

	int order = 0;
	float IOU = 0;
	float maxX = 0;
	float maxY = 0;
	float minX = 0;
	float minY = 0;
	while (bboxScore.size() > 0)
	{
		order = bboxScore.back().oriOrder;
		bboxScore.pop_back();
		if (order < 0)continue;
		heros.push_back(order);
		int cur_overlap = 0;
		boundingBox[order].exist = false;//delete it
		int box_num = boundingBox.size();
		if (thread_num == 1)
		{
			for (int num = 0; num < box_num; num++)
			{
				if (boundingBox[num].exist)
				{
					//the iou
					maxY = std::max(boundingBox[num].row1, boundingBox[order].row1);
					maxX = std::max(boundingBox[num].col1, boundingBox[order].col1);
					minY = std::min(boundingBox[num].row2, boundingBox[order].row2);
					minX = std::min(boundingBox[num].col2, boundingBox[order].col2);
					//maxX1 and maxY1 reuse 
					maxX = std::max(minX - maxX + 1, 0.f);
					maxY = std::max(minY - maxY + 1, 0.f);
					//IOU reuse for the area of two bbox
					IOU = maxX * maxY;
					float area1 = boundingBox[num].area;
					float area2 = boundingBox[order].area;
					if (!modelname.compare("Union"))
						IOU = IOU / (area1 + area2 - IOU);
					else if (!modelname.compare("Min"))
					{
						IOU = IOU / std::min(area1, area2);
					}
					if (IOU > overlap_threshold)
					{
						cur_overlap++;
						boundingBox[num].exist = false;
						for (std::vector<Longinus_CNN_OrderScore>::iterator it = bboxScore.begin(); it != bboxScore.end(); it++)
						{
							if ((*it).oriOrder == num)
							{
								(*it).oriOrder = -1;
								break;
							}
						}
					}
				}
			}
		}
		else
		{
			int chunk_size = ceil(box_num / thread_num);
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_num)
			for (int num = 0; num < box_num; num++)
			{
				if (boundingBox.at(num).exist)
				{
					//the iou
					maxY = std::max(boundingBox[num].row1, boundingBox[order].row1);
					maxX = std::max(boundingBox[num].col1, boundingBox[order].col1);
					minY = std::min(boundingBox[num].row2, boundingBox[order].row2);
					minX = std::min(boundingBox[num].col2, boundingBox[order].col2);
					//maxX1 and maxY1 reuse 
					maxX = std::max(minX - maxX + 1, 0.f);
					maxY = std::max(minY - maxY + 1, 0.f);
					//IOU reuse for the area of two bbox
					IOU = maxX * maxY;
					float area1 = boundingBox[num].area;
					float area2 = boundingBox[order].area;
					if (!modelname.compare("Union"))
						IOU = IOU / (area1 + area2 - IOU);
					else if (!modelname.compare("Min"))
					{
						IOU = IOU / std::min(area1, area2);
					}
					if (IOU > overlap_threshold)
					{
						cur_overlap++;
						boundingBox.at(num).exist = false;
						for (std::vector<Longinus_CNN_OrderScore>::iterator it = bboxScore.begin(); it != bboxScore.end(); it++)
						{
							if ((*it).oriOrder == num)
							{
								(*it).oriOrder = -1;
								break;
							}
						}
					}
				}
			}
		}
		overlap_num.push_back(cur_overlap);
	}
	for (int i = 0; i < heros.size(); i++)
	{
		if (!boundingBox[heros[i]].need_check_overlap_count
			|| overlap_num[i] >= overlap_count_thresh)
			boundingBox[heros[i]].exist = true;
	}
	//clear exist= false;
	for (int i = boundingBox.size() - 1; i >= 0; i--)
	{
		if (!boundingBox[i].exist)
		{
			boundingBox.erase(boundingBox.begin() + i);
		}
	}
}


/// <summary>
/// refine bbox coordinates according to regreCoord
/// </summary>
/// <param name="vecBbox">input and output boxes</param>
/// <param name="net">'p':P-NET, 'r':R-NET, 'o':O-NET</param>
/// <param name="square">output square boxes</param>
static void _refine_and_square_bbox(std::vector<Longinus_CNN_BBox> &vecBbox, char net, bool square)
{
	float bbw = 0, bbh = 0, bboxSize = 0;
	float h = 0, w = 0;
	float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
	for (std::vector<Longinus_CNN_BBox>::iterator it = vecBbox.begin(); it != vecBbox.end(); it++)
	{
		if ((*it).exist)
		{
			bbh = (*it).row2 - (*it).row1 + 1;
			bbw = (*it).col2 - (*it).col1 + 1;
			y1 = (*it).row1 + (*it).regreCoord[1] * bbh;
			x1 = (*it).col1 + (*it).regreCoord[0] * bbw;
			if (net == 'p' || net == 'r')
			{
				y2 = (*it).row2 + (*it).regreCoord[3] * bbh;
				x2 = (*it).col2 + (*it).regreCoord[2] * bbw;
			}
			else if (net == 'o')
			{
				y2 = (*it).row1 + (*it).regreCoord[3] * bbh;
				x2 = (*it).col1 + (*it).regreCoord[2] * bbw;
			}


			w = x2 - x1 + 1;
			h = y2 - y1 + 1;
			if (square)
			{
				bboxSize = (h > w) ? h : w;
				y1 = round(y1 + h * 0.5 - bboxSize * 0.5);
				x1 = round(x1 + w * 0.5 - bboxSize * 0.5);
				(*it).row2 = round(y1 + bboxSize - 1);
				(*it).col2 = round(x1 + bboxSize - 1);
				(*it).row1 = round(y1);
				(*it).col1 = round(x1);
			}
			else
			{
				(*it).row2 = round(y1 + h - 1);
				(*it).col2 = round(x1 + w - 1);
				(*it).row1 = round(y1);
				(*it).col1 = round(x1);
			}

			//boundary check
			/*if ((*it).row1 < 0)(*it).row1 = 0;
			if ((*it).col1 < 0)(*it).col1 = 0;
			if ((*it).row2 > height)(*it).row2 = height - 1;
			if ((*it).col2 > width)(*it).col2 = width - 1;*/

			it->area = (it->row2 - it->row1)*(it->col2 - it->col1);
		}
	}
}


/// <summary>
/// generate proposal bboxes according to P-NET result
/// </summary>
/// <param name="bounding_bbox">proposed boxes</param>
/// <param name="maps">confidence score</param>
/// <param name="scales">scale factors</param>
/// <param name="mapH">scaled image height</param>
/// <param name="mapW">scaled image width</param>
/// <param name="thresh">score threshold</param>
/// <param name="nms_thresh">nms threshold</param>
/// <param name="stride">4</param>
/// <param name="cellSize">20</param>
/// <param name="image_width">original image width</param>
/// <param name="image_height">original image height</param>
bool glasssix::longinus::mtcnn_mobile::GenerateBoundingBox(std::vector<Longinus_CNN_BBox>& bounding_bbox, std::vector<std::vector<float>>& maps, std::vector<float>& scales, 
	std::vector<int>& mapH, std::vector<int>& mapW, float thresh, float nms_thresh, int stride, int cellSize, int image_width, int image_height)
{
	Longinus_CNN_OrderScore order;
	std::vector<std::vector<Longinus_CNN_BBox>> bounding_boxes(scales.size());
	std::vector<std::vector<Longinus_CNN_OrderScore>> bounding_scores(scales.size());
	const int block_size = 32;
	int border_size = cellSize / stride;
	float nms_thresh_per_scale = 0.45;
	int pnet_overlap_thresh_count = 3;

	for (int i = 0; i < maps.size(); i++)
	{
		int changedH = (int)ceil(image_height*scales[i]);
		int changedW = (int)ceil(image_width*scales[i]);
		if (changedH < cellSize || changedW < cellSize)
			continue;
		float cur_scale_x = (float)image_width / changedW;
		float cur_scale_y = (float)image_height / changedH;

		int count = 0;
		int scoreH = mapH[i];
		int scoreW = mapW[i];
		const float *p = &maps[i][0];
		if (scoreW <= block_size && scoreH < block_size)
		{

			Longinus_CNN_BBox bbox;
			Longinus_CNN_OrderScore order;
			for (int row = 0; row < scoreH; row++)
			{
				for (int col = 0; col < scoreW; col++)
				{
					if (*p > thresh)
					{
						bbox.score = *p;
						order.score = *p;
						order.oriOrder = count;
						bbox.row1 = stride * row;
						bbox.col1 = stride * col;
						bbox.row2 = stride * row + cellSize;
						bbox.col2 = stride * col + cellSize;
						bbox.exist = true;
						bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
						bbox.need_check_overlap_count = (row >= border_size && row < scoreH - border_size)
							&& (col >= border_size && col < scoreW - border_size);
						bounding_boxes[i].push_back(bbox);
						bounding_scores[i].push_back(order);
						count++;
					}
					p++;
				}
			}
			int before_count = bounding_boxes[i].size();
			_nms(bounding_boxes[i], bounding_scores[i], nms_thresh_per_scale, "Union", pnet_overlap_thresh_count);
			int after_count = bounding_boxes[i].size();
			for (int j = 0; j < after_count; j++)
			{
				Longinus_CNN_BBox& bbox = bounding_boxes[i][j];
				bbox.row1 = round(bbox.row1 *cur_scale_y);
				bbox.col1 = round(bbox.col1 *cur_scale_x);
				bbox.row2 = round(bbox.row2 *cur_scale_y);
				bbox.col2 = round(bbox.col2 *cur_scale_x);
				bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
			}
		}
		else
		{
			int before_count = 0, after_count = 0;
			int block_H_num = std::max(1, scoreH / block_size);
			int block_W_num = std::max(1, scoreW / block_size);
			int block_num = block_H_num * block_W_num;
			int width_per_block = scoreW / block_W_num;
			int height_per_block = scoreH / block_H_num;
			std::vector<std::vector<Longinus_CNN_BBox> > tmp_bounding_boxes(block_num);
			std::vector<std::vector<Longinus_CNN_OrderScore> > tmp_bounding_scores(block_num);
			std::vector<int> block_start_w(block_num), block_end_w(block_num);
			std::vector<int> block_start_h(block_num), block_end_h(block_num);
			for (int bh = 0; bh < block_H_num; bh++)
			{
				for (int bw = 0; bw < block_W_num; bw++)
				{
					int bb = bh * block_W_num + bw;
					block_start_w[bb] = (bw == 0) ? 0 : (bw*width_per_block - border_size);
					block_end_w[bb] = (bw == block_num - 1) ? scoreW : ((bw + 1)*width_per_block);
					block_start_h[bb] = (bh == 0) ? 0 : (bh*height_per_block - border_size);
					block_end_h[bb] = (bh == block_num - 1) ? scoreH : ((bh + 1)*height_per_block);
				}
			}

			for (int bb = 0; bb < block_num; bb++)
			{
				Longinus_CNN_BBox bbox;
				Longinus_CNN_OrderScore order;
				int count = 0;
				for (int row = block_start_h[bb]; row < block_end_h[bb]; row++)
				{
					p = &maps[i][0] + row * scoreW + block_start_w[bb];
					for (int col = block_start_w[bb]; col < block_end_w[bb]; col++)
					{
						if (*p > thresh)
						{
							bbox.score = *p;
							order.score = *p;
							order.oriOrder = count;
							bbox.row1 = stride * row;
							bbox.col1 = stride * col;
							bbox.row2 = stride * row + cellSize;
							bbox.col2 = stride * col + cellSize;
							bbox.exist = true;
							bbox.need_check_overlap_count = (row >= border_size && row < scoreH - border_size)
								&& (col >= border_size && col < scoreW - border_size);
							bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
							tmp_bounding_boxes[bb].push_back(bbox);
							tmp_bounding_scores[bb].push_back(order);
							count++;
						}
						p++;
					}
				}
				int tmp_before_count = tmp_bounding_boxes[bb].size();
				_nms(tmp_bounding_boxes[bb], tmp_bounding_scores[bb], nms_thresh_per_scale, "Union", pnet_overlap_thresh_count);
				int tmp_after_count = tmp_bounding_boxes[bb].size();
				before_count += tmp_before_count;
				after_count += tmp_after_count;
			}

			count = 0;
			for (int bb = 0; bb < block_num; bb++)
			{
				std::vector<Longinus_CNN_BBox>::iterator it = tmp_bounding_boxes[bb].begin();
				for (; it != tmp_bounding_boxes[bb].end(); it++)
				{
					if ((*it).exist)
					{
						bounding_boxes[i].push_back(*it);
						order.score = (*it).score;
						order.oriOrder = count;
						bounding_scores[i].push_back(order);
						count++;
					}
				}
			}

			//ZQ_CNN_BBoxUtils::_nms(bounding_boxes[i], bounding_scores[i], nms_thresh_per_scale, "Union", 0);
			after_count = bounding_boxes[i].size();
			for (int j = 0; j < after_count; j++)
			{
				Longinus_CNN_BBox& bbox = bounding_boxes[i][j];
				bbox.row1 = round(bbox.row1 *cur_scale_y);
				bbox.col1 = round(bbox.col1 *cur_scale_x);
				bbox.row2 = round(bbox.row2 *cur_scale_y);
				bbox.col2 = round(bbox.col2 *cur_scale_x);
				bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
			}
		}
	}

	std::vector<Longinus_CNN_OrderScore> firstOrderScore;
	int count = 0;
	for (int i = 0; i < scales.size(); i++)
	{
		std::vector<Longinus_CNN_BBox>::iterator it = bounding_boxes[i].begin();
		for (; it != bounding_boxes[i].end(); it++)
		{
			if ((*it).exist)
			{
				bounding_bbox.push_back(*it);
				order.score = (*it).score;
				order.oriOrder = count;
				firstOrderScore.push_back(order);
				count++;
			}
		}
	}

	//the first stage's nms
	if (count < 1) return false;
	_nms(bounding_bbox, firstOrderScore, nms_thresh, "Union", 0, 1);
	_refine_and_square_bbox(bounding_bbox, 'p', true);

	return true;
}


/// <summary>
/// R-NET
/// </summary>
/// <param name="pnetBbox">input proposed boxes</param>
/// <param name="rnet_result">output refined boxes</param>
/// <param name="bgr_8uc3">image data</param>
/// <param name="min_size">minimum box window</param>
/// <param name="thresh">score threshold</param>
/// <param name="nms_thresh">nms threshold</param>
bool glasssix::longinus::mtcnn_mobile::RNet_Process(std::vector<Longinus_CNN_BBox>& pnetBbox, std::vector<Longinus_CNN_BBox>& rnet_result, 
	std::shared_ptr<tensor<unsigned char>>& bgr_8uc3, int min_size, float thresh, float nms_thresh)
{
	if (pnetBbox.size() == 0)
	{
		return false;
	}

	rnet_result.clear();
	std::vector<Longinus_CNN_BBox>::iterator it = pnetBbox.begin();
	std::vector<int> src_off_x, src_off_y, src_rect_w, src_rect_h;
	int r_count = 0;
	for (; it != pnetBbox.end(); it++)
	{
		if ((*it).exist)
		{
			int off_x = it->col1;
			int off_y = it->row1;
			int rect_w = it->col2 - off_x;
			int rect_h = it->row2 - off_y;
			if (rect_w <= 0.5*min_size || rect_h <= 0.5*min_size)
			{
				(*it).exist = false;
				continue;
			}
			else
			{
				src_off_x.push_back(off_x);
				src_off_y.push_back(off_y);
				src_rect_w.push_back(rect_w);
				src_rect_h.push_back(rect_h);
				r_count++;
				rnet_result.push_back(*it);
			}
		}
	}

	int batch_size = 64;
	int per_num = r_count;
	int need_thread_num = 1;
	if (per_num > batch_size)
	{
		need_thread_num = ceil((float)r_count / batch_size);
		per_num = batch_size;
	}
	
	std::vector<std::vector<int> > task_src_off_x(need_thread_num);
	std::vector<std::vector<int> > task_src_off_y(need_thread_num);
	std::vector<std::vector<int> > task_src_rect_w(need_thread_num);
	std::vector<std::vector<int> > task_src_rect_h(need_thread_num);
	std::vector<std::vector<Longinus_CNN_BBox> > task_secondBbox(need_thread_num);

	float means[3] = { 104.0f, 117.0f, 124.0f };
	float var = 0.0078125;
	for (int i = 0; i < need_thread_num; i++)
	{
		int st_id = per_num * i;
		int end_id = std::min(r_count, per_num*(i + 1));
		int cur_num = end_id - st_id;
		if (cur_num > 0)
		{
			task_src_off_x[i].resize(cur_num);
			task_src_off_y[i].resize(cur_num);
			task_src_rect_w[i].resize(cur_num);
			task_src_rect_h[i].resize(cur_num);
			task_secondBbox[i].resize(cur_num);
			for (int j = 0; j < cur_num; j++)
			{
				task_src_off_x[i][j] = src_off_x[st_id + j];
				task_src_off_y[i][j] = src_off_y[st_id + j];
				task_src_rect_w[i][j] = src_rect_w[st_id + j];
				task_src_rect_h[i][j] = src_rect_h[st_id + j];
				task_secondBbox[i][j] = rnet_result[st_id + j];
			}
		}
	}

	int h = bgr_8uc3->height();
	int w = bgr_8uc3->width();
	int channels = bgr_8uc3->channels();

	std::shared_ptr<tensor<float> > task_rnet_images;
	for (int pp = 0; pp < need_thread_num; pp++)
	{
		std::shared_ptr<tensor<unsigned char>> roi_tensor, roi_resized_tensor;
		std::shared_ptr<tensor<float> > roi_resized_float_tensor;

		regressed_pading_.clear();
		for (int i = 0; i<task_secondBbox[pp].size(); i++) {
			FaceBox tempFace;
			tempFace.ymax = (task_secondBbox[pp][i].row2 >= h) ? h : task_secondBbox[pp][i].row2;
			tempFace.xmax = (task_secondBbox[pp][i].col2 >= w) ? w : task_secondBbox[pp][i].col2;
			tempFace.ymin = (task_secondBbox[pp][i].row1 <1) ? 1 : task_secondBbox[pp][i].row1;
			tempFace.xmin = (task_secondBbox[pp][i].col1 <1) ? 1 : task_secondBbox[pp][i].col1;
			regressed_pading_.push_back(tempFace);
		}

		task_rnet_images.reset(new tensor<float>(std::vector<int>{(int)task_secondBbox[pp].size(), channels, rnet_size, rnet_size}, device_id_, NCHW));
		float *input_data = nullptr;
		if (device_id_ < 0)
		{
			input_data = task_rnet_images->mutable_cpu_data();
		}
		else
		{
#ifdef USE_CUDA
			input_data = task_rnet_images->mutable_gpu_data();
#else
			NO_GPU;
#endif
		}

		for (int i = 0; i < task_secondBbox[pp].size(); i++)
		{
			int rect_h = (int)regressed_pading_[i].ymax - (int)regressed_pading_[i].ymin + 1;
			int rect_w = (int)regressed_pading_[i].xmax - (int)regressed_pading_[i].xmin + 1;

			rectangle<int> roi_rect((int)(regressed_pading_[i].xmin - 1), (int)(regressed_pading_[i].ymin - 1), rect_h, rect_w);
			float *input_data_i = input_data + task_rnet_images->offset(i);
			if (device_id_ < 0)
			{
				if (rect_h > 0 && rect_w > 0)
				{
					tensor_operation_cpu::safty_cut_cpu(bgr_8uc3, roi_tensor, &roi_rect);
					tensor_operation_cpu::resize_cpu(roi_tensor, roi_resized_tensor, rnet_size, rnet_size);
					tensor_operation_cpu::preprocess_tensors_cpu(roi_resized_tensor, roi_resized_float_tensor, means, var);
					memcpy(input_data_i, roi_resized_float_tensor->cpu_data(), channels * rnet_size * rnet_size * sizeof(float));
				}
				else
				{
					memset(input_data_i, 0, channels * rnet_size * rnet_size * sizeof(float));
				}
			}
			else
			{
#ifdef USE_CUDA
				if (rect_h > 0 && rect_w > 0)
				{
					tensor_operation_gpu::safty_cut_gpu(bgr_8uc3, roi_tensor, &roi_rect);
					tensor_operation_gpu::resize_gpu(roi_tensor, roi_resized_tensor, rnet_size, rnet_size);
					tensor_operation_gpu::preprocess_tensors_gpu(roi_resized_tensor, roi_resized_float_tensor, means, var);
					cudaMemcpy(input_data_i, roi_resized_float_tensor->gpu_data(), channels * rnet_size * rnet_size * sizeof(float), cudaMemcpyDefault);
				}
				else
				{
					cudaMemset(input_data_i, 0, channels * rnet_size * rnet_size * sizeof(float));
				}
#else
				NO_GPU;
#endif
			}
		}

		RNet_->Forward(task_rnet_images);

		std::shared_ptr<tensor<float> > confidence = RNet_->get_cls_prob();
		const float *confidence_data = confidence->cpu_data();

		std::shared_ptr<tensor<float> > location = RNet_->get_conv5_2();
		const float *location_data = location->cpu_data();

		int confidence_sliceStep = confidence->channels() * confidence->width() * confidence->height();

		int location_sliceStep = location->channels() * location->width() * location->height();
		int task_count = 0;
		for (int i = 0; i < task_secondBbox[pp].size(); i++)
		{
			if (confidence_data[i*confidence_sliceStep + 1] > thresh)
			{
				for (int j = 0; j < 4; j++)
					task_secondBbox[pp][i].regreCoord[j] = location_data[i*location_sliceStep + j];
				task_secondBbox[pp][i].area = task_src_rect_w[pp][i] * task_src_rect_h[pp][i];
				task_secondBbox[pp][i].score = confidence_data[i*confidence_sliceStep + 1];
				task_count++;
			}
			else
			{
				task_secondBbox[pp][i].exist = false;
			}
		}
		if (task_count < 1)
		{
			task_secondBbox[pp].clear();
			continue;
		}
		for (int i = task_secondBbox[pp].size() - 1; i >= 0; i--)
		{
			if (!task_secondBbox[pp][i].exist)
				task_secondBbox[pp].erase(task_secondBbox[pp].begin() + i);
		}
	}

	int count = 0;
	for (int i = 0; i < need_thread_num; i++)
	{
		count += task_secondBbox[i].size();
	}
	rnet_result.resize(count);

	std::vector<Longinus_CNN_OrderScore> secondScore;
	secondScore.resize(count);
	int id = 0;
	for (int i = 0; i < need_thread_num; i++)
	{
		for (int j = 0; j < task_secondBbox[i].size(); j++)
		{
			rnet_result[id] = task_secondBbox[i][j];
			secondScore[id].score = rnet_result[id].score;
			secondScore[id].oriOrder = id;
			id++;
		}
	}

	_nms(rnet_result, secondScore, nms_thresh, "Min");
	_refine_and_square_bbox(rnet_result, 'r', true);
	count = rnet_result.size();

	return true;
}


/// <summary>
/// O-NET
/// </summary>
/// <param name="rnetBbox">input refined boxes</param>
/// <param name="onet_result">final output boxes</param>
/// <param name="bgr_8uc3">image data</param>
/// <param name="min_size">minimum box window</param>
/// <param name="thresh">score threshold</param>
/// <param name="nms_thresh">nms threshold</param>
/// <param name="doLandmark">output landmarks</param>
bool glasssix::longinus::mtcnn_mobile::ONet_Process(std::vector<Longinus_CNN_BBox>& rnetBbox, std::vector<Longinus_CNN_BBox>& onet_result, 
	std::shared_ptr<tensor<unsigned char>>& bgr_8uc3, int min_size, float thresh, float nms_thresh, bool doLandmark)
{
	if (rnetBbox.size() == 0)
	{
		return false;
	}

	int batch_size = 64;
	onet_result.clear();
	std::vector<Longinus_CNN_BBox>::iterator it = rnetBbox.begin();
	std::vector<Longinus_CNN_OrderScore> thirdScore;
	std::vector<int> src_off_x, src_off_y, src_rect_w, src_rect_h;
	int o_count = 0;
	for (; it != rnetBbox.end(); it++)
	{
		if ((*it).exist)
		{
			int off_x = it->col1;
			int off_y = it->row1;
			int rect_w = it->col2 - off_x;
			int rect_h = it->row2 - off_y;
			if (rect_w <= 0.5*min_size || rect_h <= 0.5*min_size)
			{
				(*it).exist = false;
				continue;
			}
			else
			{
				src_off_x.push_back(off_x);
				src_off_y.push_back(off_y);
				src_rect_w.push_back(rect_w);
				src_rect_h.push_back(rect_h);
				o_count++;
				onet_result.push_back(*it);
			}
		}
	}


	int per_num = o_count;
	int need_thread_num = 1;
	if (per_num > batch_size)
	{
		need_thread_num = ceil((float)o_count / batch_size);
		per_num = batch_size;
	}

	std::vector<std::vector<int> > task_src_off_x(need_thread_num);
	std::vector<std::vector<int> > task_src_off_y(need_thread_num);
	std::vector<std::vector<int> > task_src_rect_w(need_thread_num);
	std::vector<std::vector<int> > task_src_rect_h(need_thread_num);
	std::vector<std::vector<Longinus_CNN_BBox> > task_thirdBbox(need_thread_num);

	for (int i = 0; i < need_thread_num; i++)
	{
		int st_id = per_num * i;
		int end_id = std::min(o_count, per_num*(i + 1));
		int cur_num = end_id - st_id;
		if (cur_num > 0)
		{
			task_src_off_x[i].resize(cur_num);
			task_src_off_y[i].resize(cur_num);
			task_src_rect_w[i].resize(cur_num);
			task_src_rect_h[i].resize(cur_num);
			task_thirdBbox[i].resize(cur_num);
			for (int j = 0; j < cur_num; j++)
			{
				task_src_off_x[i][j] = src_off_x[st_id + j];
				task_src_off_y[i][j] = src_off_y[st_id + j];
				task_src_rect_w[i][j] = src_rect_w[st_id + j];
				task_src_rect_h[i][j] = src_rect_h[st_id + j];
				task_thirdBbox[i][j] = onet_result[st_id + j];
			}
		}
	}

	int h = bgr_8uc3->height();
	int w = bgr_8uc3->width();
	int channels = bgr_8uc3->channels();

	std::shared_ptr<tensor<float> > task_onet_images;
	for (int pp = 0; pp < need_thread_num; pp++)
	{
		std::shared_ptr<tensor<unsigned char>> roi_tensor, roi_resized_tensor;
		std::shared_ptr<tensor<float> > roi_resized_float_tensor;

		regressed_pading_.clear();
		for (int i = 0; i<task_thirdBbox[pp].size(); i++) {
			FaceBox tempFace;
			tempFace.ymax = (task_thirdBbox[pp][i].row2 >= h) ? h : task_thirdBbox[pp][i].row2;
			tempFace.xmax = (task_thirdBbox[pp][i].col2 >= w) ? w : task_thirdBbox[pp][i].col2;
			tempFace.ymin = (task_thirdBbox[pp][i].row1 <1) ? 1 : task_thirdBbox[pp][i].row1;
			tempFace.xmin = (task_thirdBbox[pp][i].col1 <1) ? 1 : task_thirdBbox[pp][i].col1;
			regressed_pading_.push_back(tempFace);
		}

		task_onet_images.reset(new tensor<float>(std::vector<int>{(int)task_thirdBbox[pp].size(), channels, onet_size, onet_size}, device_id_, NCHW));
		float *input_data = nullptr;
		if (device_id_ < 0)
		{
			input_data = task_onet_images->mutable_cpu_data();
		}
		else
		{
#ifdef USE_CUDA
			input_data = task_onet_images->mutable_gpu_data();
#else
			NO_GPU;
#endif
		}

		float means[3] = { 104.0f, 117.0f, 124.0f };
		float var = 0.0078125;
		for (int i = 0; i < task_thirdBbox[pp].size(); i++)
		{
			int rect_h = (int)regressed_pading_[i].ymax - (int)regressed_pading_[i].ymin + 1;
			int rect_w = (int)regressed_pading_[i].xmax - (int)regressed_pading_[i].xmin + 1;

			rectangle<int> roi_rect((int)(regressed_pading_[i].xmin - 1), (int)(regressed_pading_[i].ymin - 1), rect_h, rect_w);
			float *input_data_i = input_data + task_onet_images->offset(i);
			if (device_id_ < 0)
			{
				if (rect_h > 0 && rect_w > 0)
				{
					tensor_operation_cpu::safty_cut_cpu(bgr_8uc3, roi_tensor, &roi_rect);
					tensor_operation_cpu::resize_cpu(roi_tensor, roi_resized_tensor, onet_size, onet_size);
					tensor_operation_cpu::preprocess_tensors_cpu(roi_resized_tensor, roi_resized_float_tensor, means, var);
					memcpy(input_data_i, roi_resized_float_tensor->cpu_data(), channels * onet_size * onet_size * sizeof(float));
				}
				else
				{
					memset(input_data_i, 0, channels * onet_size * onet_size * sizeof(float));
				}
			}
			else
			{
#ifdef USE_CUDA
				if (rect_h > 0 && rect_w > 0)
				{
					tensor_operation_gpu::safty_cut_gpu(bgr_8uc3, roi_tensor, &roi_rect);
					tensor_operation_gpu::resize_gpu(roi_tensor, roi_resized_tensor, onet_size, onet_size);
					tensor_operation_gpu::preprocess_tensors_gpu(roi_resized_tensor, roi_resized_float_tensor, means, var);
					cudaMemcpy(input_data_i, roi_resized_float_tensor->gpu_data(), channels * onet_size * onet_size * sizeof(float), cudaMemcpyDefault);
				}
				else
				{
					cudaMemset(input_data_i, 0, channels * onet_size * onet_size * sizeof(float));
				}
#else
				NO_GPU;
#endif
			}
		}

		ONet_->Forward(task_onet_images);

		std::shared_ptr<tensor<float> > confidence = ONet_->get_conv6_1();
		const float *confidence_data = confidence->cpu_data();

		std::shared_ptr<tensor<float> > location = ONet_->get_conv6_2();
		const float *location_data = location->cpu_data();

		std::shared_ptr<tensor<float> > headpose = ONet_->get_conv6_3();
		const float *headpose_data = headpose->cpu_data();

		std::shared_ptr<tensor<float> > pts = ONet_->get_conv6_4();
		const float *pts_data = pts->cpu_data();

		int task_count = 0;
		Longinus_CNN_OrderScore order;
		for (int i = 0; i < task_thirdBbox[pp].size(); i++)
		{
			if (confidence_data[i] > thresh)
			{
				memcpy(task_thirdBbox[pp][i].regreCoord, location_data + i * 4, 4 * sizeof(float));

				task_thirdBbox[pp][i].area = task_src_rect_w[pp][i] * task_src_rect_h[pp][i];
				task_thirdBbox[pp][i].score = confidence_data[i];
				memcpy(task_thirdBbox[pp][i].ppoint, pts_data + i * 10, 10 * sizeof(float));
				memcpy(task_thirdBbox[pp][i].headpose, headpose_data + i * 3, 3 * sizeof(float));

				int w = task_thirdBbox[pp][i].col2 - task_thirdBbox[pp][i].col1;
				int h = task_thirdBbox[pp][i].row2 - task_thirdBbox[pp][i].row1;
				for (int k = 0; k < 5; k++)
				{
					task_thirdBbox[pp][i].ppoint[k * 2] = task_thirdBbox[pp][i].ppoint[k * 2] * w + task_thirdBbox[pp][i].col1;
					task_thirdBbox[pp][i].ppoint[k * 2 + 1] = task_thirdBbox[pp][i].ppoint[k * 2 + 1] * h + task_thirdBbox[pp][i].row1;
				}

				for (int k = 0; k < 3; k++)
				{
					task_thirdBbox[pp][i].headpose[k] = task_thirdBbox[pp][i].headpose[k] * 90;
				}

				task_count++;
			}
			else
			{
				task_thirdBbox[pp][i].exist = false;
			}
		}

		if (task_count < 1)
		{
			task_thirdBbox[pp].clear();
			continue;
		}
		for (int i = task_thirdBbox[pp].size() - 1; i >= 0; i--)
		{
			if (!task_thirdBbox[pp][i].exist)
				task_thirdBbox[pp].erase(task_thirdBbox[pp].begin() + i);
		}
	}

	int count = 0;
	for (int i = 0; i < need_thread_num; i++)
	{
		count += task_thirdBbox[i].size();
	}
	onet_result.resize(count);
	thirdScore.resize(count);
	int id = 0;
	for (int i = 0; i < need_thread_num; i++)
	{
		for (int j = 0; j < task_thirdBbox[i].size(); j++)
		{
			onet_result[id] = task_thirdBbox[i][j];
			thirdScore[id].score = task_thirdBbox[i][j].score;
			thirdScore[id].oriOrder = id;
			id++;
		}
	}

	_refine_and_square_bbox(onet_result, 'o', true);
	_nms(onet_result, thirdScore, nms_thresh, "Min");

	return true;
}