#include <vector>
#include "retina_face.hpp"

using namespace glasssix;

template <typename Dtype>
static void mat2tensor_cpu(const cv::Mat &srcu, std::shared_ptr<glasssix::memory::tensor<Dtype>>& dst, 
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

//processing
anchor_win  _whctrs(anchor_box anchor)
{
    //Return width, height, x center, and y center for an anchor (window).
    anchor_win win;
    win.w = anchor.x2 - anchor.x1 + 1;
    win.h = anchor.y2 - anchor.y1 + 1;
    win.x_ctr = anchor.x1 + 0.5 * (win.w - 1);
    win.y_ctr = anchor.y1 + 0.5 * (win.h - 1);

    return win;
}

anchor_box _mkanchors(anchor_win win)
{
    //Given a vector of widths (ws) and heights (hs) around a center
    //(x_ctr, y_ctr), output a set of anchors (windows).
    anchor_box anchor;
    anchor.x1 = win.x_ctr - 0.5 * (win.w - 1);
    anchor.y1 = win.y_ctr - 0.5 * (win.h - 1);
    anchor.x2 = win.x_ctr + 0.5 * (win.w - 1);
    anchor.y2 = win.y_ctr + 0.5 * (win.h - 1);

    return anchor;
}

std::vector<anchor_box> _ratio_enum(anchor_box anchor, std::vector<float> ratios)
{
    //Enumerate a set of anchors for each aspect ratio wrt an anchor.
	std::vector<anchor_box> anchors;
    for(size_t i = 0; i < ratios.size(); i++) 
	{
        anchor_win win = _whctrs(anchor);
        float size = win.w * win.h;
        float scale = size / ratios[i];

        win.w = std::round(sqrt(scale));
        win.h = std::round(win.w * ratios[i]);

        anchor_box tmp = _mkanchors(win);
        anchors.push_back(tmp);
    }

    return anchors;
}

std::vector<anchor_box> _scale_enum(anchor_box anchor, std::vector<int> scales)
{
    //Enumerate a set of anchors for each scale wrt an anchor.
	std::vector<anchor_box> anchors;
    for(size_t i = 0; i < scales.size(); i++) 
	{
        anchor_win win = _whctrs(anchor);

        win.w = win.w * scales[i];
        win.h = win.h * scales[i];

        anchor_box tmp = _mkanchors(win);
        anchors.push_back(tmp);
    }

    return anchors;
}

std::vector<anchor_box> generate_anchors(int base_size = 16, std::vector<float> ratios = {0.5, 1, 2},
                                         std::vector<int> scales = {8, 64}, int stride = 16, bool dense_anchor = false)
{
    //Generate anchor (reference) windows by enumerating aspect ratios X
    //scales wrt a reference (0, 0, 15, 15) window.

    anchor_box base_anchor;
    base_anchor.x1 = 0;
    base_anchor.y1 = 0;
    base_anchor.x2 = base_size - 1;
    base_anchor.y2 = base_size - 1;

	std::vector<anchor_box> ratio_anchors;
    ratio_anchors = _ratio_enum(base_anchor, ratios);

	std::vector<anchor_box> anchors;
    for(size_t i = 0; i < ratio_anchors.size(); i++) 
	{
	    std::vector<anchor_box> tmp = _scale_enum(ratio_anchors[i], scales);
        anchors.insert(anchors.end(), tmp.begin(), tmp.end());
    }

    if(dense_anchor) 
	{
        assert(stride % 2 == 0);
	    std::vector<anchor_box> anchors2 = anchors;
        for(size_t i = 0; i < anchors2.size(); i++) {
            anchors2[i].x1 += stride / 2;
            anchors2[i].y1 += stride / 2;
            anchors2[i].x2 += stride / 2;
            anchors2[i].y2 += stride / 2;
        }
        anchors.insert(anchors.end(), anchors2.begin(), anchors2.end());
    }

    return anchors;
}

std::vector<std::vector<anchor_box>> generate_anchors_fpn(bool dense_anchor = false, std::vector<anchor_cfg> cfg = {})
{
    //Generate anchor (reference) windows by enumerating aspect ratios X
    //scales wrt a reference (0, 0, 15, 15) window.

	std::vector<std::vector<anchor_box>> anchors;
    for(size_t i = 0; i < cfg.size(); i++) 
	{
        //stride从小到大[32 16 8]
        anchor_cfg tmp = cfg[i];
        int bs = tmp.BASE_SIZE;
		std::vector<float> ratios = tmp.RATIOS;
		std::vector<int> scales = tmp.SCALES;
        int stride = tmp.STRIDE;

		std::vector<anchor_box> r = generate_anchors(bs, ratios, scales, stride, dense_anchor);
        anchors.push_back(r);
    }

    return anchors;
}

std::vector<anchor_box> anchors_plane(int height, int width, int stride, std::vector<anchor_box> base_anchors)
{
    /*
    height: height of plane
    width:  width of plane
    stride: stride ot the original image
    anchors_base: a base set of anchors
    */

	std::vector<anchor_box> all_anchors;
    for(size_t k = 0; k < base_anchors.size(); k++) {
        for(int ih = 0; ih < height; ih++) {
            int sh = ih * stride;
            for(int iw = 0; iw < width; iw++) {
                int sw = iw * stride;

                anchor_box tmp;
                tmp.x1 = base_anchors[k].x1 + sw;
                tmp.y1 = base_anchors[k].y1 + sh;
                tmp.x2 = base_anchors[k].x2 + sw;
                tmp.y2 = base_anchors[k].y2 + sh;
                all_anchors.push_back(tmp);
            }
        }
    }

    return all_anchors;
}

void clip_boxes(std::vector<anchor_box> &boxes, int width, int height)
{
    //Clip boxes to image boundaries.
    for(size_t i = 0; i < boxes.size(); i++) 
	{
        if(boxes[i].x1 < 0) 
		{
            boxes[i].x1 = 0;
        }
        if(boxes[i].y1 < 0) 
		{
            boxes[i].y1 = 0;
        }
        if(boxes[i].x2 > width - 1) 
		{
            boxes[i].x2 = width - 1;
        }
        if(boxes[i].y2 > height - 1) 
		{
            boxes[i].y2 = height -1;
        }
//        boxes[i].x1 = std::max<float>(std::min<float>(boxes[i].x1, width - 1), 0);
//        boxes[i].y1 = std::max<float>(std::min<float>(boxes[i].y1, height - 1), 0);
//        boxes[i].x2 = std::max<float>(std::min<float>(boxes[i].x2, width - 1), 0);
//        boxes[i].y2 = std::max<float>(std::min<float>(boxes[i].y2, height - 1), 0);
    }
}

void clip_boxes(anchor_box &box, int width, int height)
{
    //Clip boxes to image boundaries.
    if(box.x1 < 0) {
        box.x1 = 0;
    }
    if(box.y1 < 0) {
        box.y1 = 0;
    }
    if(box.x2 > width - 1) {
        box.x2 = width - 1;
    }
    if(box.y2 > height - 1) {
        box.y2 = height -1;
    }
//    boxes[i].x1 = std::max<float>(std::min<float>(boxes[i].x1, width - 1), 0);
//    boxes[i].y1 = std::max<float>(std::min<float>(boxes[i].y1, height - 1), 0);
//    boxes[i].x2 = std::max<float>(std::min<float>(boxes[i].x2, width - 1), 0);
//    boxes[i].y2 = std::max<float>(std::min<float>(boxes[i].y2, height - 1), 0);

}

//######################################################################
//retina_face
//######################################################################

retina_face::retina_face(std::string &model, std::string network, float nms)
    : network(network), nms_threshold(nms)
{
    //主干网络选择
    int fmc = 3;
	ratio_ = { 1.0 };
    //anchor setting
	feat_stride_fpn_ = { 32, 16, 8 };
	anchor_cfg tmp;
	tmp.SCALES = { 32, 16 };
	tmp.BASE_SIZE = 16;
	tmp.RATIOS = ratio_;
	tmp.ALLOWED_BORDER = 9999;
	tmp.STRIDE = 32;
	cfg_.push_back(tmp);

	tmp.SCALES = { 8, 4 };
	tmp.BASE_SIZE = 16;
	tmp.RATIOS = ratio_;
	tmp.ALLOWED_BORDER = 9999;
	tmp.STRIDE = 16;
	cfg_.push_back(tmp);

	tmp.SCALES = { 2, 1 };
	tmp.BASE_SIZE = 16;
	tmp.RATIOS = ratio_;
	tmp.ALLOWED_BORDER = 9999;
	tmp.STRIDE = 8;
	cfg_.push_back(tmp);

	/* Load the network. */
	pipe.reset(new glasssix::excalibur::pipeline<float>("D:\\Research\\Excalibur\\models\\retina.phai", "D:\\Research\\Excalibur\\models\\retina.racy"));
	bool dense_anchor = false;
	std::vector<std::vector<anchor_box>> anchors_fpn = generate_anchors_fpn(dense_anchor, cfg_);
	for (size_t i = 0; i < anchors_fpn.size(); i++) 
	{
		std::string key = "stride" + std::to_string(feat_stride_fpn_[i]);
		anchors_fpn_[key] = anchors_fpn[i];
		num_anchors_[key] = anchors_fpn[i].size();
	}
}

retina_face::~retina_face()
{
	
}

std::vector<anchor_box> retina_face::bbox_pred(std::vector<anchor_box> anchors, std::vector<cv::Vec4f> regress)
{
    //"""
    //  Transform the set of class-agnostic boxes into class-specific boxes
    //  by applying the predicted offsets (box_deltas)
    //  :param boxes: !important [N 4]
    //  :param box_deltas: [N, 4 * num_classes]
    //  :return: [N 4 * num_classes]
    //  """

	std::vector<anchor_box> rects(anchors.size());
    for(size_t i = 0; i < anchors.size(); i++) 
	{
        float width = anchors[i].x2 - anchors[i].x1 + 1;
        float height = anchors[i].y2 - anchors[i].y1 + 1;
        float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
        float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

        float pred_ctr_x = regress[i][0] * width + ctr_x;
        float pred_ctr_y = regress[i][1] * height + ctr_y;
        float pred_w = exp(regress[i][2]) * width;
        float pred_h = exp(regress[i][3]) * height;

        rects[i].x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
        rects[i].y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
        rects[i].x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
        rects[i].y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);
    }

    return rects;
}

anchor_box retina_face::bbox_pred(anchor_box anchor, cv::Vec4f regress)
{
    anchor_box rect;

    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
    float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

    float pred_ctr_x = regress[0] * width + ctr_x;
    float pred_ctr_y = regress[1] * height + ctr_y;
    float pred_w = exp(regress[2]) * width;
    float pred_h = exp(regress[3]) * height;

    rect.x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
    rect.y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
    rect.x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
    rect.y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);

    return rect;
}

std::vector<face_pts> retina_face::landmark_pred(std::vector<anchor_box> anchors, std::vector<face_pts> facepts)
{
	std::vector<face_pts> pts(anchors.size());
    for(size_t i = 0; i < anchors.size(); i++) 
	{
        float width = anchors[i].x2 - anchors[i].x1 + 1;
        float height = anchors[i].y2 - anchors[i].y1 + 1;
        float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
        float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

        for(size_t j = 0; j < 5; j ++) 
		{
            pts[i].x[j] = facepts[i].x[j] * width + ctr_x;
            pts[i].y[j] = facepts[i].y[j] * height + ctr_y;
        }
    }

    return pts;
}

face_pts retina_face::landmark_pred(anchor_box anchor, face_pts facePt)
{
    face_pts pt;
    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
    float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

    for(size_t j = 0; j < 5; j ++) 
	{
        pt.x[j] = facePt.x[j] * width + ctr_x;
        pt.y[j] = facePt.y[j] * height + ctr_y;
    }

    return pt;
}

bool retina_face::compare_bbox(const face_detect_info & a, const face_detect_info & b)
{
    return a.score > b.score;
}

std::vector<face_detect_info> retina_face::nms(std::vector<face_detect_info>& bboxes, float threshold)
{
    std::vector<face_detect_info> bboxes_nms;
    std::sort(bboxes.begin(), bboxes.end(), compare_bbox);

    int32_t select_idx = 0;
    int32_t num_bbox = static_cast<int32_t>(bboxes.size());
    std::vector<int32_t> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged) 
	{
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        //如果全部执行完则返回
        if (select_idx == num_bbox) 
		{
            all_merged = true;
            continue;
        }

        bboxes_nms.push_back(bboxes[select_idx]);
        mask_merged[select_idx] = 1;

        anchor_box select_bbox = bboxes[select_idx].rect;
        float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) * (select_bbox.y2 - select_bbox.y1 + 1));
        float x1 = static_cast<float>(select_bbox.x1);
        float y1 = static_cast<float>(select_bbox.y1);
        float x2 = static_cast<float>(select_bbox.x2);
        float y2 = static_cast<float>(select_bbox.y2);

        select_idx++;
        for (int32_t i = select_idx; i < num_bbox; i++) 
		{
            if (mask_merged[i] == 1)
                continue;

            anchor_box& bbox_i = bboxes[i].rect;
            float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
            float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
            float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;   //<- float 型不加1
            float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
            if (w <= 0 || h <= 0)
                continue;

            float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1));
            float area_intersect = w * h;

   
            if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > threshold) 
			{
                mask_merged[i] = 1;
            }
        }
    }

    return bboxes_nms;
}


std::vector<face_detect_info> retina_face::detect(cv::Mat img, float threshold, float scales)
{
    if(img.empty()) 
	{
        return std::vector<face_detect_info>();
    }
	//cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2));
    double pre = (double)cv::getTickCount();
    int ws = (img.cols + 31) / 32 * 32;
    int hs = (img.rows + 31) / 32 * 32;

    cv::copyMakeBorder(img, img, 0, hs - img.rows, 0, ws - img.cols, cv::BORDER_CONSTANT,cv::Scalar(0));

    cv::Mat src = img.clone();

    pre = (double)cv::getTickCount() - pre;
    std::cout << "pre compute time :" << pre*1000.0 / cv::getTickFrequency() << " ms \n";

    //LOG(INFO) << "Start net_->Forward()";
    double t1 = (double)cv::getTickCount();
    //Net_->Forward();
	//auto blob_data = mclc.Forward(std::vector<cv::Mat>{img}, net_id);
	std::shared_ptr<glasssix::memory::tensor<float>> temp;
	mat2tensor_cpu(img, temp, glasssix::memory::NCHW, false);
	auto blob_data = pipe->forward_cpu(temp);
    t1 = (double)cv::getTickCount() - t1;
    std::cout << "infer compute time :" << t1*1000.0 / cv::getTickFrequency() << " ms \n";
    //LOG(INFO) << "Done net_->Forward()";

    double post = (double)cv::getTickCount();
    std::string name_bbox = "face_rpn_bbox_pred_";
    std::string name_score ="face_rpn_cls_prob_reshape_";
    std::string name_landmark ="face_rpn_landmark_pred_";

	std::vector<face_detect_info> faceInfo;
    for(size_t i = 0; i < feat_stride_fpn_.size(); i++) 
	{
///////////////////////////////////////////////
        double s1 = (double)cv::getTickCount();
///////////////////////////////////////////////
        std::string key = "stride" + std::to_string(feat_stride_fpn_[i]);
        int stride = feat_stride_fpn_[i];
		
        std::string str = name_score + key;
        //const boost::shared_ptr<Blob<float>> score_blob = Net_->blob_by_name(str);
		//caffe::DataBlob score_blob = mclc.GetBlobData(str, net_id);
		auto score_blob = pipe->get_featmap(str);
		auto score_blob_count = score_blob->count();
        const float* scoreB = score_blob->cpu_data() + score_blob_count / 2;
        const float* scoreE = scoreB + score_blob_count / 2;
        std::vector<float> score = std::vector<float>(scoreB, scoreE);

        str = name_bbox + key;
        //const boost::shared_ptr<Blob<float>> bbox_blob = Net_->blob_by_name(str);
		//caffe::DataBlob bbox_blob = mclc.GetBlobData(str, net_id);
		auto bbox_blob = pipe->get_featmap(str);
		auto bbox_blob_count = bbox_blob->count();
        const float* bboxB = bbox_blob->cpu_data();
        const float* bboxE = bboxB + bbox_blob_count;
        std::vector<float> bbox_delta = std::vector<float>(bboxB, bboxE);

        str = name_landmark + key;
        //const boost::shared_ptr<Blob<float>> landmark_blob = Net_->blob_by_name(str);
		//caffe::DataBlob landmark_blob = mclc.GetBlobData(str, net_id);
		auto landmark_blob = pipe->get_featmap(str);
		auto landmark_blob_count = landmark_blob->count();
        const float* landmarkB = landmark_blob->cpu_data();
        const float* landmarkE = landmarkB + landmark_blob_count;
        std::vector<float> landmark_delta = std::vector<float>(landmarkB, landmarkE);

		int width = score_blob->width();//score_blob->width();
        int height = score_blob->height();; //score_blob->height();
        size_t count = width * height;
        size_t num_anchor = num_anchors_[key];

///////////////////////////////////////////////
        s1 = (double)cv::getTickCount() - s1;
        std::cout << "s1 compute time :" << s1*1000.0 / cv::getTickFrequency() << " ms \n";
///////////////////////////////////////////////

        //存储顺序 h * w * num_anchor
		std::vector<anchor_box> anchors = anchors_plane(height, width, stride, anchors_fpn_[key]);

        for(size_t num = 0; num < num_anchor; num++) 
		{
            for(size_t j = 0; j < count; j++) 
			{
                //置信度小于阈值跳过
                float conf = score[j + count * num];
                if(conf <= threshold) 
				{
                    continue;
                }

                cv::Vec4f regress;
                float dx = bbox_delta[j + count * (0 + num * 4)];
                float dy = bbox_delta[j + count * (1 + num * 4)];
                float dw = bbox_delta[j + count * (2 + num * 4)];
                float dh = bbox_delta[j + count * (3 + num * 4)];
                regress = cv::Vec4f(dx, dy, dw, dh);

                //回归人脸框
                anchor_box rect = bbox_pred(anchors[j + count * num], regress);
                //越界处理
                clip_boxes(rect, ws, hs);

                face_pts pts;
                for(size_t k = 0; k < 5; k++) 
				{
                    pts.x[k] = landmark_delta[j + count * (num * 10 + k * 2)];
                    pts.y[k] = landmark_delta[j + count * (num * 10 + k * 2 + 1)];
                }
                //回归人脸关键点
                face_pts landmarks = landmark_pred(anchors[j + count * num], pts);

                face_detect_info tmp;
                tmp.score = conf;
                tmp.rect = rect;
                tmp.pts = landmarks;
                faceInfo.push_back(tmp);
            }
        }
    }

    //排序nms
    faceInfo = nms(faceInfo, nms_threshold);

    post = (double)cv::getTickCount() - post;
    std::cout << "post compute time :" << post*1000.0 / cv::getTickFrequency() << " ms \n";
	return faceInfo;

    
}

