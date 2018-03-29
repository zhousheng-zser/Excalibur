#include "cunpd.hpp"
#include <memory>
#include "nplogic.hpp"


namespace glasssix
{
	std::vector<std::shared_ptr<nplogic>> npd_models_;

	int cunpd::AddNpdModel(int device)
	{
		std::shared_ptr<nplogic> new_model = std::make_shared<nplogic>(device);
		new_model->load();
		npd_models_.push_back(new_model);
		return npd_models_.size() - 1;
	}

	int cunpd::AddNpdModel(std::string modelpath, int device)
	{
		std::shared_ptr<nplogic> new_model = std::make_shared<nplogic>(device);
		new_model->load(modelpath.c_str());
		npd_models_.push_back(new_model);
		return npd_models_.size() - 1;
	}

	std::vector<FaceInfomation> cunpd::detect(cv::Mat img, int model_id, int min_size)
	{
		int n = npd_models_[model_id]->detect(img.data, img.cols, img.rows, min_size);
		std::vector< int >& Xs = npd_models_[model_id]->getXs();
		std::vector< int >& Ys = npd_models_[model_id]->getYs();
		std::vector< int >& Ss = npd_models_[model_id]->getSs();
		std::vector< float >& Scores = npd_models_[model_id]->getScores();
		std::vector<FaceInfomation> output;
		for (size_t i = 0; i < n; i++)
		{
			output.push_back({ cv::Rect(Xs[i], Ys[i], Ss[i], Ss[i]), Scores[i] });
		}
		return output;
	}

}