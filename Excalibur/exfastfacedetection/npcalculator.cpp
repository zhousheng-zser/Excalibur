#include "npcalculator.hpp"
#include "model_data.hpp"
#include "guinevere.hpp"

namespace glasssix
{
	void npcalculator::init()
	{
		objSize = 0;
		numStages = 0;
		numScales = 0;
		numLeafNodes = 0;
		numBranchNodes = 0;
		scaleFactor = 0.0;
		device_ = -1;
	}

	void npcalculator::prepare(int os, int nst, int nb, int nl, float sf, int nsa)
	{
		objSize = os;
		numStages = nst;
		numBranchNodes = nb;
		numLeafNodes = nl;
		scaleFactor = sf;
		numScales = nsa;

		stageThreshold.reset(new tensor<float>(numStages, device_));
		treeRoot.reset(new tensor<int>(numStages, device_));
		pixelx.reset(new tensor<int>(numScales * numBranchNodes, device_));
		pixely.reset(new tensor<int>(numScales * numBranchNodes, device_));
		cutpoint.reset(new tensor<unsigned char>(2 * numBranchNodes, device_));
		leftChild.reset(new tensor<int>(numBranchNodes, device_));
		rightChild.reset(new tensor<int>(numBranchNodes, device_));
		fit.reset(new tensor<float>(numLeafNodes, device_));
		winSize.reset(new tensor<int>(numScales, device_));
		fea.reset(new tensor<unsigned char>(256 * 256, device_));
	}


	void npcalculator::load()
	{
		objSize = obj_Size;
		numStages = num_Stages;
		numBranchNodes = num_Branch_Nodes;
		numLeafNodes = num_Leaf_Nodes;
		scaleFactor = scale_Factor;
		numScales = num_Scales;
		modelversion = model_version;
		//
		stageThreshold.reset(new tensor<float>(numStages, device_));
		treeRoot.reset(new tensor<int>(numStages, device_));
		pixelx.reset(new tensor<int>(numScales * numBranchNodes, device_));
		pixely.reset(new tensor<int>(numScales * numBranchNodes, device_));
		cutpoint.reset(new tensor<unsigned char>(2 * numBranchNodes, device_));
		leftChild.reset(new tensor<int>(numBranchNodes, device_));
		rightChild.reset(new tensor<int>(numBranchNodes, device_));
		fit.reset(new tensor<float>(numLeafNodes, device_));
		winSize.reset(new tensor<int>(numScales, device_));
		fea.reset(new tensor<unsigned char>(256 * 256, device_));
		//
		memcpy(stageThreshold->mutable_cpu_data(), stage_Threshold, sizeof(stage_Threshold));
		memcpy(treeRoot->mutable_cpu_data(), tree_Root, sizeof(tree_Root));
		memcpy(pixelx->mutable_cpu_data(), pixel_x, sizeof(pixel_x));
		memcpy(pixely->mutable_cpu_data(), pixel_y, sizeof(pixel_y));
		memcpy(cutpoint->mutable_cpu_data(), cut_point, sizeof(cut_point));
		memcpy(leftChild->mutable_cpu_data(), left_Child, sizeof(left_Child));
		memcpy(rightChild->mutable_cpu_data(), right_Child, sizeof(right_Child));
		memcpy(fit->mutable_cpu_data(), fit_, sizeof(fit_));
		memcpy(winSize->mutable_cpu_data(), win_Size, sizeof(win_Size));
		memcpy(fea->mutable_cpu_data(), npdTable, 256 * 256 * sizeof(unsigned char));
	}

	void npcalculator::load(const char* modelpath)
	{
		int n = 0;
		int os, nst, nb, nl, nsa;
		float sf;
		FILE* fp = fopen(modelpath, "rb");
		size_t rs;
		//
		rs = fread(&os, sizeof(int), 1, fp);//objSize
		rs = fread(&nst, sizeof(int), 1, fp);//numStages
		rs = fread(&nb, sizeof(int), 1, fp);//numBranchNode
		rs = fread(&nl, sizeof(int), 1, fp);//numLeafNode
		rs = fread(&sf, sizeof(float), 1, fp);//scaleFactor
		rs = fread(&nsa, sizeof(int), 1, fp);//numScales
											 // Malloc space.
		prepare(os, nst, nb, nl, sf, nsa);

		rs = fread(stageThreshold->mutable_cpu_data(), sizeof(float), numStages, fp);
		n += rs;
		rs = fread(treeRoot->mutable_cpu_data(), sizeof(int), numStages, fp);
		n += rs;
		for (int i = 0; i < numScales; i++)
		{
			rs = fread(pixelx->mutable_cpu_data() + i * numBranchNodes, sizeof(int), numBranchNodes, fp);
			n += rs;
		}
		for (int i = 0; i < numScales; i++)
		{
			rs = fread(pixely->mutable_cpu_data() + i * numBranchNodes, sizeof(int), numBranchNodes, fp);
			n += rs;
		}
		for (int i = 0; i < 2; i++)
		{
			rs = fread(cutpoint->mutable_cpu_data() + i * numBranchNodes, sizeof(unsigned char), numBranchNodes, fp);
			n += rs;
		}
		rs = fread(leftChild->mutable_cpu_data(), sizeof(int), numBranchNodes, fp);
		n += rs;
		rs = fread(rightChild->mutable_cpu_data(), sizeof(int), numBranchNodes, fp);
		n += rs;
		rs = fread(fit->mutable_cpu_data(), sizeof(float), numLeafNodes, fp);
		n += rs;
		rs = fread(winSize->mutable_cpu_data(), sizeof(int), numScales, fp);
		n += rs;
		fclose(fp);
		memcpy(fea->mutable_cpu_data(), npdTable, 256 * 256 * sizeof(unsigned char));
	}

	npcalculator::npcalculator()
	{
		init();
		load();
	}

	npcalculator::npcalculator(int device)
	{
		init();
		device_ = device;
		load();
	}

	npcalculator::npcalculator(const char* modelpath, int device)
	{
		init();
		device_ = device;
		load(modelpath);
	}

	/*npcalculator::~npcalculator()
	{
		
	}*/

}