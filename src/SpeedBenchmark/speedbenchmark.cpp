/*
*SpeedBenchmark can be used to test neural network inference performance
*Only the network definition files (phai file) are required.
*The large model binary files (racy file) are not loaded but generated randomly for speed test.
*All CNN models will load in "../../models/" dir automatically.
*/

#include <iostream>
#include <random>
#include "../../include/Excalibur/pipeline.hpp"
#include "../../include/Primitives/profiler.hpp"

using namespace glasssix;

std::shared_ptr<memory::tensor<float>> get_random_tensor(std::vector<int> input_shape)
{
	std::default_random_engine e;
	std::normal_distribution<float> n(128, 64);
	std::shared_ptr<memory::tensor<float>> out;
	out.reset(new memory::tensor<float>(input_shape, -1, memory::NCHW));
	for (size_t i = 0; i < out->count(); i++)
	{
		out->mutable_cpu_data()[i] = n(e);
	}
	return out;
}

int main()
{
	std::vector<std::pair<std::string, std::vector<int>>> pipe_infos =
	{
		{"retina", {1, 3, 240, 320}}
		,{"unicorn", {1, 3, 128, 128}}
		,{"unicorn_int8", {1, 3, 128, 128}}
		,{"unicorn_li", {1, 3, 128, 128}}
		,{"mobile_unicorn", {1, 3, 128, 128}}
	};

	timer t;
	std::vector<excalibur::pipeline<float>*> pipes;
	int warmup_loop_count = 2;
	int loop_count = 100;

	for (size_t i = 0; i < pipe_infos.size(); i++)
	{
		pipes.push_back(new excalibur::pipeline<float>(std::string("../../models/") + pipe_infos[i].first + ".phai", -1));
	}
	std::cout << "Pipeline\t Min\t Max\t Ave " << std::endl;
	for (size_t i = 0; i < pipe_infos.size(); i++)
	{
		//excalibur::pipeline<float> pipe(std::string("../../models/") + pipe_infos[i].first + ".phai", -1);
		auto allocator = new memory::pool_allocator<float>();
		auto input_tensor = get_random_tensor(pipe_infos[i].second);
		input_tensor->set_allocator(allocator);
		// Warming up
		for (size_t j = 0; j < warmup_loop_count; j++)
		{
			pipes[i]->forward_cpu(input_tensor);
		}

		double time_min = DBL_MAX;
		double time_max = -DBL_MAX;
		double time_avg = 0;

		for (size_t j = 0; j < loop_count; j++)
		{
			t.start();
			pipes[i]->forward_cpu(input_tensor);
			t.stop();

			double time = t.get_elapsed_milli_seconds();
			time_min = std::min(time_min, time);
			time_max = std::max(time_max, time);
			time_avg += time;
		}
		time_avg /= loop_count;
		std::cout << pipe_infos[i].first << "\t" << time_min << "\t" << time_max << "\t" << time_avg << std::endl; 
		
		delete pipes[i];
		pipes[i] = nullptr;
	}

	return 0;
}