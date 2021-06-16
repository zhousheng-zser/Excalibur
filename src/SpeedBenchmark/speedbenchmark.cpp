/*
*SpeedBenchmark can be used to test neural network inference performance
*Only the network definition files (phai file) are required.
*The large model binary files (racy file) are not loaded but generated randomly for speed test.
*All CNN models will load in "../../models/" dir automatically.
*/

#include <iostream>
#include <fstream>
#include <random>
#include <cfloat>
#include "../../include/Excalibur/pipeline.hpp"
#include "../../include/Primitives/profiler.hpp"
#include "../../include/Primitives/tensor_conversions.hpp"

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
    std::vector<std::tuple<std::string, std::vector<int>, int>> pipe_infos =
        {
            {"rec_crnn_resnet34", {1, 3, 32, 100}, -1}
            // {"pfld11_landmark65_simp", {1, 3, 112, 112}, -1}
            // {"pfld_land71_simp", {1, 3, 80, 80}, -1}
            // {"pfld_attri_simp", {1, 3, 80, 80}, -1}
            // {"hat_simp-opt", {1, 3, 640, 640}, 0}
            // {"longinus", {1, 3, 240, 320}, 0}
            // ,{"longinus", {1, 3, 240, 320}, -1}
            //,{"mobile_unicorn_666398_usefulpart_merged", {1, 3, 128, 128}}
            // ,{"unicorn", {1, 3, 128, 128}, 0}
            //,{"unicorn_gdc_0x1C_int8_w6a8", {1, 3, 128, 128}, -1}
            //{"unicorn_li_0x42_usefulpart", {1, 3, 128, 128}}
            //,{"unicorn_li_0x42_usefulpart_int8", {1, 3, 128, 128}}
            // ,{"mobile_unicorn", {1, 3, 128, 128}, 0}
            // ,{"FASMV2_merged", {1, 3, 80, 80}, -1}
            //,{"mobile_unicorn_666398_usefulpart_merged", {1, 3, 128, 128}, -1}
            //,{"mobile_unicorn_666398_usefulpart_merged_int8", {1, 3, 128, 128}}
        };

    timer t;
    std::vector<excalibur::pipeline<float> *> pipes;
    int warmup_loop_count = 0;
    int loop_count = 1;

    for (size_t i = 0; i < pipe_infos.size(); i++)
    {
        // pipes.push_back(new excalibur::pipeline<float>(std::string("C:/Users/Glasssix-ZYF/Desktop/models/") + std::get<0>(pipe_infos[i]) + ".phai", std::string("C:/Users/Glasssix-ZYF/Desktop/models/") + std::get<0>(pipe_infos[i]) + ".racy", std::get<2>(pipe_infos[i])));
        pipes.push_back(new excalibur::pipeline<float>(std::string("../../../models/") + std::get<0>(pipe_infos[i]) + ".phai", std::get<2>(pipe_infos[i])));
    }
    std::cout << "Pipeline\t Min\t Max\t Ave " << std::endl;

    //std::ifstream in("1.bin", std::ios::binary);
    //std::shared_ptr<memory::tensor<uint8_t>> input_tensor_u8(new memory::tensor<uint8_t>(std::vector<int>{1, 3, 128, 128}, -1, memory::NCHW));
    //in.read((char *)input_tensor_u8->mutable_cpu_data(), 3 * 128 * 128);
    //auto input_tensor  = input_tensor_u8 | memory::tensor_convert_to<float>;
    for (size_t i = 0; i < pipe_infos.size(); i++)
    {
        //excalibur::pipeline<float> pipe(std::string("../../models/") + pipe_infos[i].first + ".phai", -1);
        auto allocator = new memory::pool_allocator<float>();
        auto input_tensor = get_random_tensor(std::get<1>(pipe_infos[i]));
        input_tensor->set_allocator(allocator);
        // Warming up
        for (size_t j = 0; j < warmup_loop_count; j++)
        {
            pipes[i]->forward(input_tensor);
        }

        double time_min = DBL_MAX;
        double time_max = -DBL_MAX;
        double time_avg = 0;

        for (size_t j = 0; j < loop_count; j++)
        {
            profiler *p = profiler::get();
            pipes[i]->enable_profiler();
            t.start();
            pipes[i]->forward(input_tensor);
            t.stop();
            double time = t.get_elapsed_milli_seconds();
            time_min = std::min(time_min, time);
            time_max = std::max(time_max, time);
            time_avg += time;
        }
        time_avg /= loop_count;
        std::cout << std::get<0>(pipe_infos[i]) << "\t" << time_min << "\t" << time_max << "\t" << time_avg << std::endl;

        delete pipes[i];
        pipes[i] = nullptr;
    }

    return 0;
}