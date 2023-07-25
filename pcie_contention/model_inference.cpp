// link to the original code https://github.com/future-xy/Serverless-sample-apps/blob/main/python/opt/inference.cpp
// This script loads a TorchScript model and runs inference on a given input.

#include <torch/script.h> // One-stop header.
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: inference <model_path>" << std::endl;
    return -1;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  torch::jit::script::Module module = torch::jit::load(argv[1]);

  // Create a vector of inputs.
  // input_ids = [2, 25216, 47, 236, 16]
  // attention_mask = [1, 1, 1, 1, 1]
  std::vector<torch::jit::IValue> inputs;
  std::vector<torch::jit::IValue> inputs;

  torch::Tensor input_ids = torch::tensor({2, 25216, 47, 236, 16}).unsqueeze(0).repeat({256, 1});
  torch::Tensor attention_mask = torch::tensor({1, 1, 1, 1, 1}).unsqueeze(0).repeat({256, 1});
  
  inputs.push_back(input_ids);
  inputs.push_back(attention_mask);
  
  std::vector<double> inferenceTimings; // Vector to store inference timings

  // Perform 1000 forward passes and measure the inference timings
  for (int i = 0; i < 1000; ++i) {
    auto start = std::chrono::steady_clock::now();
    
    // Execute the model and turn its output into a tensor.
    torch::jit::IValue output = module.forward(inputs);
    
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;
    
    inferenceTimings.push_back(duration.count());
  }

  std::ofstream timingsFile("inference_timings.txt");
  if (timingsFile.is_open()) {
    for (const auto& timing : inferenceTimings) {
      timingsFile << timing << std::endl;
    }
    timingsFile.close();
    std::cout << "Inference timings stored in 'inference_timings.txt'." << std::endl;
  }
  else {
    std::cerr << "Unable to open file for writing." << std::endl;
  }


  return 0;
}
