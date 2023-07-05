#include <torch/script.h>
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

  // Create pinned memory for input tensors
  auto options = torch::TensorOptions().device(torch::kCUDA).pinned_memory(true);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::empty({1, 5}, options).copy_(torch::tensor({2, 25216, 47, 236, 16})));
  inputs.push_back(torch::empty({1, 5}, options).copy_(torch::tensor({1, 1, 1, 1, 1})));

  std::vector<torch::Tensor> outputTensors;
  std::vector<double> inferenceTimings; // Vector to store inference timings

  // Perform 1000 forward passes and measure the inference timings
  for (int i = 0; i < 1000; ++i) {
    auto start = std::chrono::steady_clock::now();

    // Execute the model and turn its output into a tensor.
    torch::jit::IValue output = module.forward(inputs);
    torch::Tensor outputTensor = output.toTensor();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;

    outputTensors.push_back(outputTensor.to(options)); // Pin the output tensor to memory
    inferenceTimings.push_back(duration.count());
  }

  // Access the pinned output tensors
  for (const auto& outputTensor : outputTensors) {
    // Access the pinned output tensor for further processing
    // e.g., print the tensor shape
    std::cout << "Output tensor shape: " << outputTensor.sizes() << std::endl;
  }

  std::ofstream timingsFile("inference_timings.txt");
  if (timingsFile.is_open()) {
    for (const auto& timing : inferenceTimings) {
      timingsFile << timing << std::endl;
    }
    timingsFile.close();
    std::cout << "Inference timings stored in 'inference_timings.txt'." << std::endl;
  } else {
    std::cerr << "Unable to open file for writing." << std::endl;
  }

  return 0;
}
