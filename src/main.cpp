#include <iostream>
#include <vector>

#include "AlexNet.h"
#include "Layer.h"
#include "MNISTReader.h"
#include "MatLib/Matrix.h"

int main() {
  auto dataset = readMNIST("/home/aquariusj/Download/", 0);
  std::cout << "Finished loading data..." << std::endl;

  std::vector<CPU::Matrix> cpu_train_images(dataset.train_images.size());
  std::vector<CPU::Matrix> cpu_train_labels(dataset.train_images.size());
  std::vector<GPU::Matrix> gpu_train_images(dataset.train_images.size());
  std::vector<GPU::Matrix> gpu_train_labels(dataset.train_images.size());
  for (size_t i = 0; i < dataset.train_images.size(); i++) {
    cpu_train_images[i] = CPU::Matrix(dataset.train_images[i]);
    cpu_train_labels[i] = CPU::Matrix(dataset.train_labels[i]);
    gpu_train_images[i] = GPU::Matrix(cpu_train_images[i]);
    gpu_train_labels[i] = GPU::Matrix(cpu_train_labels[i]);
  }

  std::vector<CPU::Matrix> cpu_test_images(dataset.test_images.size());
  std::vector<CPU::Matrix> cpu_test_labels(dataset.test_images.size());
  std::vector<GPU::Matrix> gpu_test_images(dataset.test_images.size());
  std::vector<GPU::Matrix> gpu_test_labels(dataset.test_images.size());
  for (size_t i = 0; i < dataset.test_images.size(); i++) {
    cpu_test_images[i] = CPU::Matrix(dataset.test_images[i]);
    cpu_test_labels[i] = CPU::Matrix(dataset.test_labels[i]);
    gpu_test_images[i] = GPU::Matrix(cpu_test_images[i]);
    gpu_test_labels[i] = GPU::Matrix(cpu_test_labels[i]);
  }

  auto CPUAlexNet = AlexNet<CPU::Matrix>();
  auto GPUAlexNet = AlexNet<GPU::Matrix>();

  CPUAlexNet.buildModel();
  GPUAlexNet.buildModel();

  std::cout << "Training CPU AlexNet..." << std::endl;
  CPUAlexNet.train(cpu_train_images, cpu_train_labels);
  std::cout << "Testing CPU AlexNet..." << std::endl;
  CPUAlexNet.test(cpu_test_images, cpu_test_labels);

  std::cout << "Training GPU AlexNet..." << std::endl;
  GPUAlexNet.train(gpu_train_images, gpu_train_labels);
  std::cout << "Testing GPU AlexNet..." << std::endl;
  GPUAlexNet.test(gpu_test_images, gpu_test_labels);

  std::cout << "Training and testing completed." << std::endl;

  return 0;
}