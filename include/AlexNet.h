#ifndef ALEXNET_H
#define ALEXNET_H

#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <typeinfo>

#include "MNISTReader.h"
#include "MatLib/Matrix.h"
#include "ModelBuilder.h"

template <typename Matrix>
class AlexNet {
 private:
  ModelBuilder<Matrix> model;

 public:
  AlexNet() {
    // Initialize common settings
    model.setOptimizer("Adam");
    model.setLossFunction("Cross Entropy");
    model.setPrintEvery(5000, 1000);
  }

  void buildModel() {
    std::cout << "Building AlexNet model..." << std::endl;

    // First conv block
    model.addLayer(
        new ConvLayer<Matrix>(1, 8, MatrixSize(5, 5)));  // 8 * (24 * 24)
    model.addLayer(new ReLULayer<Matrix>());             // 8 * (24 * 24)
    model.addLayer(
        new MaxPoolLayer<Matrix>(MatrixSize(2, 2)));  // 8 * (12 * 12)

    // Second conv block
    model.addLayer(
        new ConvLayer<Matrix>(8, 8, MatrixSize(5, 5)));          // 64 * (8 * 8)
    model.addLayer(new ReLULayer<Matrix>());                     // 64 * (8 * 8)
    model.addLayer(new MaxPoolLayer<Matrix>(MatrixSize(2, 2)));  // 64 * (4 * 4)

    // Flatten
    model.addLayer(new FlattenLayer<Matrix>());  // 1 * 1 * 1024

    // Fully connected layers
    model.addLayer(new LinearLayer<Matrix>(1024, 512));
    model.addLayer(new ReLULayer<Matrix>());
    model.addLayer(new LinearLayer<Matrix>(512, 512));
    model.addLayer(new ReLULayer<Matrix>());
    model.addLayer(new LinearLayer<Matrix>(512, 10));
  }

  void train(const std::vector<Matrix>& train_images,
             const std::vector<Matrix>& train_labels) {
    model.train(train_images, train_labels);
  }

  void test(const std::vector<Matrix>& test_images,
            const std::vector<Matrix>& test_labels) {
    model.test(test_images, test_labels);
  }
};

#endif