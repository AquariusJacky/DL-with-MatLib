#ifndef MODELBUILDER_INL
#define MODELBUILDER_INL

#include <string>
#include <vector>

#include "Layer.h"
#include "LossFunction.h"
#include "MatLib/Matrix.h"
#include "ModelBuilder.h"
#include "Optimizer.h"

template <typename Matrix>
ModelBuilder<Matrix>::ModelBuilder() {
  train_size = 0;
  test_size = 0;
  train_print_every = 0;
  test_print_every = 0;
}

template <typename Matrix>
ModelBuilder<Matrix>::~ModelBuilder() {
  for (size_t i = 0; i < layers.size(); i++) {
    delete layers[i];
  }
}

template <typename Matrix>
void ModelBuilder<Matrix>::addLayer(Layer<Matrix>* layer_ptr) {
  layer_ptr->setOptimizer(optimizer);
  layers.push_back(layer_ptr);
}

template <typename Matrix>
Matrix ModelBuilder<Matrix>::predict(const Matrix& input) {
  std::vector<Matrix> matrixVec(1, input);
  for (auto& layer : layers) {
    matrixVec = layer->forward(matrixVec);
  }
  return matrixVec[0];
}

template <typename Matrix>
void ModelBuilder<Matrix>::train(const std::vector<Matrix>& inputs,
                                 const std::vector<Matrix>& labels) {
  size_t layer_num = layers.size();
  train_size = inputs.size();
  if (train_print_every == 0) {
    train_print_every = train_size;
  }

  float sum_loss = 0;
  float sum_accuracy = 0;

  for (size_t input_id = 0; input_id < train_size; input_id++) {
    // Forward pass
    std::vector<Matrix> matrixVec(1, inputs[input_id]);
    for (size_t i = 0; i < layer_num; i++) {
      matrixVec = layers[i]->forward(matrixVec);
    }

    // Calculate loss
    float loss = lossFunction.calculateLoss(matrixVec[0], labels[input_id]);
    sum_loss += loss;

    // Calculate accuracy
    int predict_idx = 0, golden_idx = 0;
    for (size_t i = 0; i < labels[input_id].size(); i++) {
      float predict_max = matrixVec[0](0, 0);
      if (matrixVec[0](0, i) > predict_max) {
        predict_max = matrixVec[0](0, i);
        predict_idx = i;
      }
      if (labels[input_id](0, i) == 1) golden_idx = i;
    }
    if (predict_idx == golden_idx) sum_accuracy++;

    // Print loss and accuracy
    if (input_id % train_print_every == 0) {
      std::cout << "Image: " << input_id << std::endl;
      std::cout << "Current Average Loss = "
                << (sum_loss / (input_id + 1)) * 100 << "%" << std::endl;
      std::cout << "Current Accuracy = "
                << (sum_accuracy / (input_id + 1)) * 100 << "%" << std::endl;
      std::cout << std::endl;
    }

    Matrix gradient =
        lossFunction.calculateGradient(matrixVec[0], labels[input_id]);

    // Backward pass (backpropagation)
    std::vector<Matrix> gradientVec(1, gradient);
    for (int i = layer_num - 1; i >= 0; i--) {
      gradientVec = layers[i]->backward(gradientVec);
    }
  }

  loss_avg = sum_loss / train_size;
  std::cout << "Average Loss = " << loss_avg * 100 << "%" << std::endl;
  accuracy = sum_accuracy / train_size;
  std::cout << "Accuracy = " << accuracy * 100 << "%" << std::endl;
}

template <typename Matrix>
void ModelBuilder<Matrix>::test(const std::vector<Matrix>& inputs,
                                const std::vector<Matrix>& labels) {
  size_t layer_num = layers.size();
  size_t test_size = inputs.size();
  if (test_print_every == 0) {
    test_print_every = test_size;
  }

  float sum_loss = 0;
  float sum_accuracy = 0;
  for (size_t input_id = 0; input_id < test_size; input_id++) {
    Matrix result = predict(inputs[input_id]);

    float loss = lossFunction.calculateLoss(result, labels[input_id]);
    sum_loss += loss;

    int predict_idx = 0, golden_idx = 0;
    for (size_t i = 0; i < labels[input_id].size(); i++) {
      float predict_max = result(0, 0);
      if (result(0, i) > predict_max) {
        predict_max = result(0, i);
        predict_idx = i;
      }
      if (labels[input_id](0, i) == 1) golden_idx = i;
    }
    if (predict_idx == golden_idx) sum_accuracy++;

    if (input_id % test_print_every == 0) {
      std::cout << "Image: " << input_id << std::endl;
      std::cout << "Current loss = " << (sum_loss / (input_id + 1))
                << std::endl;
      std::cout << "Current accuracy = " << (sum_accuracy / (input_id + 1))
                << std::endl;
      std::cout << std::endl;
    }
  }
  float test_loss_avg = sum_loss / test_size;
  std::cout << "Testing average loss: " << test_loss_avg * 100 << "%"
            << std::endl;
  float test_accuracy = sum_accuracy / test_size;
  std::cout << "Accuracy = " << test_accuracy * 100 << "%" << std::endl;
}

template <typename Matrix>
void ModelBuilder<Matrix>::setOptimizer(std::string optimizer_name) {
  optimizer.setOptimizer(optimizer_name);
  for (size_t i = 0; i < layers.size(); i++) {
    layers[i]->setOptimizer(optimizer);
  }
}

template <typename Matrix>
void ModelBuilder<Matrix>::setLossFunction(std::string loss_function_name) {
  lossFunction.setLossFunction(loss_function_name);
}

#endif  // MODELBUILDER_INL