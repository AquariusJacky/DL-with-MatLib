#ifndef LAYER_LINEAR_INL
#define LAYER_LINEAR_INL

#include "Layer.h"
#include "MatLib/Matrix.h"

template <typename Matrix>
LinearLayer<Matrix>::LinearLayer(const MatrixSize& inputSize,
                                 const MatrixSize& outputSize)
    : inputSize(inputSize), outputSize(outputSize) {
  // Initialize weight and biases with random values
  init();
}

template <typename Matrix>
LinearLayer<Matrix>::LinearLayer(const size_t& inputSize,
                                 const size_t& outputSize)
    : inputSize(MatrixSize(1, inputSize)),
      outputSize(MatrixSize(1, outputSize)) {
  // Initialize weight and biases with random values
  init();
}

template <typename Matrix>
void LinearLayer<Matrix>::init() {
  if (inputSize.m != 1) {
    throw std::runtime_error("Linear layer: Input must have size m x 1");
  }
  float rand_limit = 0.5;

  weights = std::vector<Matrix>(1, Matrix(inputSize.n, outputSize.n));
  bias = std::vector<Matrix>(1, Matrix(outputSize.m, outputSize.n));

  weights[0].rand(rand_limit * -1, rand_limit);
  bias[0].rand(rand_limit * -1, rand_limit);
}

template <typename Matrix>
std::vector<Matrix> LinearLayer<Matrix>::forward(
    const std::vector<Matrix>& input) {
  if (input.size() != 1) {
    throw std::runtime_error("Dense forward: Input must only contain 1 Matrix");
  }
  if (input[0].m() != 1 || input[0].n() != inputSize.n) {
    throw std::runtime_error("Input size incorrect for LinearLayer forward");
  }

  // Store input for backward pass
  layerInput = input;

  // Apply each weight to the input
  std::vector<Matrix> layerOutput(1);
  layerOutput[0] = input[0];
  layerOutput[0].dot(weights[0]);
  layerOutput[0] += bias[0];

  return layerOutput;
}

template <typename Matrix>
std::vector<Matrix> LinearLayer<Matrix>::backward(
    const std::vector<Matrix>& outputGradient) {
  if (outputGradient.size() != 1) {
    throw std::runtime_error("Output gradient vector size incorrect");
  }

  weightGradient = std::vector<Matrix>(1);
  biasGradient = std::vector<Matrix>(1);
  inputGradient = std::vector<Matrix>(1);

  weightGradient[0] = layerInput[0].T();
  biasGradient[0] = outputGradient[0];
  inputGradient[0] = outputGradient[0];

  // Weight gradient = dot(input^T, outputGradient)
  weightGradient[0].dot(outputGradient[0]);

  // Bias gradient = outputGradient
  // Do nothing

  // This is for previous layer
  // Input gradient = dot(weight^T, outputGradient)
  inputGradient[0].dot(weights[0].copy().T());

  // Update weight and bias
  this->optimizer.update(weights[0], weightGradient[0]);
  this->optimizer.update(bias[0], biasGradient[0]);

  return inputGradient;
}

#endif  // LAYER_LINEAR_INL