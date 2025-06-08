#ifndef LAYER_CONV_INL
#define LAYER_CONV_INL

#include "Layer.h"
#include "MatLib/Matrix.h"

template <typename Matrix>
ConvLayer<Matrix>::ConvLayer(const size_t& in_channels, const size_t& numWeight,
                             const MatrixSize& weightSize, size_t stride)
    : in_channels(in_channels),
      numWeight(numWeight),
      weightSize(weightSize),
      stride(stride) {
  // Initialize weights and biases with random values
  init();
}

template <typename Matrix>
void ConvLayer<Matrix>::init() {
  float rand_limit = 0.5;

  // Initialize weights with random values
  for (size_t i = 0; i < numWeight; ++i) {
    weights.push_back(Matrix(weightSize.m, weightSize.n));
    weights[i].rand(rand_limit * -1, rand_limit);
  }
}

template <typename Matrix>
std::vector<Matrix> ConvLayer<Matrix>::forward(
    const std::vector<Matrix>& input) {
  if (input.size() == 0) {
    throw std::runtime_error("Input is empty");
  }

  std::vector<Matrix> layerOutput(in_channels * numWeight);

  // Store input for backward pass
  layerInput = input;
  inputSize = MatrixSize(input[0].m(), input[0].n());

  // Apply each weight to the input
  for (size_t channel = 0; channel < in_channels; channel++) {
    for (size_t weight_num = 0; weight_num < numWeight; weight_num++) {
      size_t curr_output = in_channels * channel + weight_num;
      layerOutput[curr_output] = input[channel];
      layerOutput[curr_output].convolution(weights[weight_num], stride);
    }
  }

  return layerOutput;
}

template <typename Matrix>
std::vector<Matrix> ConvLayer<Matrix>::backward(
    const std::vector<Matrix>& outputGradient) {
  if (outputGradient.size() != (in_channels * numWeight)) {
    throw std::runtime_error("Output gradient size incorrect");
  }

  inputGradient = std::vector<Matrix>(in_channels, inputSize);
  weightGradient = std::vector<Matrix>(numWeight, weightSize);

  for (size_t channel = 0; channel < in_channels; channel++) {
    for (size_t weight_num = 0; weight_num < numWeight; weight_num++) {
      size_t curr_output = channel * in_channels + weight_num;
      Matrix curr_weightGradient = layerInput[channel];
      Matrix curr_inputGradient = outputGradient[curr_output];

      // Weight gradient = Conv(input, outputGradient)
      curr_weightGradient.convolution(outputGradient[curr_output], stride);

      // This is for previous layer
      // Input gradient = Conv(weight rotate 180, outputGradient)
      curr_inputGradient.convolution(weights[weight_num].copy().rotate90(2),
                                     stride);

      weightGradient[weight_num] += curr_weightGradient;
      inputGradient[channel] += curr_inputGradient;
    }
  }

  // Update weights
  for (size_t weight_num = 0; weight_num < numWeight; weight_num++) {
    this->optimizer.update(weights[weight_num], weightGradient[weight_num]);
  }

  return inputGradient;
}

#endif  // LAYER_CONV_INL