#ifndef LAYER_FLATEN_INL
#define LAYER_FLATEN_INL

#include "Layer.h"
#include "MatLib/Matrix.h"

template <typename Matrix>
std::vector<Matrix> FlattenLayer<Matrix>::forward(
    const std::vector<Matrix>& input) {
  if (input.size() == 0) {
    throw std::runtime_error("Input is empty");
  }

  // Store input for backward pass
  layerInput = input;
  in_channels = input.size();
  inputSize.m = input[0].m();
  inputSize.n = input[0].n();
  outputSize.m = 1;
  outputSize.n = input[0].size();

  std::vector<Matrix> layerOutput = std::vector<Matrix>(1);

  layerOutput[0] = layerInput[0];
  layerOutput[0].reshape(outputSize);
  for (int channel = 1; channel < in_channels; channel++) {
    layerOutput[0].concatenate(layerInput[channel].copy().reshape(outputSize),
                               1);
  }

  return layerOutput;
}

template <typename Matrix>
std::vector<Matrix> FlattenLayer<Matrix>::backward(
    const std::vector<Matrix>& outputGradient) {
  if (outputGradient.size() != 1) {
    throw std::runtime_error("Output gradient size incorrect");
  }

  inputGradient = std::vector<Matrix>(in_channels);

  for (int channel = 0; channel < in_channels; channel++) {
    inputGradient[channel] = outputGradient[0].copy().cols(
        channel * outputSize.n, (channel + 1) * outputSize.n);
    inputGradient[channel].reshape(inputSize);
  }

  return inputGradient;
}

#endif  // LAYER_FLATEN_INL