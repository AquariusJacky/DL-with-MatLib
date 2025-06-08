#include "Layer.h"
#include "MatLib/Matrix.h"

template <typename Matrix>
std::vector<Matrix> ReLULayer<Matrix>::forward(
    const std::vector<Matrix>& input) {
  if (input.size() == 0) {
    throw std::runtime_error("Input is empty");
  }
  std::vector<Matrix> layerOutput(input.size());

  // Store input for backward pass
  layerInput = input;
  in_channels = input.size();

  for (int channel = 0; channel < in_channels; channel++) {
    layerOutput[channel] = layerInput[channel];
    for (int i = 0; i < layerInput[channel].m(); ++i) {
      for (int j = 0; j < layerInput[channel].n(); ++j) {
        if (layerOutput[channel](i, j) < 0) {
          layerOutput[channel](i, j) = 0;
        }
      }
    }
  }

  return layerOutput;
}

template <typename Matrix>
std::vector<Matrix> ReLULayer<Matrix>::backward(
    const std::vector<Matrix>& outputGradient) {
  if (outputGradient.size() != in_channels) {
    throw std::runtime_error("Output gradient size incorrect");
  }

  inputGradient = std::vector<Matrix>(in_channels);

  // ReLU gradient:
  // - If input > 0, gradient is 1 (pass through)
  // - If input <= 0, gradient is 0 (kill gradient)
  for (int channel = 0; channel < in_channels; channel++) {
    inputGradient[channel] = outputGradient[channel];
    for (int i = 0; i < layerInput[channel].m(); ++i) {
      for (int j = 0; j < layerInput[channel].n(); ++j) {
        if (layerInput[channel](i, j) < 0) {
          inputGradient[channel](i, j) = 0;
        }
      }
    }
  }

  return inputGradient;
}