#ifndef LAYER_MAXPOOL_INL
#define LAYER_MAXPOOL_INL

#include "Layer.h"
#include "MatLib/Matrix.h"

template <typename Matrix>
std::vector<Matrix> MaxPoolLayer<Matrix>::forward(
    const std::vector<Matrix>& input) {
  if (input.size() == 0) {
    throw std::runtime_error("Input is empty");
  }

  inputSize = MatrixSize(input[0].m(), input[0].n());
  in_channels = input.size();
  layerOutput = std::vector<Matrix>(in_channels);

  // Store input for backward pass
  layerInput = input;

  // Apply max pooling to the input
  for (size_t channel = 0; channel < in_channels; channel++) {
    layerOutput[channel] = input[channel];
    layerOutput[channel].maxPooling(poolingSize.m);
  }

  outputSize.m = layerOutput[0].m();
  outputSize.n = layerOutput[0].n();

  return layerOutput;
}

template <typename Matrix>
std::vector<Matrix> MaxPoolLayer<Matrix>::backward(
    const std::vector<Matrix>& outputGradient) {
  if (outputGradient.size() != in_channels) {
    throw std::runtime_error("Output gradient size incorrect");
  }
  inputGradient = std::vector<Matrix>(in_channels, Matrix(inputSize));

  // MaxPooling gradient:
  // - If input == output, gradient is 1 (pass through)
  // - If input != 0, gradient is 0 (kill gradient)
  for (size_t channel = 0; channel < in_channels; channel++) {
    for (size_t i = 0; i < outputSize.m; i++) {
      for (size_t j = 0; j < outputSize.n; j++) {
        bool maxFound = false;
        for (size_t k = 0; k < poolingSize.m; k++) {
          for (size_t l = 0; l < poolingSize.n; l++) {
            size_t curr_x = i * poolingSize.m + k;
            size_t curr_y = j * poolingSize.n + l;
            if (!maxFound && layerInput[channel](curr_x, curr_y) ==
                                 layerOutput[channel](i, j)) {
              inputGradient[channel](curr_x, curr_y) =
                  outputGradient[channel](i, j);
              maxFound = true;
            }
          }
        }
      }
    }
  }

  return inputGradient;
}

#endif  // LAYER_MAXPOOL_INL