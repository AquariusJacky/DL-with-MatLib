#ifndef LOSSFUNCTION_INL
#define LOSSFUNCTION_INL

template <typename Matrix>
void LossFunction<Matrix>::setLossFunction(std::string loss_function) {
  if (loss_function == "Mean Square Error" || loss_function == "MSE") {
    type = lossType::MSE;
  } else if (loss_function == "Cross Entropy") {
    type = lossType::CROSSENTROPY;
  }
}

template <typename Matrix>
float LossFunction<Matrix>::calculateLoss(const Matrix& prediction,
                                          const Matrix& target) {
  switch (type) {
    case lossType::CROSSENTROPY:
      return crossEntropyLoss(prediction, target);
    case lossType::MSE:
    default:
      return MSELoss(prediction, target);
  }
  return 0;
}

template <typename Matrix>
Matrix LossFunction<Matrix>::calculateGradient(const Matrix& prediction,
                                               const Matrix& target) {
  switch (type) {
    case CROSSENTROPY:
      return crossEntropyGradient(prediction, target);
    case MSE:
    default:
      return MSEGradient(prediction, target);
      break;
  }
}

template <typename Matrix>
float LossFunction<Matrix>::crossEntropyLoss(const Matrix& prediction,
                                             const Matrix& target) {
  float loss = 0.0;
  size_t batch_size = prediction.size();
  for (size_t j = 0; j < prediction.n(); j++) {
    // Clip predictions to avoid numerical instability
    float pred = std::max(std::min(prediction(0, j), 1 - epsilon), epsilon);
    loss -= target(0, j) * std::log(pred);
  }

  return loss / batch_size;
}

// Backward pass: compute gradients
template <typename Matrix>
Matrix LossFunction<Matrix>::crossEntropyGradient(const Matrix& prediction,
                                                  const Matrix& target) {
  size_t batch_size = prediction.n();
  Matrix gradient(prediction.m(), prediction.n());

  for (size_t j = 0; j < prediction.n(); j++) {
    float pred = std::max(std::min(prediction(0, j), 1 - epsilon), epsilon);
    gradient(0, j) = (-target(0, j) / pred) / batch_size;
  }

  return gradient;
}

template <typename Matrix>
float LossFunction<Matrix>::MSELoss(const Matrix& prediction,
                                    const Matrix& target) {
  size_t n = prediction.size();

  float sum_squared_error = 0.0;

  // MSE = (1/n) * Σ(y_pred - y_true)²
  for (size_t j = 0; j < n; j++) {
    float error = prediction(1, j) - target(1, j);
    sum_squared_error += error * error;
  }

  return sum_squared_error / n;
}

template <typename Matrix>
Matrix LossFunction<Matrix>::MSEGradient(const Matrix& prediction,
                                         const Matrix& target) {
  Matrix gradient(prediction.size());
  size_t n = prediction.size();

  // Gradient of MSE = (2/n) * (y_pred - y_true)
  for (size_t j = 0; j < n; j++) {
    gradient(1, j) = (2.0 / n) * (prediction(1, j) - target(1, j));
  }

  return gradient;
}

#endif  // LOSSFUNCTION_INL