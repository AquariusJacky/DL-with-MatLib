#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#include <cmath>
#include <string>
#include <vector>

#include "MatLib/Matrix.h"

template <typename Matrix>
class LossFunction {
 private:
  float epsilon = 1e-15;  // Avoid log(0)

  enum lossType { MSE, CROSSENTROPY };
  lossType type;

 public:
  LossFunction() { type = MSE; }
  ~LossFunction() {}

  void setLossFunction(std::string loss_function);
  float calculateLoss(const Matrix& prediction, const Matrix& target);
  Matrix calculateGradient(const Matrix& prediction, const Matrix& target);
  float crossEntropyLoss(const Matrix& prediction, const Matrix& target);

  // Backward pass: compute gradients
  Matrix crossEntropyGradient(const Matrix& prediction, const Matrix& target);
  float MSELoss(const Matrix& prediction, const Matrix& target);
  Matrix MSEGradient(const Matrix& prediction, const Matrix& target);
};

#include "LossFunction.inl"
#endif