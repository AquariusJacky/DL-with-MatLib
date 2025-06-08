#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <cmath>
#include <string>
#include <vector>

#include "MatLib/Matrix.h"

template <typename Matrix>
class Optimizer {
 private:
  float learning_rate;
  float beta1;
  float beta2;
  float epsilon;
  Matrix m;  // First moment
  Matrix v;  // Second moment
  size_t t;  // Time step

  bool initialized;

  enum OptimizerType { ADAM };
  OptimizerType type;

 public:
  Optimizer(float lr = 0.001, float b1 = 0.9, float b2 = 0.999,
            float eps = 1e-8);

  void initialize(const Matrix& weights);
  void setOptimizer(std::string optimizer);
  void update(Matrix& weights, const Matrix& gradients);

 private:
  void AdamUpdate(Matrix& weights, const Matrix& gradients);
};

#include "Optimizer.inl"
#endif