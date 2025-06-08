#ifndef OPTIMIZER_INL
#define OPTIMIZER_INL

template <typename Matrix>
Optimizer<Matrix>::Optimizer(float lr, float b1, float b2, float eps)
    : learning_rate(lr),
      beta1(b1),
      beta2(b2),
      epsilon(eps),
      t(0),
      initialized(false),
      type(OptimizerType::ADAM) {}

template <typename Matrix>
void Optimizer<Matrix>::initialize(const Matrix& weights) {
  m = Matrix(weights.m(), weights.n());
  v = Matrix(weights.m(), weights.n());
  initialized = true;
}

template <typename Matrix>
void Optimizer<Matrix>::setOptimizer(std::string optimizer) {
  if (optimizer == "Adam") type = ADAM;
}

template <typename Matrix>
void Optimizer<Matrix>::update(Matrix& weights, const Matrix& gradients) {
  switch (type) {
    case ADAM:
    default:
      AdamUpdate(weights, gradients);
      break;
  }
}

template <typename Matrix>
void Optimizer<Matrix>::AdamUpdate(Matrix& weights, const Matrix& gradients) {
  t++;

  // If not initialized, initialize moment vectors
  if (!initialized) {
    initialize(weights);
  }

  for (size_t i = 0; i < weights.m(); i++) {
    for (size_t j = 0; j < weights.n(); j++) {
      // Update biased first moment estimate
      m(i, j) = beta1 * m(i, j) + (1 - beta1) * gradients(i, j);

      // Update biased second raw moment estimate
      v(i, j) =
          beta2 * v(i, j) + (1 - beta2) * gradients(i, j) * gradients(i, j);

      // Compute bias-corrected first moment estimate
      float m_hat = m(i, j) / (1 - std::pow(beta1, t));

      // Compute bias-corrected second raw moment estimate
      float v_hat = v(i, j) / (1 - std::pow(beta2, t));

      // Update parameters
      weights(i, j) -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
  }
}

#endif  // OPTIMIZER_INL