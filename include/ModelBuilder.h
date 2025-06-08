#ifndef MODELBUILDER_H
#define MODELBUILDER_H

#include <string>
#include <vector>

#include "Layer.h"
#include "LossFunction.h"
#include "MatLib/Matrix.h"
#include "Optimizer.h"

// ModelBuilder Class to Manage Layers
template <typename Matrix>
class ModelBuilder {
 private:
  std::vector<Layer<Matrix>*> layers;
  Optimizer<Matrix> optimizer;
  LossFunction<Matrix> lossFunction;

  size_t train_size;
  size_t test_size;

  size_t train_print_every;
  size_t test_print_every;

  float loss_avg;
  float accuracy;

 public:
  ModelBuilder();
  ~ModelBuilder();

  void addLayer(Layer<Matrix>* layer_ptr);

  Matrix predict(const Matrix& input);

  void train(const std::vector<Matrix>& inputs,
             const std::vector<Matrix>& labels);

  void test(const std::vector<Matrix>& inputs,
            const std::vector<Matrix>& labels);

  void setOptimizer(std::string optimizer_name);
  void setLossFunction(std::string loss_function_name);

  void setTrainPrintEvery(const size_t& num) { train_print_every = num; }
  void setTestPrintEvery(const size_t& num) { test_print_every = num; }
  void setPrintEvery(const size_t& train, const size_t& test) {
    setTrainPrintEvery(train);
    setTestPrintEvery(test);
  }
};

#include "ModelBuilder.inl"
#endif