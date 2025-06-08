#ifndef LAYER_H
#define LAYER_H

#include <vector>

#include "MatLib/Matrix.h"
#include "Optimizer.h"

// Abstract base class for all layer types
template <typename Matrix>
class Layer {
 protected:
  Optimizer<Matrix> optimizer;

 public:
  virtual ~Layer() = default;

  void setOptimizer(Optimizer<Matrix> optimizer_) { optimizer = optimizer_; };

  virtual std::vector<Matrix> forward(const std::vector<Matrix>& input) = 0;
  virtual std::vector<Matrix> backward(
      const std::vector<Matrix>& gradOutput) = 0;
};

template <typename Matrix>
class FlattenLayer : public Layer<Matrix> {
 private:
  size_t in_channels;
  MatrixSize inputSize;
  MatrixSize outputSize;

  std::vector<Matrix> layerInput;
  std::vector<Matrix> inputGradient;

 public:
  FlattenLayer() {}

  std::vector<Matrix> forward(const std::vector<Matrix>& input) override;
  std::vector<Matrix> backward(const std::vector<Matrix>& gradOutput) override;
};

template <typename Matrix>
class ReLULayer : public Layer<Matrix> {
 private:
  size_t in_channels;
  MatrixSize inputSize;

  std::vector<Matrix> layerInput;

  std::vector<Matrix> inputGradient;

 public:
  ReLULayer() {}

  std::vector<Matrix> forward(const std::vector<Matrix>& input) override;
  std::vector<Matrix> backward(const std::vector<Matrix>& gradOutput) override;
};

template <typename Matrix>
class MaxPoolLayer : public Layer<Matrix> {
 private:
  size_t in_channels;
  size_t numWeight;
  MatrixSize inputSize;
  MatrixSize poolingSize;
  MatrixSize outputSize;

  std::vector<Matrix> layerInput;
  std::vector<Matrix> layerOutput;

  std::vector<Matrix> inputGradient;

 public:
  MaxPoolLayer(const MatrixSize& poolingSize_) : poolingSize(poolingSize_) {}

  std::vector<Matrix> forward(const std::vector<Matrix>& input) override;
  std::vector<Matrix> backward(const std::vector<Matrix>& gradOutput) override;

 private:
  void init();
};

template <typename Matrix>
class ConvLayer : public Layer<Matrix> {
 private:
  size_t in_channels;
  size_t numWeight;
  MatrixSize inputSize;
  MatrixSize weightSize;
  size_t stride;

  // Weights (weights)
  std::vector<Matrix> weights;
  std::vector<Matrix> layerInput;

  std::vector<Matrix> weightGradient;
  std::vector<Matrix> inputGradient;

 public:
  ConvLayer(const size_t& in_channels, const size_t& num_weight,
            const MatrixSize& weightSize, size_t stride = 1);

  std::vector<Matrix> forward(const std::vector<Matrix>& input) override;
  std::vector<Matrix> backward(const std::vector<Matrix>& gradOutput) override;

 private:
  void init();
};

// Fully connected layer (FCN)
template <typename Matrix>
class LinearLayer : public Layer<Matrix> {
 private:
  MatrixSize inputSize;
  MatrixSize outputSize;

  // Weights (weights)
  std::vector<Matrix> weights;
  std::vector<Matrix> bias;
  std::vector<Matrix> layerInput;

  std::vector<Matrix> weightGradient;
  std::vector<Matrix> biasGradient;
  std::vector<Matrix> inputGradient;

 public:
  LinearLayer(const MatrixSize& inputSize, const MatrixSize& outputSize);
  LinearLayer(const size_t& inputSize, const size_t& outputSize);

  std::vector<Matrix> forward(const std::vector<Matrix>& input) override;
  std::vector<Matrix> backward(const std::vector<Matrix>& gradOutput) override;

 private:
  void init();
};

#include "Layer_Conv.inl"
#include "Layer_Flatten.inl"
#include "Layer_Linear.inl"
#include "Layer_MaxPool.inl"
#include "Layer_ReLU.inl"
#endif