#ifndef MNISTREADER_H
#define MNISTREADER_H

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define HEIGHT 28
#define WIDTH 28
#define LABEL 10
#define NUMTRAIN 60000
#define NUMTEST 10000

/**
 * @brief Each individual image is stored as a 2D vector.
 * @brief A list of images is also stored in vector.
 * @brief Thus 3D vector for all images.
 */
struct Dataset {
  // Vector of 2D matrices
  std::vector<std::vector<std::vector<float>>> train_images;
  std::vector<std::vector<std::vector<float>>> test_images;

  // Vector of 1D matrix
  std::vector<std::vector<float>> train_labels;
  std::vector<std::vector<float>> test_labels;

  size_t train_size = NUMTRAIN;
  size_t test_size = NUMTEST;

  Dataset() {
    train_images = std::vector<std::vector<std::vector<float>>>(
        NUMTRAIN,
        std::vector<std::vector<float>>(HEIGHT, std::vector<float>(WIDTH)));
    test_images = std::vector<std::vector<std::vector<float>>>(
        NUMTEST,
        std::vector<std::vector<float>>(HEIGHT, std::vector<float>(WIDTH)));

    train_labels =
        std::vector<std::vector<float>>(NUMTRAIN, std::vector<float>(LABEL, 0));
    test_labels =
        std::vector<std::vector<float>>(NUMTEST, std::vector<float>(LABEL, 0));
  }
};

Dataset readMNIST(std::string mnist_dir_path, bool print) {
  std::string train_images_path = mnist_dir_path + "train-images.idx3-ubyte";
  std::string train_labels_path = mnist_dir_path + "train-labels.idx1-ubyte";
  std::string test_images_path = mnist_dir_path + "t10k-images.idx3-ubyte";
  std::string test_labels_path = mnist_dir_path + "t10k-labels.idx1-ubyte";

  std::ifstream train_images_ifs, train_labels_ifs;
  std::ifstream test_images_ifs, test_labels_ifs;

  train_images_ifs.open(train_images_path.c_str(),
                        std::ios::in | std::ios::binary);  // Binary image file
  train_labels_ifs.open(train_labels_path.c_str(),
                        std::ios::in | std::ios::binary);  // Binary label file
  test_images_ifs.open(test_images_path.c_str(),
                       std::ios::in | std::ios::binary);  // Binary image file
  test_labels_ifs.open(test_labels_path.c_str(),
                       std::ios::in | std::ios::binary);  // Binary label file

  char number;
  for (size_t i = 0; i < 16; i++) {
    train_images_ifs.read(&number, sizeof(char));
    test_images_ifs.read(&number, sizeof(char));
  }
  for (size_t i = 0; i < 8; i++) {
    train_labels_ifs.read(&number, sizeof(char));
    test_labels_ifs.read(&number, sizeof(char));
  }

  Dataset dataset;

  // Reading training images and labels
  for (size_t sample = 0; sample < NUMTRAIN; ++sample) {
    if (print && sample % 10000 == 0)
      std::cout << "Sample: " << sample << std::endl;
    for (size_t i = 0; i < HEIGHT; i++) {
      for (size_t j = 0; j < WIDTH; j++) {
        train_images_ifs.read(&number, sizeof(char));
        dataset.train_images[sample][i][j] = ((float)number) / 255.0;

        if (print && sample % 10000 == 0) {
          std::cout << dataset.train_images[sample][i][j] << " ";
        }
      }
      if (print && sample % 10000 == 0) {
        std::cout << std::endl;
      }
    }

    train_labels_ifs.read(&number, sizeof(char));
    dataset.train_labels[sample][(float)number] = 1;
    if (print && sample % 10000 == 0) {
      std::cout << "Label: " << (float)number << std::endl;
    }
  }

  // Reading testing images and labels
  for (size_t sample = 0; sample < NUMTEST; ++sample) {
    for (size_t i = 0; i < HEIGHT; i++) {
      for (size_t j = 0; j < WIDTH; j++) {
        test_images_ifs.read(&number, sizeof(char));
        dataset.test_images[sample][i][j] = ((float)number) / 255.0;
      }
    }
    test_labels_ifs.read(&number, sizeof(char));
    dataset.test_labels[sample][number] = 1;
  }

  return dataset;
}

#endif