# Compiler settings
CXX := g++
NVCC := nvcc

# Change these paths to match your Matrix installation
MATLIB_PATH := /home/aquariusj/Projects/install
CUDA_PATH := /usr/local/cuda

# Directory structure
SRC_DIR := src
INC_DIR := include
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj

# Compiler flags
CXX_FLAGS := -O3 -std=c++11 -Wall
INCLUDES := -I$(INC_DIR) -I$(MATLIB_PATH)/include/

# Library paths
LIB_PATHS := -L$(MATLIB_PATH)/lib -L$(CUDA_PATH)/lib64
LIBS := -lmatlib -lcudart

# Source files and objects
SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS := $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
TARGET := myprogram

all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR) $(OBJ_DIR):
	@mkdir -p $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/$(TARGET): $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) $^ -o $@ $(LIB_PATHS) $(LIBS)

run: $(BUILD_DIR)/$(TARGET)
	@LD_LIBRARY_PATH=$(MATLIB_PATH)/build/release/lib:$(CUDA_PATH)/lib64 ./$(BUILD_DIR)/$(TARGET)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run clean