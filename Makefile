CXX ?= c++
CPPFLAGS += -I.
CXXFLAGS ?= -O3 -std=c++17
BUILD_DIR ?= build

ifeq ($(shell uname -s),Darwin)
LDLIBS += -framework Accelerate
else
OPENMP_FLAGS = -fopenmp
LDLIBS += -fopenmp
endif

.PHONY: all test test-mlx-bpe sanitize clean
all: $(BUILD_DIR)/train

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/train: train.cpp BPE.cpp BPE.h checkpoint.h json.hpp stb_image.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP_FLAGS) train.cpp BPE.cpp -o $@ $(LDLIBS)

$(BUILD_DIR)/regression: tests/regression.cpp train.cpp BPE.cpp BPE.h checkpoint.h json.hpp stb_image.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP_FLAGS) tests/regression.cpp BPE.cpp -o $@ $(LDLIBS)

$(BUILD_DIR)/mlx-bpe-regression: mlx/tests/bpe_tests.cpp mlx/BPE.cpp mlx/BPE.h mlx/data.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -Imlx -pthread mlx/tests/bpe_tests.cpp mlx/BPE.cpp -o $@

test-mlx-bpe: $(BUILD_DIR)/mlx-bpe-regression
	$(BUILD_DIR)/mlx-bpe-regression

test: all $(BUILD_DIR)/regression $(BUILD_DIR)/mlx-bpe-regression
	OMP_NUM_THREADS=2 $(BUILD_DIR)/regression
	OMP_NUM_THREADS=2 python3 tests/smoke.py $(BUILD_DIR)/train
	$(BUILD_DIR)/mlx-bpe-regression

sanitize:
	$(MAKE) test BUILD_DIR=build-sanitize CXXFLAGS='-O1 -g -std=c++17 -fsanitize=address,undefined -fno-omit-frame-pointer'

clean:
	rm -rf $(BUILD_DIR)
