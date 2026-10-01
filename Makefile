# Simplest possible Makefile for Tremor.
# Requires an MPI C++ wrapper (mpicxx, e.g. Intel MPI or Open MPI) in PATH and
# FFTW 3: make FFTW=/path/to/fftw

CXX      := mpicxx
CXXFLAGS ?= -O2 -std=c++17 -fopenmp -march=native
FFTW     ?= /path/to/fftw
CXXFLAGS += -I$(FFTW)/include
LDLIBS   += -fopenmp -lpthread -lm -L$(FFTW)/lib -Wl,-rpath,$(FFTW)/lib -lfftw3

BIN_DIR  := bin

SRCS := tremor_main.cpp \
        $(wildcard tremor/isax/*.cpp) \
        $(wildcard tremor/distance_computers/*.cpp) \
        $(wildcard tremor/utils/*.cpp) \
        $(wildcard tremor/algos/tremor/*.cpp) \
        $(wildcard tremor/algos/sss/*.cpp)

OBJS := $(patsubst %.cpp,$(BIN_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

TARGET := $(BIN_DIR)/tremor_main

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

-include $(DEPS)

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
