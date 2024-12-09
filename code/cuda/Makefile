# Compiler settings
CC = nvcc
CFLAGS = -O2 -arch=sm_53

# Target executable name
TARGET=rotate_cuda

# Source file
SRC=rotate_cuda.cu

# Build target
all: $(TARGET)

# Compile the program
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $^ -o $@

# Clean up
clean:
	rm -f $(TARGET)
.PHONY: all clean



