# Compiler settings - change these if necessary
MPICC=mpicc

# Target executable name
TARGET=rotate_mpi

# Source files
SRC=rotate_mpi.c

# Compiler flags
CFLAGS=-Wall

# Libraries
LIBS=-lm -lmpiP -lbfd -ldl

# mpiP library path
MPIPLIBPATH=/home/nvidia/Desktop/mpiP-3.5/mpiP-3.5/lib

# LDFLAGS for linking, include mpiP library path
LDFLAGS=-L$(MPIPLIBPATH)

# Rule to build the target
$(TARGET): $(SRC)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o $(TARGET) $(SRC) $(LIBS)

# Rule for cleaning the project
clean:
	rm -f $(TARGET)

# Phony targets for clean
.PHONY: clean




