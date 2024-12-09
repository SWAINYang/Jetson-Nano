# Performance Optimization of Parallel Computing Algorithm

This repository is focused on optimizing the performance of a parallel computing algorithm by analyzing the runtime of key functions using the **gprof** profiler tool.

## Overview
The project identifies performance bottlenecks in a computational program and suggests optimizations to improve overall runtime. The primary bottleneck was found to be the `rotatePGM` function, which consumes over **77%** of the total execution time.

## Key Findings
- **rotatePGM**: The main bottleneck function, consuming **77.68%** of the total runtime.
- **readPGM** and **writePGM**: These functions each account for about **10%** of the execution time.
- The **main** function calls `rotatePGM`, `readPGM`, and `writePGM`.

## Approach
The analysis was carried out using **gprof**, a performance profiling tool. The runtime of various functions was measured, and reports were generated to show the relationships between function calls and time consumption.

## Optimizations Suggested
- Focus on optimizing the `rotatePGM` function to reduce its time consumption and enhance performance.

## Installation

To get started, clone this repository and follow the instructions below.

```bash
git clone https://github.com/yourusername/repository.git
cd repository
```

### Prerequisites
Ensure that you have the necessary tools installed:

- gprof (GNU profiler)

### Running the Profiler
Compile your program with profiling enabled:

```bash
gcc -pg -o program program.c

```
Run the program:


```bash
./program
```

Generate the profiling report:
```bash
gprof ./program gmon.out > report.txt
```
