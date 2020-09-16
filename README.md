# Cache-Benchmark-Light
Tests the cache on Jetson boards: both the CPU and the GPU cycle through the input array to perform mathematical operations, the CPU fills the array as fast as possible with random data which is computed by the GPU after a single read. This benchmark should not use cache on the GPU side, showing the same performance on Zero Copy and Unified Memory
## Usage
Simply compile by running the compile script (currently configured for Volta/Xavier) on a Jetson Board:
```
./compile
```
and then run with
```
./fb
```
## Configuration
To modify the configuration change the variables contained at the start of "AS.cu":
```
// Set PRINT to 1 for debug output
#define PRINT 0
#define FROM_debug 0
#define TO_debug 16

// Set ZERO to 1 to use Zero copy, set ZERO to 0 to use Unified Memory
#define ZERO 1

// N is later overwritten as N = N^POW, making N the size of the input array
unsigned int N = 2;
const int POW = 16;

const float MINUTES = 0.1; // Dictates the length of the benchmark, but doesn't actually follow the length 

const int BLOCK_SIZE_X = 512;
const int BLOCK_SIZE_Y = 1;
```
# Result
As expected, in this scenario, both CPU and GPU behave the same under Zero Copy and Unified Memory.
