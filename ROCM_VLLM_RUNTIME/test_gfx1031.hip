#include <hip/hip_runtime.h> 
__global__ void test_gfx1031(float* a){ a[0]=1; } 
