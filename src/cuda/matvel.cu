#include <math.h>

__global__
void* mult(int n, float* A, float* b, float* x) {
  for (int i = 0; i < n; i++){
    b[i] += A[i]*x[i];
  }
  return;
}
