#include <iostream>
#include <math.h>

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__
// function to add the elements of two array
void add(int n, float *x, float *y)
{
  // get the index of the thread inside the block
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// as well as the number of threads per block
	int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1M elements
  std::cout << "Num elements: " << N << std::endl;

  // allocate unified memory -- accessible from CPU or GPU
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  std::cout << " num blocks: " << numBlocks << std::endl;
  // Run kernel on 1M elements on the CPU
  add<<<numBlocks, blockSize>>>(N, x, y);

  // wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}