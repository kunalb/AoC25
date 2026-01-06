#include <stdio.h>


__global__ void kernel_hello() {
  printf("Hello from the GPU!\n");
}


int main(int argc, char **argv) {
  kernel_hello<<<1, 1>>>();
  cudaDeviceSynchronize();
  printf("Hello, World!\n");
  return 0;
}
