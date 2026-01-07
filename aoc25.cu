/*
   An attempt at solving all the AoC'25 problems in parallel.
   Assumes all inputs are in folders like day00/input.txt, day01/input.txt, etc.
*/

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>


/* Utilities */

#define CUDA_CHECK(expr_to_check) do {                      \
  cudaError_t result = expr_to_check;                       \
  if (result != cudaSuccess) {                              \
    fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n",  \
    __FILE__, __LINE__, result,cudaGetErrorString(result)); \
  }                                                         \
} while(0)

struct Result {
  char part1[128];
  char part2[128];
};


/* Day 1 */


__global__ void solve_day01_part1(int *input, size_t t, size_t *output) {
  size_t counter = 0;
  int posn = 50;
  for (size_t i = 0; i < t; i++) {
    posn = (posn + input[i]) % 100;
    if (posn == 0) {
      counter += 1;
    }
  }
  output[0] = counter;
}

__global__ void solve_day01_part2(int *input, size_t t, size_t *output) {
  size_t counter = 0;
  int posn = 50;
  int prev_posn = posn;
  for (size_t i = 0; i < t; i++) {
    prev_posn = posn;
    posn = (posn + input[i]);
    if (posn < 0) {
      counter += (-posn / 100);
      if (prev_posn != 0)
        counter++;
    } else if (posn == 0) {
      counter++;
    } else {
      counter += (posn / 100);
    }

    posn = posn % 100;
    if (posn < 0)
      posn += 100;
  }
  output[1] = counter;
}

struct Result* day01() {
  struct Result *result = NULL;

  FILE *input = fopen("day01/input.txt", "r");
  const size_t BUFFER_SIZE = 128;
  char line[BUFFER_SIZE];

  if (input == NULL) {
    printf("Couldn't read input file for day 1!\n");
    return NULL;
  }

  // Parse numbers into an int array
  size_t tsz = 128, t = 0;
  int *turns = (int*) malloc(sizeof(int) * tsz);
  while (fgets(line, BUFFER_SIZE, input)) {
    int val = atoi(&line[1]);
    if (line[0] == 'R') {
      turns[t++] = val;
    } else if (line[0] == 'L') {
      turns[t++] = -val;
    } else {
      assert(false);
    }

    if (t == tsz) {
      turns = (int*) realloc(turns, tsz * 2 * sizeof(int));
      assert(turns != NULL);
      tsz *= 2;
    }
  }

  int *d_turns;
  size_t *d_output;
  size_t output[2];

  CUDA_CHECK(cudaMalloc((void**)&d_turns, sizeof(int) * t));
  CUDA_CHECK(cudaMalloc((void**) &d_output, sizeof(size_t) * 2));
  CUDA_CHECK(cudaMemcpy(d_turns, turns, t * sizeof(int), cudaMemcpyHostToDevice));
  solve_day01_part1<<<1, 1>>>(d_turns, t, d_output);
  solve_day01_part2<<<1, 1>>>(d_turns, t, d_output);
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaMemcpy(output, d_output, 2 * sizeof(size_t), cudaMemcpyDeviceToHost));

  result = (struct Result*) malloc(sizeof(struct Result));
  sprintf(result->part1, "%lu", output[0]);
  sprintf(result->part2, "%lu", output[1]);

  cudaFree(d_turns);
  free(turns);
  fclose(input);
  return result;
}


int main(int argc, char** argv) {
  // Run in threads?
  struct Result *res = day01();
  printf("Day 1 Part 1 %s\n", res->part1);
  printf("Day 1 Part 2 %s\n", res->part2);
  free(res);


  return 0;
}
