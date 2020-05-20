#include <cuda.h>
#include <cuhash/hash_table.h>

int main() {
  auto table = cuhash::HashTable();
  table.Initialize(10, 2.0);
  const int N = 10;

  // ハッシュテーブルに格納するデータ
  int keys[N] = {1, 6, 4, 9, 0, 3, 7, 2, 5, 8};
  int vals[N] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  // デバイスメモリにコピー
  int *d_keys, *d_vals;
  cudaMalloc((void **)&d_keys, sizeof(int) * N);
  cudaMemcpy(d_keys, keys, sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_vals, sizeof(int) * N);
  cudaMemcpy(d_vals, vals, sizeof(int) * N, cudaMemcpyHostToDevice);

  // ハッシュテーブルにクエリするデータ
  int input[N] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int output[N];

  // デバイスメモリにコピー
  int *d_input, *d_output;
  cudaMalloc((void **)&d_input, sizeof(int) * N);
  cudaMemcpy(d_input, input, sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_output, sizeof(int) * N);
  cudaMemset(d_output, 0, sizeof(int) * N);
  bool s = table.Build(N, (const unsigned int *)d_keys,
                       (const unsigned int *)d_vals);

  std::cout << s << std::endl;
  table.Retrieve(N, (const unsigned int *)d_input, (unsigned int *)d_output);

  std::cout << s << std::endl;
  cudaMemcpy(output, d_output, sizeof(int) * N, cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; ++i) {
    printf("%d\n", output[i]);
  }

  return 0;
}