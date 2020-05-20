// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision:$
// $Date:$
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file hash_table.cu
 *
 * @brief Hides all of the CUDA calls from the actual CPP file.
 */

#include <cuhash/cuda_util.h>
#include <cuhash/debugging.h>
#include <cuhash/definitions.h>
#include <cuhash/hash_table.cuh>

#include <cuda.h>

namespace cuhash {

namespace CUDAWrapper {
void ClearTable(const unsigned slots_in_table, const Entry fill_value,
                Entry *d_contents) {
  clear_table<Entry><<<ComputeGridDim(slots_in_table), kBlockSize>>>(
      slots_in_table, fill_value, d_contents);
  TV_CHECK_CUDA_ERR_V2("Error occurred during hash table clear.\n");
}

void CallCuckooHash(const unsigned n, const unsigned num_hash_functions,
                    const unsigned *d_keys, const unsigned *d_values,
                    const unsigned table_size, const Functions<2> constants_2,
                    const Functions<3> constants_3,
                    const Functions<4> constants_4,
                    const Functions<5> constants_5,
                    const unsigned max_iterations, Entry *d_contents,
                    uint2 stash_constants, unsigned *d_stash_count,
                    unsigned *d_failures, unsigned *d_iterations_taken) {
  // Build the table.
  cudaMemset(d_failures, 0, sizeof(unsigned));
  if (num_hash_functions == 2) {
    CuckooHash<<<ComputeGridDim(n), kBlockSize>>>(
        n, d_keys, d_values, table_size, constants_2, max_iterations,
        d_contents, stash_constants, d_stash_count, d_failures,
        d_iterations_taken);
  } else if (num_hash_functions == 3) {
    CuckooHash<<<ComputeGridDim(n), kBlockSize>>>(
        n, d_keys, d_values, table_size, constants_3, max_iterations,
        d_contents, stash_constants, d_stash_count, d_failures,
        d_iterations_taken);
  } else if (num_hash_functions == 4) {
    CuckooHash<<<ComputeGridDim(n), kBlockSize>>>(
        n, d_keys, d_values, table_size, constants_4, max_iterations,
        d_contents, stash_constants, d_stash_count, d_failures,
        d_iterations_taken);
  } else {
    CuckooHash<<<ComputeGridDim(n), kBlockSize>>>(
        n, d_keys, d_values, table_size, constants_5, max_iterations,
        d_contents, stash_constants, d_stash_count, d_failures,
        d_iterations_taken);
  }

  CUDA_CHECK_ERROR("Error occurred during hash table build.\n");
}

void CallHashRetrieve(const unsigned n_queries,
                      const unsigned num_hash_functions, const unsigned *d_keys,
                      const unsigned table_size, const Entry *d_contents,
                      const Functions<2> constants_2,
                      const Functions<3> constants_3,
                      const Functions<4> constants_4,
                      const Functions<5> constants_5,
                      const uint2 stash_constants, const unsigned stash_count,
                      unsigned *d_values) {
  unsigned *d_retrieval_probes = NULL;
#ifdef TRACK_ITERATIONS
  CUDA_SAFE_CALL(
      cudaMalloc((void **)&d_retrieval_probes, sizeof(unsigned) * n_queries));
#endif

  if (num_hash_functions == 2) {
    hash_retrieve<<<ComputeGridDim(n_queries), kBlockSize>>>(
        n_queries, d_keys, table_size, d_contents, constants_2, stash_constants,
        stash_count, d_values, d_retrieval_probes);
  } else if (num_hash_functions == 3) {
    hash_retrieve<<<ComputeGridDim(n_queries), kBlockSize>>>(
        n_queries, d_keys, table_size, d_contents, constants_3, stash_constants,
        stash_count, d_values, d_retrieval_probes);
  } else if (num_hash_functions == 4) {
    hash_retrieve<<<ComputeGridDim(n_queries), kBlockSize>>>(
        n_queries, d_keys, table_size, d_contents, constants_4, stash_constants,
        stash_count, d_values, d_retrieval_probes);
  } else {
    hash_retrieve<<<ComputeGridDim(n_queries), kBlockSize>>>(
        n_queries, d_keys, table_size, d_contents, constants_5, stash_constants,
        stash_count, d_values, d_retrieval_probes);
  }

  CUDA_CHECK_ERROR("Retrieval failed.\n");

#ifdef TRACK_ITERATIONS
  OutputRetrievalStatistics(n_queries, d_retrieval_probes, num_hash_functions);
  CUDA_SAFE_CALL(cudaFree(d_retrieval_probes));
#endif
}
}; // namespace CUDAWrapper

}; // namespace cuhash
