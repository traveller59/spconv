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
 * @file
 * debugging.cu
 *
 * @brief Debugging/statistics/performance utilities for hash tables.
 */

#include <cuhash/debugging.h>
#include <cuhash/definitions.h>
#include <cuhash/hash_table.cuh>

#include <algorithm>
#include <cuhash/cuda_util.h>

namespace cuhash {

//! Debugging function: Takes statistics on the hash functions' distribution.
/*! Determines:
 *    - How many unique slots each key has.
 *    - How many keys hash into each slot.
 *    - Whether any keys failed to get a full set of slots.
 */
__global__ void take_hash_function_statistics_kernel(
    const unsigned *keys, const unsigned n_entries, const unsigned table_size,
    const uint2 *constants, const unsigned num_functions,
    unsigned *num_slots_available, unsigned *num_hashing_in, unsigned *failed) {
  unsigned thread_index = threadIdx.x + blockIdx.x * blockDim.x +
                          blockIdx.y * blockDim.x * gridDim.x;

  if (thread_index >= n_entries)
    return;
  unsigned key = keys[thread_index];

  // Determine all of the locations the key hashes into.
  // Also count how many keys hash into each location.
  unsigned locations[kMaxHashFunctions];
  for (unsigned i = 0; i < num_functions; ++i) {
    locations[i] = hash_function_inner(constants[i], key) % table_size;

    if (num_hashing_in != NULL) {
      atomicAdd(num_hashing_in + locations[i], 1);
    }
  }

  // Determine whether all of the locations were different.
  unsigned num_slots = 1;
  for (unsigned i = 1; i < num_functions; ++i) {
    bool matched = false;
    for (unsigned j = 0; j < i; ++j) {
      if (locations[i] == locations[j]) {
        matched = true;
        break;
      }
    }
    if (!matched) {
      num_slots++;
    }
  }

  if (num_slots_available != NULL) {
    num_slots_available[thread_index] = num_slots;
  }

  if (failed != NULL && num_slots != num_functions) {
    *failed = 1;
  }
}

void TakeHashFunctionStatistics(const unsigned num_keys, const unsigned *d_keys,
                                const unsigned table_size,
                                const uint2 *constants,
                                const unsigned kNumHashFunctions) {
  char buffer[16000];
  PrintMessage("Hash function constants: ");

  for (unsigned i = 0; i < kNumHashFunctions; ++i) {
    sprintf(buffer, "\t%10u, %10u", constants[i].x, constants[i].y);
    PrintMessage(buffer);
  }

  unsigned *d_num_hashing_in = NULL;
#ifdef COUNT_HOW_MANY_HASH_INTO_EACH_SLOT
  CUDA_SAFE_CALL(
      cudaMalloc((void **)&d_num_hashing_in, sizeof(unsigned) * table_size));
  CUDA_SAFE_CALL(
      cudaMemset(d_num_hashing_in, 0, sizeof(unsigned) * table_size));
#endif

  unsigned *d_num_slots_available = NULL;
#ifdef COUNT_HOW_MANY_HAVE_CYCLES
  CUDA_SAFE_CALL(
      cudaMalloc((void **)&d_num_slots_available, sizeof(unsigned) * num_keys));
#endif
  uint2 *d_constants = NULL;
  CUDA_SAFE_CALL(
      cudaMalloc((void **)&d_constants, sizeof(uint2) * kNumHashFunctions));
  CUDA_SAFE_CALL(cudaMemcpy(d_constants, constants,
                            sizeof(uint2) * kNumHashFunctions,
                            cudaMemcpyHostToDevice));

  take_hash_function_statistics_kernel<<<ComputeGridDim(num_keys),
                                         kBlockSize>>>(
      d_keys, num_keys, table_size, d_constants, kNumHashFunctions,
      d_num_slots_available, d_num_hashing_in, NULL);
  CUDA_SAFE_CALL(cudaFree(d_constants));

#ifdef COUNT_HOW_MANY_HASH_INTO_EACH_SLOT
  unsigned *num_hashing_in = new unsigned[table_size];
  CUDA_SAFE_CALL(cudaMemcpy(num_hashing_in, d_num_hashing_in,
                            sizeof(unsigned) * table_size,
                            cudaMemcpyDeviceToHost));

  /*
  // Print how many items hash into each slot.
  // Used to make sure items are spread evenly throughout the table.
  buffer[0] = '\0';
  PrintMessage("Num hashing into each: ", true);
  for (unsigned i = 0; i < table_size; ++i) {
    sprintf(buffer, "%s\t%2u", buffer, num_hashing_in[i]);
    if (i % 25 == 24) {
      PrintMessage(buffer, true);
      buffer[0] = '\0';
    }
  }
  PrintMessage(buffer,true);
  */

  // Print a histogram of how many items are hashed into each slot.  Shows
  // if average number of items hashing into each slot is low.
  std::sort(num_hashing_in, num_hashing_in + table_size);
  int count = 1;
  unsigned previous = num_hashing_in[0];
  sprintf(buffer, "Num items hashing into a slot:\t");
  PrintMessage(buffer);
  for (unsigned i = 1; i < table_size; ++i) {
    if (num_hashing_in[i] != previous) {
      sprintf(buffer, "\t(%u, %u)", previous, count);
      PrintMessage(buffer);
      previous = num_hashing_in[i];
      count = 1;
    } else {
      count++;
    }
  }
  sprintf(buffer, "\t(%u, %u)", previous, count);
  PrintMessage(buffer);

  delete[] num_hashing_in;
  CUDA_SAFE_CALL(cudaFree(d_num_hashing_in));
#endif

#ifdef COUNT_HOW_MANY_HAVE_CYCLES
  unsigned *num_slots_available = new unsigned[num_keys];
  CUDA_SAFE_CALL(cudaMemcpy(num_slots_available, d_num_slots_available,
                            sizeof(unsigned) * num_keys,
                            cudaMemcpyDeviceToHost));

  static const unsigned kHistogramSize = kNumHashFunctions + 1;
  unsigned *histogram = new unsigned[kHistogramSize];
  memset(histogram, 0, sizeof(unsigned) * kHistogramSize);
  for (unsigned i = 0; i < num_keys; ++i) {
    histogram[num_slots_available[i]]++;
  }

  sprintf(buffer, "Slots assigned to each key: ");
  for (unsigned i = 1; i < kHistogramSize; ++i) {
    sprintf(buffer, "%s(%u, %u) ", buffer, i, histogram[i]);
  }
  PrintMessage(buffer);

  delete[] histogram;
  delete[] num_slots_available;
  CUDA_SAFE_CALL(cudaFree(d_num_slots_available));
#endif
}

bool CheckAssignedSameSlot(const unsigned N, const unsigned num_keys,
                           const unsigned *d_keys, const unsigned table_size,
                           uint2 *constants) {
  unsigned *d_cycle_exists = NULL;
  uint2 *d_constants = NULL;

  CUDA_SAFE_CALL(cudaMalloc((void **)&d_cycle_exists, sizeof(unsigned)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_constants, sizeof(uint2) * N));

  CUDA_SAFE_CALL(cudaMemset(d_cycle_exists, 0, sizeof(unsigned)));
  CUDA_SAFE_CALL(cudaMemcpy(d_constants, constants, sizeof(uint2) * N,
                            cudaMemcpyHostToDevice));

  // Check if all keys were given a full set of N slots by the functions.
  take_hash_function_statistics_kernel<<<ComputeGridDim(num_keys),
                                         kBlockSize>>>(
      d_keys, num_keys, table_size, d_constants, N, NULL, NULL, d_cycle_exists);

  unsigned cycle_exists;
  CUDA_SAFE_CALL(cudaMemcpy(&cycle_exists, d_cycle_exists, sizeof(unsigned),
                            cudaMemcpyDeviceToHost));

  CUDA_SAFE_CALL(cudaFree(d_cycle_exists));
  CUDA_SAFE_CALL(cudaFree(d_constants));

  return (cycle_exists != 0);
}

void PrintStashContents(const Entry *d_stash) {
  Entry *stash = new Entry[cuhash::kStashSize];
  CUDA_SAFE_CALL(cudaMemcpy(stash, d_stash, sizeof(Entry) * cuhash::kStashSize,
                            cudaMemcpyDeviceToHost));
  for (unsigned i = 0; i < cuhash::kStashSize; ++i) {
    if (get_key(stash[i]) != kKeyEmpty) {
      char buffer[256];
      sprintf(buffer, "Stash[%u]: %u = %u", i, get_key(stash[i]),
              get_value(stash[i]));
      PrintMessage(buffer, true);
    }
  }
  delete[] stash;
}

}; // namespace cuhash

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
