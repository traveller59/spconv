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
 * @file hash_table.cuh
 *
 * @brief Implements kernel and __device__ functions for a basic hash table.
 */

#ifndef CUDAHT__CUCKOO__SRC__LIBRARY__HASH_TABLE__CUH
#define CUDAHT__CUCKOO__SRC__LIBRARY__HASH_TABLE__CUH

#include "definitions.h"
#include "hash_table.h"
#include <driver_types.h>
#include <tensorview/tensorview.h>

namespace cuhash {

//! Makes an 64-bit Entry out of a key-value pair for the hash table.
TV_HOST_DEVICE_INLINE Entry make_entry(unsigned key, unsigned value) {
  return (Entry(key) << 32) + value;
}

//! Returns the key of an Entry.
TV_HOST_DEVICE_INLINE unsigned get_key(Entry entry) {
  return (unsigned)(entry >> 32);
}

//! Returns the value of an Entry.
TV_HOST_DEVICE_INLINE unsigned get_value(Entry entry) {
  return (unsigned)(entry & 0xffffffff);
}

//! @name Internal
//! @brief Functions used for building the hash table.
//! @{

//! Fills the entire array with a specific value.
template <class T>
__global__ void clear_table(const unsigned table_size, const T value,
                            T *table) {
  unsigned thread_index = threadIdx.x + blockIdx.x * blockDim.x +
                          blockIdx.y * blockDim.x * gridDim.x;
  if (thread_index < table_size) {
    table[thread_index] = value;
  }
}

//! Determine where in the hash table the key could be located.
template <unsigned kNumHashFunctions>
__device__ void KeyLocations(const Functions<kNumHashFunctions> constants,
                             const unsigned table_size, const unsigned key,
                             unsigned locations[kNumHashFunctions]) {
// Compute all possible locations for the key in the big table.
#pragma unroll
  for (int i = 0; i < kNumHashFunctions; ++i) {
    locations[i] = hash_function(constants, i, key) % table_size;
  }
}
//! @}

/* --------------------------------------------------------------------------
   Retrieval functions.
   -------------------------------------------------------------------------- */
//! Answers a single query.
/*! @ingroup PublicInterface
 *  @param[in]  key                   Query key
 *  @param[in]  table_size            Size of the hash table
 *  @param[in]  table                 The contents of the hash table
 *  @param[in]  constants             The hash functions used to build the table
 *  @param[in]  stash_constants       The hash function used to build the stash
 *  @param[in]  stash_count           The number of items in the stash
 *  @param[out] num_probes_required   Debug only: The number of probes required
 * to resolve the query.
 *  @returns The value of the query key, if the key exists in the table.
 * Otherwise, \ref kNotFound will be returned.
 */
template <unsigned kNumHashFunctions>
__device__ unsigned
retrieve(const unsigned query_key, const unsigned table_size,
         const Entry *table, const Functions<kNumHashFunctions> constants,
         const uint2 stash_constants, const unsigned stash_count,
         unsigned *num_probes_required = NULL) {
  // Identify all of the locations that the key can be located in.
  unsigned locations[kNumHashFunctions];
  KeyLocations(constants, table_size, query_key, locations);

  // Check each location until the key is found.
  unsigned num_probes = 1;
  Entry entry = table[locations[0]];
  unsigned key = get_key(entry);

#pragma unroll
  for (unsigned i = 1; i < kNumHashFunctions; ++i) {
    if (key != query_key && key != kNotFound) {
      num_probes++;
      entry = table[locations[i]];
      key = get_key(entry);
    }
  }

  // Check the stash.
  if (stash_count && get_key(entry) != query_key) {
    num_probes++;
    const Entry *stash = table + table_size;
    unsigned slot = stash_hash_function(stash_constants, query_key);
    entry = stash[slot];
  }

#ifdef TRACK_ITERATIONS
  if (num_probes_required) {
    *num_probes_required = num_probes;
  }
#endif

  if (get_key(entry) == query_key) {
    return get_value(entry);
  } else {
    return kNotFound;
  }
}

//! Perform a retrieval from a basic hash table.  Each thread manages a single
//! query.
template <unsigned kNumHashFunctions>
__global__ void hash_retrieve(const unsigned n_queries, const unsigned *keys_in,
                              const unsigned table_size, const Entry *table,
                              const Functions<kNumHashFunctions> constants,
                              const uint2 stash_constants,
                              const unsigned stash_count, unsigned *values_out,
                              unsigned *num_probes_required = NULL) {
  // Get the key.
  unsigned thread_index = threadIdx.x + blockIdx.x * blockDim.x +
                          blockIdx.y * blockDim.x * gridDim.x;
  if (thread_index >= n_queries)
    return;
  unsigned key = keys_in[thread_index];

  values_out[thread_index] = retrieve<kNumHashFunctions>(
      key, table_size, table, constants, stash_constants, stash_count,
      (num_probes_required ? num_probes_required + thread_index : NULL));
}

/* --------------------------------------------------------------------------
   Build a cuckoo hash table.
   -------------------------------------------------------------------------- */
//! @name Internal
//! @{

//! Determine where to insert the key next.  The hash functions are used in
//! round-robin order.
template <unsigned kNumHashFunctions>
__device__ unsigned
determine_next_location(const Functions<kNumHashFunctions> constants,
                        const unsigned table_size, const unsigned key,
                        const unsigned previous_location) {
  // Identify all possible locations for the entry.
  unsigned locations[kNumHashFunctions];
#pragma unroll
  for (unsigned i = 0; i < kNumHashFunctions; ++i) {
    locations[i] = hash_function(constants, i, key) % table_size;
  }

  // Figure out where the item should be inserted next.
  unsigned next_location = locations[0];
#pragma unroll
  for (int i = kNumHashFunctions - 2; i >= 0; --i) {
    next_location =
        (previous_location == locations[i] ? locations[i + 1] : next_location);
  }
  return next_location;
}

//! Attempts to insert a single entry into the hash table.
/*! This process stops after a certain number of iterations.  If the thread is
    still holding onto an item because of an eviction, it tries the stash.
    If it fails to enter the stash, it returns false.
    Otherwise, it succeeds and returns true.
 */
template <unsigned kNumHashFunctions>
__device__ bool
insert(const unsigned table_size, const Functions<kNumHashFunctions> constants,
       const uint2 stash_constants, const unsigned max_iteration_attempts,
       Entry *table, unsigned *stash_count, Entry entry,
       unsigned *iterations_used) {
  unsigned key = get_key(entry);

  // The key is always inserted into its first slot at the start.
  unsigned location = hash_function(constants, 0, key) % table_size;

  // Keep inserting until an empty slot is found or the eviction chain grows too
  // large.
  for (unsigned its = 1; its <= max_iteration_attempts; its++) {
    // Insert the new entry.
    entry = atomicExch(&table[location], entry);
    key = get_key(entry);

    // If no key was evicted, we're done.
    if (key == kKeyEmpty) {
      *iterations_used = its;
      break;
    }

    // Otherwise, determine where the evicted key will go.
    location = determine_next_location(constants, table_size, key, location);
  }

  if (key != kKeyEmpty) {
    // Shove it into the stash.
    unsigned slot = stash_hash_function(stash_constants, key);
    Entry *stash = table + table_size;
    Entry replaced_entry = atomicCAS(stash + slot, kEntryEmpty, entry);
    if (replaced_entry != kEntryEmpty) {
      return false;
    } else {
      atomicAdd(stash_count, 1);
    }
  }

  return true;
}

// Build a basic hash table, using one big table.
template <unsigned kNumHashFunctions>
__global__ void CuckooHash(const unsigned n_entries, const unsigned *keys,
                           const unsigned *values, const unsigned table_size,
                           const Functions<kNumHashFunctions> constants,
                           const unsigned max_iteration_attempts, Entry *table,
                           uint2 stash_constants, unsigned *stash_count,
                           unsigned *failures,
                           unsigned *iterations_taken = nullptr) {
  // Check if this thread has an item and if any previous threads failed.
  unsigned thread_index = threadIdx.x + blockIdx.x * blockDim.x +
                          blockIdx.y * blockDim.x * gridDim.x;
  if (thread_index >= n_entries || *failures)
    return;
  Entry entry = make_entry(keys[thread_index], values[thread_index]);

  unsigned iterations = 0;
  bool success = insert<kNumHashFunctions>(
      table_size, constants, stash_constants, max_iteration_attempts, table,
      stash_count, entry, &iterations);

  if (success == false) {
    // The eviction chain grew too large.  Report failure.
#ifdef COUNT_UNINSERTED
    atomicAdd(failures, 1);
#else
    *failures = 1;
#endif
  }

#ifdef TRACK_ITERATIONS
  iterations_taken[thread_index] = iterations;
#endif
}
//! @}

}; // namespace cuhash

#endif

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
