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
 * debugging.h
 *
 * @brief Debugging/statistics/performance utilities header for hash tables.
 */

#ifndef CUDAHT__CUCKOO__SRC__LIBRARY__DEBUGGING__H
#define CUDAHT__CUCKOO__SRC__LIBRARY__DEBUGGING__H

#include "definitions.h"
#include <cuda_runtime_api.h>
#include <vector_types.h>

#include <algorithm>

namespace cuhash {

//! @name Debugging functions
/// @{

void TakeHashFunctionStatistics(const unsigned num_keys, const unsigned *d_keys,
                                const unsigned table_size,
                                const uint2 *constants,
                                const unsigned kNumHashFunctions);

//! Output how many probes were required by each thread to perform the
//! retrieval.
/*! @param[in]  n_queries           Number of queries being performed.
 *  @param[in]  d_retrieval_probes  Device array: the number of probes taken for
 * each thread's retrieval.
 *  @param[in]  n_functions         Number of hash functions used.
 */
void OutputRetrievalStatistics(const unsigned n_queries,
                               const unsigned *d_retrieval_probes,
                               const unsigned n_functions);

//! Outputs information about how many iterations threads required to
//! successfully cuckoo hash.
/*! @param[in]  n                       Number of keys in the input.
 *  @param[in]  d_iterations_taken      Device mem: Number of iterations each
 * thread took.
 *  @param[in]  d_max_iterations_taken  Device mem: Largest number of iterations
 * taken by any thread.
 */
void OutputBuildStatistics(const unsigned n,
                           const unsigned *d_iterations_taken);

//! Prints out the contents of the stash.
void PrintStashContents(const Entry *d_stash);

//! Checks if a key is assigned the same slot by different hash functions.
bool CheckAssignedSameSlot(const unsigned N, const unsigned num_keys,
                           const unsigned *d_keys, const unsigned table_size,
                           uint2 *constants);

/// @}

}; // namespace cuhash

#endif

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
