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
 * debugging.cpp
 *
 * @brief Debugging/statistics/performance utilities for hash tables.
 */

#include <cuhash/debugging.h>
#include <cuhash/definitions.h>

#include <algorithm>
#include <cstring>
#include <cuhash/cuda_util.h>

namespace cuhash {

void OutputRetrievalStatistics(const unsigned n_queries,
                               const unsigned *d_retrieval_probes,
                               const unsigned n_functions) {
  unsigned *retrieval_probes = new unsigned[n_queries];
  CUDA_SAFE_CALL(cudaMemcpy(retrieval_probes, d_retrieval_probes,
                            sizeof(unsigned) * n_queries,
                            cudaMemcpyDeviceToHost));

  // Create a histogram showing how many items needed how many probes to be
  // found.
  unsigned possible_probes = n_functions + 2;
  unsigned *histogram = new unsigned[possible_probes];
  memset(histogram, 0, sizeof(unsigned) * (possible_probes));
  for (unsigned i = 0; i < n_queries; ++i) {
    histogram[retrieval_probes[i]]++;
  }

  // Dump it.
  char buffer[10000];
  sprintf(buffer, "Probes for retrieval: ");
  PrintMessage(buffer);
  for (unsigned i = 0; i < possible_probes; ++i) {
    sprintf(buffer, "\t(%u, %u)", i, histogram[i]);
    PrintMessage(buffer);
  }
  delete[] retrieval_probes;
  delete[] histogram;
}

void OutputBuildStatistics(const unsigned n,
                           const unsigned *d_iterations_taken) {
  // Output how many iterations each thread took until it found an empty slot.
  unsigned *iterations_taken = new unsigned[n];
  CUDA_SAFE_CALL(cudaMemcpy(iterations_taken, d_iterations_taken,
                            sizeof(unsigned) * n, cudaMemcpyDeviceToHost));
  std::sort(iterations_taken, iterations_taken + n);
  unsigned total_iterations = 0;
  unsigned max_iterations_taken = 0;
  for (unsigned i = 0; i < n; ++i) {
    total_iterations += iterations_taken[i];
    max_iterations_taken = std::max(max_iterations_taken, iterations_taken[i]);
  }

  unsigned current_value = iterations_taken[0];
  unsigned count = 1;
  char buffer[10000];
  sprintf(buffer, "Iterations taken:\n");
  for (unsigned i = 1; i < n; ++i) {
    if (iterations_taken[i] != current_value) {
      sprintf(buffer, "%s\t(%u, %u)\n", buffer, current_value, count);
      current_value = iterations_taken[i];
      count = 1;
    } else {
      count++;
    }
  }
  sprintf(buffer, "%s\t(%u, %u)", buffer, current_value, count);
  PrintMessage(buffer);
  sprintf(buffer, "Total iterations: %u", total_iterations);
  PrintMessage(buffer);
  sprintf(buffer, "Avg/Med/Max iterations: (%f %u %u)",
          (float)total_iterations / n, iterations_taken[n / 2],
          iterations_taken[n - 1]);
  PrintMessage(buffer);
  delete[] iterations_taken;

  // Print the length of the longest eviction chain.
  sprintf(buffer, "Max iterations: %u", max_iterations_taken);
  PrintMessage(buffer);
}

}; // namespace cuhash

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
