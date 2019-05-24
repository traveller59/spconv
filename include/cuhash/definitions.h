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
 * @file definitions.h
 *
 * @brief Stores configuration flags and definitions for hard-coded values in
 * hash table implementations.
 */

#ifndef CUDAHT__CUCKOO__SRC__LIBRARY__DEFINITIONS__H
#define CUDAHT__CUCKOO__SRC__LIBRARY__DEFINITIONS__H

#include <cstdio>
#include <limits>
#include <tensorview/tensorview.h>

/* --------------------------------------------------------------------------
   Debugging.
   -------------------------------------------------------------------------- */
#ifdef _DEBUG
//! Forces the hash functions to generate a full set of slots for each key when
//! not using subtables.
// #define FORCEFULLY_GENERATE_NO_CYCLES

//! Count how many iterations are taken to insert/find items.
#define TRACK_ITERATIONS

//! Count how many items fail to be inserted when the hash table fails to build.
#define COUNT_UNINSERTED

//! Take some statistics on the hash functions.
#define TAKE_HASH_FUNCTION_STATISTICS

#ifdef TAKE_HASH_FUNCTION_STATISTICS
//! Determine how many keys hash into each table slot.
#define COUNT_HOW_MANY_HASH_INTO_EACH_SLOT

//! Determine how many unique slots a key is assigned.
#define COUNT_HOW_MANY_HAVE_CYCLES
#endif
#endif

#ifdef USE_DAN_OUTPUT
#include <Utilities/output.h>
//! Logs any error messages.
inline void PrintMessage(const char *message, const bool error = false) {
  PrintIndentedMessage(message, error);
}
#else
//! Prints a message out to the console.
inline void PrintMessage(const char *message, const bool error = false) {
  if (error) {
    printf("cudahash: %s\n", message);
  } else {
    printf("%s\n", message);
  }
}
#endif

/* -------------------------------------------------------------------------
   Hash table constants and definitions.
   ------------------------------------------------------------------------- */
namespace cuhash {

/**
 * \addtogroup cudpp_hash_data_structures
 *
 * @{
 */

typedef unsigned long long
    Entry; //!< A key and its value are stored in a 64-bit number.  The key is
           //!< stored in the upper 32 bits.

const unsigned kMaxRestartAttempts = 10; //!< Number of build attempts.
const unsigned kKeyEmpty = 0xffffffffu; //!< Signifies empty slots in the table.
const unsigned kNotFound =
    0xffffffffu; //!< Signifies that a query key was not found.
const unsigned kMaxHashFunctions =
    5; //!< Maximum number of hash functions allowed.
const unsigned kStashSize =
    101; //!< How many slots the stash hash table contains.

//! Value indicating that a hash table slot has no valid item within it.
const Entry kEntryEmpty = Entry(kKeyEmpty) << 32;

//! Value returned when a query fails.
const Entry kEntryNotFound = (Entry(kKeyEmpty) << 32) + kNotFound;

//! Number of threads to put in a thread block.
const unsigned kBlockSize = 64;

//! Number of blocks to put along each axis of the grid.
const unsigned kGridSize = 16384;

//! Minimum table sizes for 2 through 5 functions.
const float kMinimumSpaceUsages[] = {std::numeric_limits<float>::max(),
                                     std::numeric_limits<float>::max(),
                                     2.01f,
                                     1.1f,
                                     1.03f,
                                     1.02f};

/** @} */ // end cudpp_hash_data_structures

}; // namespace cuhash

#endif
