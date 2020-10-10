// Copyright 2019-2020 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <sstream>
#ifdef TV_USE_STACKTRACE
#if defined(WIN32) || defined(_WIN32) ||                                       \
    defined(__WIN32) && !defined(__CYGWIN__)
#define BOOST_STACKTRACE_USE_WINDBG
#else
// require linking with -ldl and -lbacktrace in linux
#define BOOST_STACKTRACE_USE_BACKTRACE
#endif
#include <boost/stacktrace.hpp>
#endif
#ifdef TV_CUDA
#include <cuda.h>
#endif
#if defined(TV_USE_BOOST_TYPEOF) || (!defined(__clang__) && defined(CUDA_VERSION) && CUDA_VERSION >= 11000)
// a workaround when built with cuda 11
// two options: use BOOST_TYPEOF or identity_t.
// this is a nvcc bug, msvc/gcc/clang don't have this problem.
// #include <boost/typeof/typeof.hpp>
// #define TV_DECLTYPE(x) BOOST_TYPEOF(x)
namespace tv{
template <typename T>
using identity_t = T;
}
#define TV_DECLTYPE(x) tv::identity_t<decltype(x)>
#else
#define TV_DECLTYPE(x) decltype(x)
#endif

namespace tv {

template <class SStream, class T> void sstream_print(SStream &ss, T val) {
  ss << val;
}

template <class SStream, class T, class... TArgs>
void sstream_print(SStream &ss, T val, TArgs... args) {
  ss << val << " ";
  sstream_print(ss, args...);
}

template <class... TArgs> void ssprint(TArgs... args) {
  std::stringstream ss;
  sstream_print(ss, args...);
  std::cout << ss.str() << std::endl;
}

#ifdef TV_USE_STACKTRACE
#define TV_BACKTRACE_PRINT(ss)                                                 \
  ss << std::endl << boost::stacktrace::stacktrace();
#else
#define TV_BACKTRACE_PRINT(ss)
#endif

#define TV_THROW_RT_ERR(...)                                                   \
  {                                                                            \
    std::stringstream __macro_s;                                               \
    __macro_s << __FILE__ << " " << __LINE__ << "\n";                          \
    tv::sstream_print(__macro_s, __VA_ARGS__);                                 \
    TV_BACKTRACE_PRINT(__macro_s);                                             \
    throw std::runtime_error(__macro_s.str());                                 \
  }

#define TV_THROW_INVALID_ARG(...)                                              \
  {                                                                            \
    std::stringstream __macro_s;                                               \
    __macro_s << __FILE__ << " " << __LINE__ << "\n";                          \
    tv::sstream_print(__macro_s, __VA_ARGS__);                                 \
    TV_BACKTRACE_PRINT(__macro_s);                                             \
    throw std::invalid_argument(__macro_s.str());                              \
  }

#define TV_ASSERT_RT_ERR(expr, ...)                                            \
  {                                                                            \
    if (!(expr)) {                                                             \
      std::stringstream __macro_s;                                             \
      __macro_s << __FILE__ << " " << __LINE__ << "\n";                        \
      __macro_s << #expr << " assert faild. ";                                 \
      tv::sstream_print(__macro_s, __VA_ARGS__);                               \
      TV_BACKTRACE_PRINT(__macro_s);                                           \
      throw std::runtime_error(__macro_s.str());                               \
    }                                                                          \
  }

#define TV_ASSERT_INVALID_ARG(expr, ...)                                       \
  {                                                                            \
    if (!(expr)) {                                                             \
      std::stringstream __macro_s;                                             \
      __macro_s << __FILE__ << " " << __LINE__ << "\n";                        \
      __macro_s << #expr << " assert faild. ";                                 \
      tv::sstream_print(__macro_s, __VA_ARGS__);                               \
      TV_BACKTRACE_PRINT(__macro_s);                                           \
      throw std::invalid_argument(__macro_s.str());                            \
    }                                                                          \
  }
} // namespace tv