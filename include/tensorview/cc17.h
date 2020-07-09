/*
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou,
Iain Melvin, Jason Weston) Copyright (c) 2006      Idiap Research Institute
(Samy Bengio) Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert,
Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories
America and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include <type_traits>
#include <utility>

namespace tv {

#ifdef __cpp_lib_void_t

template <class T> using void_t = std::void_t<T>;

#else

// Implementation taken from http://en.cppreference.com/w/cpp/types/void_t
// (it takes CWG1558 into account and also works for older compilers)
template <typename... Ts> struct make_void { typedef void type; };
template <typename... Ts> using void_t = typename make_void<Ts...>::type;

#endif

namespace detail {
struct _identity final {
  template <class T> using type_identity = T;

  template <class T> decltype(auto) operator()(T &&arg) {
    return std::forward<T>(arg);
  }
};
template <class Func, class Enable = void>
struct function_takes_identity_argument : std::false_type {};
#if defined(_MSC_VER)
// For some weird reason, MSVC shows a compiler error when using guts::void_t
// instead of std::void_t. But we're only building on MSVC versions that have
// std::void_t, so let's just use that one.
template <class Func>
struct function_takes_identity_argument<
    Func, std::void_t<decltype(std::declval<Func>()(_identity()))>>
    : std::true_type {};
#else
template <class Func>
struct function_takes_identity_argument<
    Func, void_t<decltype(std::declval<Func>()(_identity()))>>
    : std::true_type {};
#endif

template <bool Condition> struct _if_constexpr;

template <> struct _if_constexpr<true> final {
  template <
      class ThenCallback, class ElseCallback,
      std::enable_if_t<function_takes_identity_argument<ThenCallback>::value,
                       void *> = nullptr>
  static decltype(auto) call(ThenCallback &&thenCallback,
                             ElseCallback && /* elseCallback */) {
    // The _identity instance passed in can be used to delay evaluation of an
    // expression, because the compiler can't know that it's just the identity
    // we're passing in.
    return thenCallback(_identity());
  }

  template <
      class ThenCallback, class ElseCallback,
      std::enable_if_t<!function_takes_identity_argument<ThenCallback>::value,
                       void *> = nullptr>
  static decltype(auto) call(ThenCallback &&thenCallback,
                             ElseCallback && /* elseCallback */) {
    return thenCallback();
  }
};

template <> struct _if_constexpr<false> final {
  template <
      class ThenCallback, class ElseCallback,
      std::enable_if_t<function_takes_identity_argument<ElseCallback>::value,
                       void *> = nullptr>
  static decltype(auto) call(ThenCallback && /* thenCallback */,
                             ElseCallback &&elseCallback) {
    // The _identity instance passed in can be used to delay evaluation of an
    // expression, because the compiler can't know that it's just the identity
    // we're passing in.
    return elseCallback(_identity());
  }

  template <
      class ThenCallback, class ElseCallback,
      std::enable_if_t<!function_takes_identity_argument<ElseCallback>::value,
                       void *> = nullptr>
  static decltype(auto) call(ThenCallback && /* thenCallback */,
                             ElseCallback &&elseCallback) {
    return elseCallback();
  }
};
} // namespace detail

/*
 * Get something like C++17 if constexpr in C++14.
 *
 * Example 1: simple constexpr if/then/else
 *   template<int arg> int increment_absolute_value() {
 *     int result = arg;
 *     if_constexpr<(arg > 0)>(
 *       [&] { ++result; }  // then-case
 *       [&] { --result; }  // else-case
 *     );
 *     return result;
 *   }
 *
 * Example 2: without else case (i.e. conditionally prune code from assembly)
 *   template<int arg> int decrement_if_positive() {
 *     int result = arg;
 *     if_constexpr<(arg > 0)>(
 *       // This decrement operation is only present in the assembly for
 *       // template instances with arg > 0.
 *       [&] { --result; }
 *     );
 *     return result;
 *   }
 *
 * Example 3: branch based on type (i.e. replacement for SFINAE)
 *   struct MyClass1 {int value;};
 *   struct MyClass2 {int val};
 *   template <class T>
 *   int func(T t) {
 *     return if_constexpr<std::is_same<T, MyClass1>::value>(
 *       [&](auto _) { return _(t).value; }, // this code is invalid for T ==
 * MyClass2, so a regular non-constexpr if statement wouldn't compile
 *       [&](auto _) { return _(t).val; }    // this code is invalid for T ==
 * MyClass1
 *     );
 *   }
 *
 * Note: The _ argument passed in Example 3 is the identity function, i.e. it
 * does nothing. It is used to force the compiler to delay type checking,
 * because the compiler doesn't know what kind of _ is passed in. Without it,
 * the compiler would fail when you try to access t.value but the member doesn't
 * exist.
 *
 * Note: In Example 3, both branches return int, so func() returns int. This is
 * not necessary. If func() had a return type of "auto", then both branches
 * could return different types, say func<MyClass1>() could return int and
 * func<MyClass2>() could return string.
 */
template <bool Condition, class ThenCallback, class ElseCallback>
decltype(auto) if_constexpr(ThenCallback &&thenCallback,
                            ElseCallback &&elseCallback) {
#if defined(__cpp_if_constexpr)
  // If we have C++17, just use it's "if constexpr" feature instead of wrapping
  // it. This will give us better error messages.
  if constexpr (Condition) {
    if constexpr (detail::function_takes_identity_argument<
                      ThenCallback>::value) {
      return std::forward<ThenCallback>(thenCallback)(detail::_identity());
    } else {
      return std::forward<ThenCallback>(thenCallback)();
    }
  } else {
    if constexpr (detail::function_takes_identity_argument<
                      ElseCallback>::value) {
      return std::forward<ElseCallback>(elseCallback)(detail::_identity());
    } else {
      return std::forward<ElseCallback>(elseCallback)();
    }
  }
#else
  // C++14 implementation of if constexpr
  return detail::_if_constexpr<Condition>::call(
      std::forward<ThenCallback>(thenCallback),
      std::forward<ElseCallback>(elseCallback));
#endif
}

template <bool Condition, class ThenCallback>
decltype(auto) if_constexpr(ThenCallback &&thenCallback) {
#if defined(__cpp_if_constexpr)
  // If we have C++17, just use it's "if constexpr" feature instead of wrapping
  // it. This will give us better error messages.
  if constexpr (Condition) {
    if constexpr (detail::function_takes_identity_argument<
                      ThenCallback>::value) {
      return std::forward<ThenCallback>(thenCallback)(detail::_identity());
    } else {
      return std::forward<ThenCallback>(thenCallback)();
    }
  }
#else
  // C++14 implementation of if constexpr
  return if_constexpr<Condition>(std::forward<ThenCallback>(thenCallback),
                                 [](auto) {});
#endif
}

} // namespace tv
