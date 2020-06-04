#ifndef MP_HELPER_H_
#define MP_HELPER_H_
#include <type_traits>
#include <utility>

namespace tv {
template <class... T> struct mp_list {};

template <class T, T... I>
using mp_list_c = mp_list<std::integral_constant<T, I>...>;

namespace detail {

template <class... Ts, class F>
constexpr F mp_for_each_impl(mp_list<Ts...>, F &&f) {
  return (void)(std::initializer_list<int>{(f(Ts()), 0)...}),
         std::forward<F>(f);
}

template <class F> constexpr F mp_for_each_impl(mp_list<>, F &&f) {
  return std::forward<F>(f);
}

} // namespace detail

template <class... T>
using mp_length = std::integral_constant<std::size_t, sizeof...(T)>;

namespace detail {

template <class A, template <class...> class B> struct mp_rename_impl {
  // An error "no type named 'type'" here means that the first argument to
  // mp_rename is not a list
};

template <template <class...> class A, class... T, template <class...> class B>
struct mp_rename_impl<A<T...>, B> {
  using type = B<T...>;
};

} // namespace detail

template <class A, template <class...> class B>
using mp_rename = typename detail::mp_rename_impl<A, B>::type;

template <class L> using mp_size = mp_rename<L, mp_length>;

template <class L, class F> constexpr F mp_for_each(F &&f) {
  return detail::mp_for_each_impl(mp_rename<L, mp_list>(), std::forward<F>(f));
}
} // namespace tv

#endif