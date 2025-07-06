#ifndef PGENUNIQ
#define PGENUNIQ

#include <cstddef>      // std::size_t
#include <memory>       // std::unique_ptr
#include <type_traits>  // std::remove_extent
#include <utility>      // std::forward

namespace detail
{

// make_unique from C++14 standard library

template<typename T>
struct unique_if
{
  using single_object = std::unique_ptr<T>;
};

template<typename T>
struct unique_if<T[]>
{
  using unknown_bound = std::unique_ptr<T[]>;
};

template<typename T, std::size_t n>
struct unique_if<T[n]>
{
  using known_bound = void;
};

template<typename T, typename... Args>
typename unique_if<T>::single_object std_make_unique(Args&&... args)
{
  return std::unique_ptr<T>(new T{std::forward<Args>(args)...});
}

template<typename T>
typename unique_if<T>::unknown_bound std_make_unique(std::size_t n)
{
  return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]{});
}

template<typename T, typename... Args>
typename unique_if<T>::known_bound std_make_unique(Args&&...) = delete;

}

using detail::std_make_unique;

#endif
