#ifndef PGENITRP_HPP
#define PGENITRP_HPP

#include <array>        // std::array
#include <cstddef>      // std::size_t
#include <fstream>      // std::ifstream
#include <ios>          // std::ios
#include <limits>       // std::numeric_limits
#include <sstream>      // std::stringstream
#include <stdexcept>    // std::runtime_error
#include <string>       // std::string
#include <type_traits>  // std::is_floating_point, std::is_same, std::remove_cv
#include <vector>       // std::vector

namespace detail
{

// full bit mask of given length

std::size_t constexpr full_bitmask(std::size_t n)
{
  return
      n < std::numeric_limits<std::size_t>::digits
    ? (static_cast<std::size_t>(1) << n) - 1
    : -1;
}

// search using bisection
// (implemented identically as in Python standard library)

template<typename T>
std::size_t bisect(std::size_t n, T const *a, T const &x)
{
  static_assert(
    std::is_floating_point<T>::value,
    "must use floating-point numbers");

  std::size_t lo = 0, hi = n;

  while (lo < hi)
  {
    std::size_t const md = (lo+hi)/2;
    if (x < a[md])
      hi = md;
    else
      lo = md+1;
  }

  return lo;
}

// interpolate over data from file

template<typename T, std::size_t m, std::size_t n>
class GridLinearInterpolator
{
  static_assert(
    std::is_same<T, typename std::remove_cv<T>::type>::value,
    "must use a non-const, non-volatile type");
  static_assert(
    std::is_floating_point<T>::value,
    "must use a floating-point type");
  static_assert(
    m >= 1 && m <= std::numeric_limits<std::size_t>::digits,
    "source dimension is invalid");
  static_assert(
    n >= 1,
    "destination dimension is invalid");

  public:
  explicit GridLinearInterpolator(std::string const &fname);

  std::array<T, m> const &DomainMin() const;
  std::array<T, m> const &DomainMax() const;

  std::array<T, n> operator()(std::array<T, m> const &p) const;

  protected:
  std::vector<T> x[m], y;
  std::array<T, m> domain_min, domain_max;
};

template<typename T, std::size_t m, std::size_t n>
GridLinearInterpolator<T, m, n>::
  GridLinearInterpolator(std::string const &fname)
{
  std::ifstream file{fname, std::ios::in};
  if (!file)
  {
    std::stringstream msg;
    msg << "cannot open file: " << fname;
    throw std::runtime_error{msg.str()};
  }

  std::size_t prd_size = 1;
  for (std::size_t i = 0; i < m; ++i)
  {
    std::size_t size;
    if (!(file >> size))
      throw std::runtime_error{"cannot parse file: size"};
    if (size < 2)
    {
      std::stringstream msg;
      msg << "size along dimension " << i << " must be at least two";
      throw std::runtime_error{msg.str()};
    }
    x[i].resize(size);
    prd_size *= size;
  }
  y.resize(prd_size*n);

  for (std::size_t i = 0; i < m; ++i)
  {
    for (std::size_t j = 0; j < x[i].size(); ++j)
    {
      if (!(file >> x[i][j]))
        throw std::runtime_error{"cannot parse file: domain"};
      if (j > 0 && !(x[i][j-1] < x[i][j]))
      {
        std::stringstream msg;
        msg << "coordinates are not strictly increasing: "
               "!(" << x[i][j-1] << " < " << x[i][j] << ")";
        throw std::runtime_error{msg.str()};
      }
    }
    domain_min[i] = x[i].front();
    domain_max[i] = x[i].back();
  }

  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < prd_size; ++j)
      if (!(file >> y[j*n+i]))
        throw std::runtime_error{"cannot parse file: range"};
}

template<typename T, std::size_t m, std::size_t n>
std::array<T, m> const &GridLinearInterpolator<T, m, n>::
  DomainMin() const
{
  return domain_min;
}

template<typename T, std::size_t m, std::size_t n>
std::array<T, m> const &GridLinearInterpolator<T, m, n>::
  DomainMax() const
{
  return domain_max;
}

template<typename T, std::size_t m, std::size_t n>
std::array<T, n> GridLinearInterpolator<T, m, n>::
  operator()(std::array<T, m> const &p) const
{
  std::size_t indices[m];
  T ratios[m];

  for (std::size_t i = 0; i < m; ++i)
  {
    T const min = domain_min[i];
    T const max = domain_max[i];

    if (!(min <= p[i] && p[i] <= max))
    {
      std::stringstream msg;
      msg << "interpolation point is out of range: "
             "!(" << min  << " <= " << p[i] << " && "
                  << p[i] << " <= " << max  << ")";
      throw std::runtime_error{msg.str()};
    }

    indices[i] = bisect(x[i].size()-1, x[i].data(), p[i]);
    T const x_lft = x[i][indices[i]-1];
    T const x_rgt = x[i][indices[i]  ];
    ratios[i] = (p[i] - x_lft) / (x_rgt - x_lft);
  }

  std::array<T, n> result{};

  for (std::size_t b = 0; b <= full_bitmask(m); ++b)
  {
    std::size_t index = 0;
    T ratio = 1;
    for (std::size_t i = 0; i < m; ++i)
    {
      std::size_t const f = b>>i & 1;
      index *= x[i].size();
      index += indices[i] - f;
      ratio *= f ? 1-ratios[i] : ratios[i];
    }
    for (std::size_t i = 0; i < n; ++i)
      result[i] += y[index*n+i] * ratio;
  }

  return result;
}

}

using detail::GridLinearInterpolator;

#endif
