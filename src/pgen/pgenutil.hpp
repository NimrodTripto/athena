#ifndef PGENUTIL_HPP
#define PGENUTIL_HPP

#include <cmath>        // std::fabs, std::fmax, std::isfinite, std::pow
#include <limits>       // std::numeric_limits
#include <tuple>        // std::make_tuple, std::tuple
#include <type_traits>  // std::is_floating_point, std::is_same, std::result_of

#include "../athena.hpp"

// generate random number between -1 and 1, both ends inclusive
Real rand_sym();

// generate random number deterministically from coordinates
Real rand_from_coords(Real x1, Real x2, Real x3, int precision = 20);

// given point and line passing through origin, calculate signed projection of
// point onto line and perpendicular distance from point to line
std::tuple<Real, Real> calc_point_line_proj_perp(
  Real px, Real py, Real pz, Real lx, Real ly, Real lz);

// calculate numerical gradient by finite difference
template<typename F, typename T>
std::tuple<T, T> calc_gradient(F const &f, T x, T y)
{
  static_assert(
    std::is_floating_point<T>::value,
    "must use floating-point numbers");
  static_assert(
    std::is_same<
      T,
      typename std::result_of<F &&(T &&, T &&)>::type
    >::value,
    "function must accept two parameters and return a value of the same type");

  // finite-differencing step
  T const e = std::pow(std::numeric_limits<T>::epsilon(), 1/3.);
  T const dx = (1+std::fabs(x))*e, dy = (1+std::fabs(y))*e;

  return std::make_tuple(
    (f(x+dx, y) - f(x-dx, y)) / (2*dx),
    (f(x, y+dy) - f(x, y-dy)) / (2*dy)
  );
}

// solve equation using Brent's method with inverse quadratic interpolation
template<typename F, typename T>
T solve_brentq(F const &f, T a, T b)
{
  static_assert(
    std::is_floating_point<T>::value,
    "must use floating-point numbers");
  static_assert(
    std::is_same<
      T,
      typename std::result_of<F &&(T &&)>::type
    >::value,
    "function must accept one parameter and return a value of the same type");

  // maximum number of iterations
  int const max_iter = 256;
  // machine epsilon
  T const e = std::numeric_limits<T>::epsilon();

  // last and second-to-last values of b
  T c = a, d = std::numeric_limits<T>::quiet_NaN();
  // function evaluated at a, b, c
  T fa = f(a), fb = f(b), fc = fa;
  // whether last step used bisection
  bool last_bisect = true;

  // ensure zero is bracketed
  if (fa * fb > 0)
    return std::numeric_limits<T>::quiet_NaN();

  for (int i = 0; i < max_iter; ++i)
  {
    // leave loop if variables are not finite
    if (!(
      std::isfinite( a) && std::isfinite( b) &&
      std::isfinite(fa) && std::isfinite(fb)))
      break;

    // leave loop if converged
    if (fb == 0 || std::fabs(b-a) < (1+std::fabs(a)) * e)
      return b;

    if (std::fabs(fa) < std::fabs(fb))
    {
      // swap a and b so that b is closer to solution
      T const temp1 =  a;  a =  b;  b = temp1;
      T const temp2 = fa; fa = fb; fb = temp2;
    }

    T x;
    if (fa != fc && fb != fc)
    {
      // perform inverse quadratic interpolation
      // x = a * fb * fc / (fa-fb) / (fa-fc) +
      //     b * fc * fa / (fb-fc) / (fb-fa) +
      //     c * fa * fb / (fc-fa) / (fc-fb);
      T const s = fb / fa, t = fa / fc;
      T const p = s * (t*t * (1-s) * (b-c) + (1-s*t) * (a-b));
      T const q = (s-1) * (t-1) * (s*t-1);
      x = b + p/q;
    }
    else
      // perform linear interpolation (secant method)
      x = b - (b-a) * fb / (fb-fa);

    if (
      // interpolation point does not lie within acceptance range
      (x-(3*a+b)/4) * (x-b) > 0 ||
      // interpolation step is small
      std::fabs(last_bisect ? b-c : c-d) < std::fmax(e, 2*std::fabs(x-b)))
    {
      // perform bisection
      x = (a+b) / 2;
      last_bisect = true;
    }
    else
      // use interpolation
      last_bisect = false;

    // shift previous values of b
    d = c; c = b; fc = fb;

    // assign new estimate to a or b to keep zero bracketed
    T const fx = f(x);
    if (fa * fx > 0)
    {
      a = x; fa = fx;
    }
    else
    {
      b = x; fb = fx;
    }
  }

  // signal lack of convergence
  return std::numeric_limits<T>::quiet_NaN();
}

#endif
