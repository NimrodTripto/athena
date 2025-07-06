#include <cmath>    // std::fmax, std::round, std::sqrt
#include <cstdlib>  // std::rand, std::srand
#include <tuple>    // std::make_tuple, std::tuple

#include "pgenutil.hpp"

Real rand_sym()
{
  Real const p = static_cast<Real>(std::rand());
  Real const q = static_cast<Real>(RAND_MAX);
  return p/q*2-1;
}

Real rand_from_coords(Real x1, Real x2, Real x3, int precision)
{
  // snap coordinates to grid of coarser resolution
  Real const s1 = std::round(x1 * (1 << precision));
  Real const s2 = std::round(x2 * (1 << precision));
  Real const s3 = std::round(x3 * (1 << precision));

  // produce integers from restricted coordinates
  unsigned int const i1 = static_cast<unsigned int>(s1);
  unsigned int const i2 = static_cast<unsigned int>(s2);
  unsigned int const i3 = static_cast<unsigned int>(s3);

  // use such integers to seed random number generator
  std::srand(i1 ^ i2 ^ i3);

  return rand_sym();
}

std::tuple<Real, Real> calc_point_line_proj_perp(
  Real px, Real py, Real pz, Real lx, Real ly, Real lz)
{
  Real const p2 = px*px + py*py + pz*pz;
  Real const l2 = lx*lx + ly*ly + lz*lz;
  Real const pl = px*lx + py*ly + pz*lz;

  Real const proj = pl/std::sqrt(l2);
  // Cauchy--Schwarz inequality justifies clipping to zero
  Real const perp = std::sqrt(std::fmax(0, p2-proj*proj));

  return std::make_tuple(proj, perp);
}
