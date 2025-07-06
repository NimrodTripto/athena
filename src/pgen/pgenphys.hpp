#ifndef PGENPHYS_HPP
#define PGENPHYS_HPP

#pragma once

#include <memory>  // std::shared_ptr
#include <tuple>   // std::tuple

#include <array>
#include <cmath>
#include <limits>

#include "../athena.hpp"
#include "../globals.hpp"   // for Globals::my_rank
#include "../mesh/mesh.hpp"        // ← add, or forward-declare class Mesh


namespace Problem
{
 //
// // -------------------------------------------------------------------------
// // 1)  Result types
// // -------------------------------------------------------------------------
struct SnapCell   { Real r,   th,  ph;  };          // "winning" cell centre
struct SnapResult { Real l, m, n;                   // snapped (ℓ,m,n) //
                    Real pitch, roll, yaw; };       // attitude actually used
  
// // ----------------- Cartesian helpers ----------------------------------------
// inline std::array<Real,3> Sph2Cart(Real r, Real th, Real ph)
// {
//   return { r*std::sin(th)*std::cos(ph),
//            r*std::sin(th)*std::sin(ph),
//            r*std::cos(th) };
// }

// // ------------------------------------------------------------------
// //  Desmos parabola → spherical
// inline std::array<Real,3> Par2Sph(Real l, Real m, Real n)
// {
//   const Real x    = 0.5*(l*l + m*m)/(l + m);
//   const Real rho2 = l*m*(l - m)/(l + m);
//   const Real rho  = std::sqrt(std::max<Real>(rho2, 0.0));
//   const Real y    = -rho*std::cos(n);
//   const Real z    =  rho*std::sin(n);

//   const Real r  = std::sqrt(x*x + y*y + z*z);
//   const Real th = std::acos(z / r);
//   const Real ph = std::atan2(y, x);
//   return {r, th, ph};
// }

// // ------------------------------------------------------------------
// //  Y-X-Z rotation: pitch (Y) → roll (X) → yaw (Z)
// inline std::array<Real,3> RotateYXZ(const std::array<Real,3>& v,
//                                     Real pitch, Real roll, Real yaw)
// {
//   // pitch
//   const Real cp = std::cos(pitch), sp = std::sin(pitch);
//   Real x1 =  cp*v[0] + sp*v[2];
//   Real y1 =            v[1];
//   Real z1 = -sp*v[0] + cp*v[2];

//   // roll
//   const Real cr = std::cos(roll), sr = std::sin(roll);
//   Real x2 = x1;
//   Real y2 =  cr*y1 - sr*z1;
//   Real z2 =  sr*y1 + cr*z1;

//   // yaw
//   const Real cy = std::cos(yaw), sy = std::sin(yaw);
//   return { cy*x2 - sy*y2,
//            sy*x2 + cy*y2,
//            z2 };
// }

// -------------------------------------------------------------------------
// 3)  SnapToGrid – template lives ENTIRELY in the header
// -------------------------------------------------------------------------

// #ifdef __cplusplus
// class Mesh;                          // forward declare; full type in .cpp
// #endif

// -------------------------------------------------------------------------
//  SnapToGrid – **now a single concrete function (no template)**
// -------------------------------------------------------------------------
SnapResult SnapToGrid(const ::Mesh *mesh,
                      Real pitch, Real roll, Real yaw,
                      Real src_l, Real src_m, Real src_n);

// 333
struct SphericalCoordsTag {};

// conserved variables
class Conserved
{
  public:
  Real dn, m1, m2, m3, en;
};

// primitive variables
class Primitive
{
  public:
  Real dn, v1, v2, v3, pr;

  Conserved ToConserved(Real adiabatic_index) const;
};

// gravity
class Gravity
{
  public:
  virtual void SetTime(Real t) {}
  virtual Real operator()(
    Real r, Real theta, Real phi, SphericalCoordsTag) const = 0;
  virtual std::tuple<Real, Real, Real> Gradient(
    Real r, Real theta, Real phi, SphericalCoordsTag) const = 0;

  Gravity()                           = default;
  Gravity(Gravity &&)                 = default;
  Gravity(Gravity const &)            = default;
  Gravity& operator=(Gravity &&)      = default;
  Gravity& operator=(Gravity const &) = default;
  virtual ~Gravity()                  = default;
};

// axisymmetric gravity
class AxisymmetricGravity : public Gravity
{
  public:
  virtual Real operator()(
    Real r, Real theta, SphericalCoordsTag) const = 0;
  virtual std::tuple<Real, Real> Gradient(
    Real r, Real theta, SphericalCoordsTag) const = 0;
  virtual Real MidplaneSecondRadialDerivative(
    Real r, SphericalCoordsTag) const = 0;
  Real operator()(
    Real r, Real theta, Real phi, SphericalCoordsTag) const override;
  std::tuple<Real, Real, Real> Gradient(
    Real r, Real theta, Real phi, SphericalCoordsTag) const override;

  AxisymmetricGravity()                                       = default;
  AxisymmetricGravity(AxisymmetricGravity &&)                 = default;
  AxisymmetricGravity(AxisymmetricGravity const &)            = default;
  AxisymmetricGravity& operator=(AxisymmetricGravity &&)      = default;
  AxisymmetricGravity& operator=(AxisymmetricGravity const &) = default;
  virtual ~AxisymmetricGravity()                              = default;
};

// axisymmetric gas
class AxisymmetricGas
{
  public:
  virtual Primitive CalcPrimitive(
    Real r, Real theta, SphericalCoordsTag) const = 0;

  AxisymmetricGas()                                   = default;
  AxisymmetricGas(AxisymmetricGas &&)                 = default;
  AxisymmetricGas(AxisymmetricGas const &)            = default;
  AxisymmetricGas& operator=(AxisymmetricGas &&)      = default;
  AxisymmetricGas& operator=(AxisymmetricGas const &) = default;
  virtual ~AxisymmetricGas()                          = default;
};

// stream
class Stream
{
  public:
  virtual Primitive CalcPrimitive(
    Real r, Real theta, Real phi, SphericalCoordsTag) const = 0;

  Stream()                          = default;
  Stream(Stream &&)                 = default;
  Stream(Stream const &)            = default;
  Stream& operator=(Stream &&)      = default;
  Stream& operator=(Stream const &) = default;
  virtual ~Stream()                 = default;
};

// softened Newtonian gravity
class SoftenedNewtonianGravity final : public AxisymmetricGravity
{
  public:
  using AxisymmetricGravity::operator();
  using AxisymmetricGravity::Gradient;
  explicit SoftenedNewtonianGravity(
    Real mass, Real soft_radius);
  Real operator()(
    Real r, Real theta, SphericalCoordsTag) const override;
  std::tuple<Real, Real> Gradient(
    Real r, Real theta, SphericalCoordsTag) const override;
  Real MidplaneSecondRadialDerivative(
    Real r, SphericalCoordsTag) const override;

  protected:
  Real const m;
  Real const s;
};

// Newtonian description of Schwarzschild gravity [1980A&A....88...23P]
class PaczynskiWiitaGravity final : public AxisymmetricGravity
{
  public:
  using AxisymmetricGravity::operator();
  using AxisymmetricGravity::Gradient;
  explicit PaczynskiWiitaGravity(
    Real mass, Real grav_radius);
  Real operator()(
    Real r, Real theta, SphericalCoordsTag) const override;
  std::tuple<Real, Real> Gradient(
    Real r, Real theta, SphericalCoordsTag) const override;
  Real MidplaneSecondRadialDerivative(
    Real r, SphericalCoordsTag) const override;

  protected:
  Real const m;
  Real const rg;
};

/*
class HernquistGravity final : public AxisymmetricGravity
class NavarroFrenkWhiteGravity final : public AxisymmetricGravity
class SechSquaredGravity final : public AxisymmetricGravity
class GalaxyGravity final : public AxisymmetricGravity
*/

// uniform static gas
class UniformGas final : public AxisymmetricGas
{
  public:
  explicit UniformGas(Real density, Real pressure);
  Primitive CalcPrimitive(
    Real r, Real theta, SphericalCoordsTag) const override;

  protected:
  Primitive const prim;
};

// polytropic gas with power-law rotational profile over cylindrical radius
class Polytrope final : public AxisymmetricGas
{
  public:
  explicit Polytrope(
    std::shared_ptr<AxisymmetricGravity const> grav,
    Real center_radius, Real center_density,
    Real polytropic_index, Real polytropic_const, Real shear);
  Primitive CalcPrimitive(
    Real r, Real theta, SphericalCoordsTag) const override;

  protected:
  Real OrbitalSpeed(
    Real r, Real theta, SphericalCoordsTag) const;
  Real DensityEquation(
    Real r, Real theta, Real density, SphericalCoordsTag) const;

  std::shared_ptr<AxisymmetricGravity const> const grav;
  Real const center_radius;
  Real const center_density;
  Real const polytropic_index;
  Real const polytropic_const;
  Real const shear;
  Real const center_speed;
  Real equation_const;
};

// gas with constant sound speed and provided midplane density profile
class ConstantSoundSpeedDisk final : public AxisymmetricGas
{
  public:
  explicit ConstantSoundSpeedDisk(
    std::shared_ptr<AxisymmetricGravity const> grav, Real sound_speed,
    Real (*midplane_density)(Real),
    Real (*midplane_density_power_law_index)(Real));
  Primitive CalcPrimitive(
    Real r, Real theta, SphericalCoordsTag) const override;

  protected:
  std::shared_ptr<AxisymmetricGravity const> const grav;
  Real const sound_speed_sq;
  Real (*const midplane_density)(Real);
  Real (*const midplane_density_power_law_index)(Real);
};

// gas with constant aspect ratio and midplane density, and with inner cavity
class ConstantAspectRatioDisk final : public AxisymmetricGas
{
  public:
  explicit ConstantAspectRatioDisk(
    std::shared_ptr<AxisymmetricGravity const> grav,
    Real aspect_ratio, Real midplane_density);
  Primitive CalcPrimitive(
    Real r, Real theta, SphericalCoordsTag) const override;

  protected:
  std::shared_ptr<AxisymmetricGravity const> const grav;
  Real const aspect_ratio_sq;
  Real const midplane_density;
};

/*
class SechSquaredDisk final : public AxisymmetricStructure
*/

class StraightStream final : public Stream
{
  public:
  explicit StraightStream(
    Real src_r, Real src_theta, Real src_phi,
    Real vel_r, Real vel_theta, Real vel_phi,
    Real width, Real current, Real sound_speed);
  Primitive CalcPrimitive(
    Real r, Real theta, Real phi, SphericalCoordsTag) const override;

  protected:
  Real const src_x, src_y, src_z;
  Real const vel_x, vel_y, vel_z, vel_sq;
  Real const width_sq;
  Real const center_density;
  Real const sound_speed_sq;
};

// stream with streamlines along paraboloid surfaces
class ParabolicStream final : public Stream
{
  public:
  enum class InjectionMode { Area, Volume };

  explicit ParabolicStream(
    Real grav_mass,
    Real grav_radius,
    Real pitch, Real roll, Real yaw,
    Real src_l, Real src_m, Real src_n,
    Real wid_l, Real wid_m, Real wid_n,
    Real current, Real sound_speed,
    Real density_mult,
    Real vel_coeff,
    InjectionMode mode);
  std::tuple<Real, Real, Real> CalcStreamCoords(
    Real r, Real theta, Real phi, SphericalCoordsTag) const;
  std::tuple<Real, Real, Real> CalcStreamTangent(
    Real r, Real theta, Real phi, SphericalCoordsTag) const;
  Primitive CalcPrimitive(
    Real r, Real theta, Real phi, SphericalCoordsTag) const override;

  protected:
  Real const grav_mass;
  Real const grav_radius;
  Real const cos_pitch, sin_pitch;
  Real const cos_roll, sin_roll;
  Real const cos_yaw, sin_yaw;
  Real const src_l, src_m, src_n;
  Real const wid_l, wid_m, wid_n;
  Real const current;
  Real const sound_speed_sq;
  Real const density_mult;
  Real const vel_coeff;
  InjectionMode const mode;
};

}

#endif
