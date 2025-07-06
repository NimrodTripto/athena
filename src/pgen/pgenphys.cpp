#include <cmath>    // std::atan2, std::cos, std::exp, std::fabs, std::fmax,
                    // std::log, std::pow, std::sin, std::sqrt
#include <limits>   // std::numeric_limits
#include <memory>   // std::shared_ptr
#include <tuple>    // std::get, std::make_tuple, std::tuple
#include <utility>  // std::move
#include <mpi.h>      // only used if Athena++ was configured with MPI #

#include "../athena.hpp"

#include "pgenphys.hpp"
#include "../mesh/mesh.hpp"   // for Mesh*
#include "pgenutil.hpp"
#include "../mesh/meshblock.hpp"       // MeshBlock internals
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"              // Globals::my_rank

namespace {
using ::Real;  // Real is a typedef from athena.hpp
using Vec3 = std::array<Real,3>;

struct Loc {
  double dr, dang2;
  int rank;
};

void minloc2_fn(void *in_data, void *inout_data, int *len,
                MPI_Datatype *datatype) {
  Loc *in      = static_cast<Loc*>(in_data);
  Loc *inout   = static_cast<Loc*>(inout_data);

  for (int i = 0; i < *len; ++i) {
    if ((in[i].dr  < inout[i].dr  - 1e-12) ||
        (std::fabs(in[i].dr - inout[i].dr) < 1e-12 && in[i].dang2 < inout[i].dang2 - 1e-12)) {
      inout[i] = in[i];
    }
  }
}

/* spherical → Cartesian */
inline Vec3 Sph2Cart(Real r, Real th, Real ph)
{
  return { r*std::sin(th)*std::cos(ph),
           r*std::sin(th)*std::sin(ph),
           r*std::cos(th) };
}

/* new parabola (ℓ,m,n) → spherical, per user's formula
      x  = ½ (ℓ − m)
      y  = √(ℓ m) cos n
      z  = √(ℓ m) sin n                                         */
inline Vec3 Par2Sph(Real l, Real m, Real n)     // returns (r,θ,φ)
{
  const Real x   = 0.5*(l - m);
  const Real rho = std::sqrt(l*m);              // ≥0 if ℓ,m ≥0
  const Real y   =  rho*std::cos(n);
  const Real z   =  rho*std::sin(n);

  const Real r   = std::sqrt(x*x + y*y + z*z);
  const Real th  = std::acos(z / r);            // polar
  const Real ph  = std::atan2(y, x);            // azimuth
  return {r, th, ph};
}

inline Vec3 RotateZXY(const Vec3 &v,
                      Real pitch,   // about Y, last
                      Real roll,    // about X, middle
                      Real yaw)     // about Z, first
{
  // 1) yaw about Z
  const Real cz = std::cos(yaw), sz = std::sin(yaw);
  Real x1 =  cz*v[0] - sz*v[1];
  Real y1 =  sz*v[0] + cz*v[1];
  Real z1 =       v[2];

  // 2) roll about X
  const Real cr = std::cos(roll), sr = std::sin(roll);
  Real x2 = x1;
  Real y2 =  cr*y1 - sr*z1;
  Real z2 =  sr*y1 + cr*z1;

  // 3) pitch about Y
  const Real cp = std::cos(pitch), sp = std::sin(pitch);
  Real x3 =  cp*x2 + sp*z2;
  Real y3 =       y2;
  Real z3 = -sp*x2 + cp*z2;

  return {x3,y3,z3};
}
}

namespace Problem
{

SnapResult SnapToGrid(const ::Mesh *mesh,
                      Real pitch, Real roll, Real yaw,
                      Real src_l, Real src_m, Real src_n)
{

  std::cout<<"[debug] SnapToGrid called with mesh="<<mesh
           <<" nblocal="<<mesh->nblocal
           <<" nbtotal="<<mesh->nbtotal
           <<"\n";

  for (int b = 0; b < mesh->nblocal; ++b) {
    auto *pmb = mesh->my_blocks(b);
    // std::cout<<"   block "<<b<<" @ "<<pmb
    //          <<", pcoord="<<pmb->pcoord<<", "
    //          <<"is/ie="<<pmb->is<<"/"<<pmb->ie
    //          <<", js/je="<<pmb->js<<"/"<<pmb->je
    //          <<", ks/ke="<<pmb->ks<<"/"<<pmb->ke
    //          <<"\n";
  }

  // 1. (ℓ,m,n) → spherical
  auto src_sph  = Par2Sph(src_l, src_m, src_n);        // (r,θ,φ)
  std::cout << "Original point in spherical coords: r=" << src_sph[0] 
            << " theta=" << src_sph[1] << " phi=" << src_sph[2] << std::endl;

  // 2. spherical → Cartesian
  auto src_cart = Sph2Cart(src_sph[0], src_sph[1], src_sph[2]);
  std::cout << "Original point in Cartesian coords: x=" << src_cart[0] 
            << " y=" << src_cart[1] << " z=" << src_cart[2] << std::endl;

  // 3. rotate
  auto tgt_cart = RotateZXY(src_cart, pitch, roll, yaw);
  std::cout << "Rotated point in Cartesian coords: x=" << tgt_cart[0] 
            << " y=" << tgt_cart[1] << " z=" << tgt_cart[2] << std::endl;

  // --- 4) Search mesh: pick smallest (|r−rₜ|, then angular distance) in one pass ---
  const Real r_tgt = std::sqrt(
    tgt_cart[0]*tgt_cart[0] +
    tgt_cart[1]*tgt_cart[1] +
    tgt_cart[2]*tgt_cart[2]);
  // unit-vector of the rotated point
  std::array<Real,3> uv_tgt = {
    tgt_cart[0]/r_tgt,
    tgt_cart[1]/r_tgt,
    tgt_cart[2]/r_tgt
  };

  Real best_dr     = 1e5;
  Real best_dang2  = 1e5;
  Real best_r      = 0, best_th = 0, best_ph = 0;

  for (int b = 0; b < mesh->nblocal; ++b) {
    auto *pmb = mesh->my_blocks(b);
    auto *pc  = pmb->pcoord;

    for (int i = pmb->is; i <= pmb->ie; ++i) {
      const Real r_cell = pc->x1v(i);
      const Real dr     = std::fabs(r_cell - r_tgt);
      
      // First pass: find best radius
      if (dr < best_dr - 1e-12) {
        best_dr = dr;
        best_r = r_cell;
        // std::cout << "Found better radius: r=" << r_cell << " (dr=" << dr << ")" << std::endl;
      }
    }
  }

  // Second pass: find best angles for the best radius
  best_dang2 = 1e5;
  for (int b = 0; b < mesh->nblocal; ++b) {
    auto *pmb = mesh->my_blocks(b);
    auto *pc = pmb->pcoord;

    for (int i = pmb->is; i <= pmb->ie; ++i) {
      const Real r_cell = pc->x1v(i);
      
      // Only consider cells at the best radius
      if (std::fabs(r_cell - best_r) < 1e-8) {
        // loop over angular coords on this shell
        for (int j = pmb->js; j <= pmb->je; ++j) {
          const Real th_cell = pc->x2v(j);
          for (int k = pmb->ks; k <= pmb->ke; ++k) {
            const Real ph_cell = pc->x3v(k);
            // build unit-vector at (th_cell,ph_cell)
            auto uv = Sph2Cart(1.0, th_cell, ph_cell);
            // angular "distance" squared on unit sphere
            Real dx = uv[0] - uv_tgt[0];
            Real dy = uv[1] - uv_tgt[1];
            Real dz = uv[2] - uv_tgt[2];
            Real dang2 = dx*dx + dy*dy + dz*dz;

            if (dang2 < best_dang2) {
              best_dang2 = dang2;
              best_th = th_cell;
              best_ph = ph_cell;
            // std::cout << "Found better angles: theta=" << th_cell 
            //           << " phi=" << ph_cell << " (dang2=" << dang2 << ")" << std::endl;
            }
          }
        }
      }
    }
  }

  std::cout << "Best selected coordinates:" << std::endl;
  std::cout << "r=" << best_r << " theta=" << best_th << " phi=" << best_ph << std::endl;
  std::cout << "Angular distance squared: " << best_dang2 << std::endl;

  // e) convert winning cell back to (ℓ,m,n)
  const Real  width = src_l - src_m;
  SnapResult res;
  res.l     = best_r + 0.5*width;
  res.m     = best_r - 0.5*width;
  res.n     = best_ph;
  res.pitch = pitch;
  res.roll  = roll;
  res.yaw   = yaw + (best_ph - src_n);

  // [CHECK_PLACE] printout for consistency with tdesph.cpp, now with l,m,n
  std::cout << "[CHECK_PLACE] snapped to ("
            << res.l << "," << res.m << "," << res.n << ")\n";
  std::cout << "[CHECK_PLACE] l=" << res.l << " m=" << res.m << " n=" << res.n << "\n";
  std::cout << "[CHECK_PLACE] angles: pitch=" << pitch
            << " roll=" << roll
            << " yaw=" << yaw << "\n";

#ifdef MPI_PARALLEL
  // Pack up for a global "minloc" on the pair (best_dr, best_dang2).
  // We need a struct that holds both:
  struct Loc in_loc{ best_dr, best_dang2, Globals::my_rank }, out_loc;
  
  // Use a custom MPI op: compare first dr, then dang2 if dr ties.
  MPI_Op minloc2;
  MPI_Datatype mpi_type_Loc;
  MPI_Type_contiguous(3, MPI_DOUBLE, &mpi_type_Loc); // dr, dang2, rank (double for rank to match Loc)
  MPI_Type_commit(&mpi_type_Loc);
  MPI_Op_create(&minloc2_fn, /*commute=*/true, &minloc2);
  
  MPI_Allreduce(&in_loc, &out_loc, 1, mpi_type_Loc, minloc2, MPI_COMM_WORLD);

  // now rank == out_loc.rank owns the true best (r,th,ph).  gather or bcast:
  if (Globals::my_rank == out_loc.rank) {
    double xyz[3] = { best_r, best_th, best_ph };
    MPI_Bcast(xyz, 3, MPI_DOUBLE, out_loc.rank, MPI_COMM_WORLD);
  } else {
    double xyz[3];
    MPI_Bcast(xyz, 3, MPI_DOUBLE, out_loc.rank, MPI_COMM_WORLD);
    best_r = xyz[0];
    best_th= xyz[1];
    best_ph= xyz[2];
  }

  MPI_Op_free(&minloc2);
  MPI_Type_free(&mpi_type_Loc);
#endif

  // Convert best spherical coordinates to Cartesian
  Real x = best_r * std::sin(best_th) * std::cos(best_ph);
  Real y = best_r * std::sin(best_th) * std::sin(best_ph);
  Real z = best_r * std::cos(best_th);

  // Print coordinates
  std::cout << "Snapped cell coordinates:" << std::endl;
  std::cout << "Spherical: r=" << best_r << " theta=" << best_th << " phi=" << best_ph << std::endl;
  std::cout << "Cartesian: x=" << x << " y=" << y << " z=" << z << std::endl;

  return res;
}


//-----------------------------------------------------------------------------
// Conserved-/Primitive conversions
Conserved Primitive::ToConserved(Real adiabatic_index) const
{
  Conserved cons;
  cons.dn = dn;
  cons.m1 = dn * v1;
  cons.m2 = dn * v2;
  cons.m3 = dn * v3;
  cons.en = dn * (v1*v1 + v2*v2 + v3*v3) * 0.5
            + pr / (adiabatic_index - 1);
  return cons;
}

Real AxisymmetricGravity::operator()(
  Real r, Real theta, Real phi, SphericalCoordsTag) const
{
  return (*this)(r, theta, SphericalCoordsTag{});
}

std::tuple<Real, Real, Real> AxisymmetricGravity::Gradient(
  Real r, Real theta, Real phi, SphericalCoordsTag) const
{
  auto const temp = Gradient(r, theta, SphericalCoordsTag{});
  return std::make_tuple(std::get<0>(temp), std::get<1>(temp), 0);
}

SoftenedNewtonianGravity::SoftenedNewtonianGravity(
  Real mass, Real soft_radius)
:
  m{mass},
  s{soft_radius}
{
}

Real SoftenedNewtonianGravity::operator()(
  Real r, Real theta, SphericalCoordsTag) const
{
  Real const a = r+s;

  return -m/a;
}

std::tuple<Real, Real> SoftenedNewtonianGravity::Gradient(
  Real r, Real theta, SphericalCoordsTag) const
{
  Real const a = r+s;

  return std::make_tuple(m/(a*a), 0);
}

Real SoftenedNewtonianGravity::MidplaneSecondRadialDerivative(
  Real r, SphericalCoordsTag) const
{
  Real const a = r+s;

  return -2*m/(a*a*a);
}

PaczynskiWiitaGravity::PaczynskiWiitaGravity(
  Real mass, Real grav_radius)
:
  m{mass},
  rg{grav_radius}
{
}

Real PaczynskiWiitaGravity::operator()(
  Real r, Real theta, SphericalCoordsTag) const
{
  Real const a = r-2*rg;

  return -m/a;
}

std::tuple<Real, Real> PaczynskiWiitaGravity::Gradient(
  Real r, Real theta, SphericalCoordsTag) const
{
  Real const a = r-2*rg;

  return std::make_tuple(m/(a*a), 0);
}

Real PaczynskiWiitaGravity::MidplaneSecondRadialDerivative(
  Real r, SphericalCoordsTag) const
{
  Real const a = r-2*rg;

  return -2*m/(a*a*a);
}

UniformGas::UniformGas(Real density, Real pressure)
:
  prim{density, 0, 0, 0, pressure}
{
}

Primitive UniformGas::CalcPrimitive(
  Real r, Real theta, SphericalCoordsTag) const
{
  return prim;
}

Polytrope::Polytrope(
  std::shared_ptr<AxisymmetricGravity const> grav,
  Real center_radius, Real center_density,
  Real polytropic_index, Real polytropic_const, Real shear)
:
  grav{std::move(grav)},
  center_radius{center_radius},
  center_density{center_density},
  polytropic_index{polytropic_index},
  polytropic_const{polytropic_const},
  shear{shear},
  center_speed{
    std::sqrt(center_radius * std::get<0>(this->grav->Gradient(
      center_radius, M_PI_2, SphericalCoordsTag{})))},
  equation_const{0}
{
  equation_const = DensityEquation(
    center_radius, M_PI_2, center_density, SphericalCoordsTag{});
}

Real Polytrope::OrbitalSpeed(
  Real r, Real theta, SphericalCoordsTag) const
{
  return center_speed * std::pow(r*std::sin(theta) / center_radius, 1-shear);
}

Real Polytrope::DensityEquation(
  Real r, Real theta, Real density, SphericalCoordsTag) const
{
  Real const speed = OrbitalSpeed(r, theta, SphericalCoordsTag{});
  Real const temp1 = polytropic_index - 1;
  Real const temp2 =
      std::fabs(temp1) > 1e-8
    ? polytropic_index/temp1 * (std::pow(density, temp1) - 1)
    : std::log(density) * (polytropic_index + temp1/2 * std::log(density));

  return
      (*grav)(r, theta, SphericalCoordsTag{})
    - speed*speed / (2*(1-shear))
    + polytropic_const * temp2
    - equation_const;
}

Primitive Polytrope::CalcPrimitive(
  Real r, Real theta, SphericalCoordsTag) const
{
  Real const density = solve_brentq([&] (Real density)
    {
      return DensityEquation(r, theta, density, SphericalCoordsTag{});
    },
    std::numeric_limits<Real>::epsilon(), center_density);

  Primitive prim;

  prim.dn = density;
  prim.v1 = 0;
  prim.v2 = 0;
  prim.v3 = OrbitalSpeed(r, theta, SphericalCoordsTag{});
  prim.pr = polytropic_const * std::pow(density, polytropic_index);

  return prim;
}

ConstantSoundSpeedDisk::ConstantSoundSpeedDisk(
  std::shared_ptr<AxisymmetricGravity const> grav, Real sound_speed,
  Real (*midplane_density)(Real),
  Real (*midplane_density_power_law_index)(Real))
:
  grav{std::move(grav)},
  sound_speed_sq{sound_speed * sound_speed},
  midplane_density{midplane_density},
  midplane_density_power_law_index{midplane_density_power_law_index}
{
}

Primitive ConstantSoundSpeedDisk::CalcPrimitive(
  Real r, Real theta, SphericalCoordsTag) const
{
  Real const r_sin_theta = r*std::sin(theta);
  Real const grav_diff =
      (*grav)(r_sin_theta, M_PI_2, SphericalCoordsTag{})
    - (*grav)(r, theta, SphericalCoordsTag{});

  Real const density =
    midplane_density(r_sin_theta) * std::exp(grav_diff/sound_speed_sq);
  Real const orbital_speed_sq =
    sound_speed_sq * midplane_density_power_law_index(r_sin_theta) +
    r_sin_theta * std::get<0>(grav->Gradient(
      r_sin_theta, M_PI_2, SphericalCoordsTag{}));

  Primitive prim;

  prim.dn = density;
  prim.v1 = 0;
  prim.v2 = 0;
  prim.v3 = std::sqrt(orbital_speed_sq);
  prim.pr = density * sound_speed_sq;

  return prim;
}

ConstantAspectRatioDisk::ConstantAspectRatioDisk(
  std::shared_ptr<AxisymmetricGravity const> grav,
  Real aspect_ratio, Real midplane_density)
:
  grav{std::move(grav)},
  aspect_ratio_sq{aspect_ratio * aspect_ratio},
  midplane_density{midplane_density}
{
}

Primitive ConstantAspectRatioDisk::CalcPrimitive(
  Real r, Real theta, SphericalCoordsTag) const
{
  Real const r_sin_theta = r*std::sin(theta);
  Real const grav_diff =
      (*grav)(r_sin_theta, M_PI_2, SphericalCoordsTag{})
    - (*grav)(r, theta, SphericalCoordsTag{});

  Real const midplane_keplerian_orbital_speed_sq =
    r_sin_theta * std::get<0>(grav->Gradient(
      r_sin_theta, M_PI_2, SphericalCoordsTag{}));
  Real const sound_speed_sq =
    aspect_ratio_sq * midplane_keplerian_orbital_speed_sq;
  Real const density =
    midplane_density * std::exp(grav_diff/sound_speed_sq);
  Real const orbital_speed_sq =
    midplane_keplerian_orbital_speed_sq +
    (midplane_keplerian_orbital_speed_sq +
      r_sin_theta * r_sin_theta * grav->MidplaneSecondRadialDerivative(
        r_sin_theta, SphericalCoordsTag{})) *
    (aspect_ratio_sq - grav_diff/midplane_keplerian_orbital_speed_sq);

  Primitive prim;

  prim.dn = density;
  prim.v1 = 0;
  prim.v2 = 0;
  prim.v3 = std::sqrt(std::fmax(0, orbital_speed_sq));
  prim.pr = density * sound_speed_sq;

  return prim;
}

StraightStream::StraightStream(
  Real src_r, Real src_theta, Real src_phi,
  Real vel_r, Real vel_theta, Real vel_phi,
  Real width, Real current, Real sound_speed)
:
  src_x{src_r * std::sin(src_theta) * std::cos(src_phi)},
  src_y{src_r * std::sin(src_theta) * std::sin(src_phi)},
  src_z{src_r * std::cos(src_theta)},
  vel_x{
      vel_r     *  std::sin(src_theta) *  std::cos(src_phi)
    + vel_theta *  std::cos(src_theta) *  std::cos(src_phi)
    + vel_phi                          * -std::sin(src_phi)},
  vel_y{
      vel_r     *  std::sin(src_theta) *  std::sin(src_phi)
    + vel_theta *  std::cos(src_theta) *  std::sin(src_phi)
    + vel_phi                          *  std::cos(src_phi)},
  vel_z{
      vel_r     *  std::cos(src_theta)
    + vel_theta * -std::sin(src_theta)},
  vel_sq{vel_r*vel_r + vel_theta*vel_theta + vel_phi*vel_phi},
  width_sq{width * width},
  center_density{current / (M_PI * width_sq * std::sqrt(vel_sq))},
  sound_speed_sq{sound_speed * sound_speed}
{
}

Primitive StraightStream::CalcPrimitive(
  Real r, Real theta, Real phi, SphericalCoordsTag) const
{
  Real const sx = src_x, sy = src_y, sz = src_z;
  Real const vx = vel_x, vy = vel_y, vz = vel_z, vs = vel_sq;
  Real const ws = width_sq;

  Real const cq = std::cos(theta), sq = std::sin(theta);
  Real const cp = std::cos(phi),   sp = std::sin(phi);

  // transform from spherical to Cartesian coordinates
  Real const x = r*sq*cp;
  Real const y = r*sq*sp;
  Real const z = r*cq;

  // calculate distance squared from stream midline
  Real const tx = sx - x;
  Real const ty = sy - y;
  Real const tz = sz - z;

  Real const t = (tx*vx + ty*vy + tz*vz) / vs;
  Real const dx = tx - t*vx;
  Real const dy = ty - t*vy;
  Real const dz = tz - t*vz;
  Real const ds = dx*dx + dy*dy + dz*dz;

  Primitive prim;

  // create a stream with circular cross section and uniform velocity
  prim.dn = center_density * std::exp(-ds/ws);
  prim.v1 = vx*sq* cp + vy*sq*sp + vz* cq;
  prim.v2 = vx*cq* cp + vy*cq*sp + vz*-sq;
  prim.v3 = vx*   -sp + vy*   cp;
  prim.pr = prim.dn * sound_speed_sq;

  return prim;
}

ParabolicStream::ParabolicStream(
  Real grav_mass,
  Real grav_radius,
  Real pitch, Real roll, Real yaw,
  Real src_l, Real src_m, Real src_n,
  Real wid_l, Real wid_m, Real wid_n,
  Real current, Real sound_speed, Real density_mult, Real vel_coeff, InjectionMode mode)
:
  grav_mass{grav_mass},
  grav_radius{grav_radius},
  cos_pitch{std::cos(pitch)}, sin_pitch{std::sin(pitch)},
  cos_roll{std::cos(roll)},   sin_roll{std::sin(roll)},
  cos_yaw{std::cos(yaw)},     sin_yaw{std::sin(yaw)},
  src_l{src_l}, src_m{src_m}, src_n{src_n},
  wid_l{wid_l}, wid_m{wid_m}, wid_n{wid_n},
  current{current},
  sound_speed_sq{sound_speed * sound_speed},
  density_mult{density_mult},
  vel_coeff{vel_coeff},
  mode{mode}
{
}

std::tuple<Real, Real, Real> ParabolicStream::CalcStreamCoords(
  Real r, Real theta, Real phi, SphericalCoordsTag) const
{
  Real const ci = cos_pitch,       si = sin_pitch;
  Real const cj = cos_roll,        sj = sin_roll;
  Real const ck = cos_yaw,         sk = sin_yaw;
  Real const cq = std::cos(theta), sq = std::sin(theta);
  Real const cp = std::cos(phi),   sp = std::sin(phi);

  // transform from spherical to Cartesian coordinates
  Real const x = r*sq*cp;
  Real const y = r*sq*sp;
  Real const z = r*cq;

  // transform from global to stream-aligned Cartesian coordinates, the latter
  // better suited to describing stream geometry, by rotating through pitch
  // angle about y-axis, then through roll angle about rotated x-axis, then
  // through yaw angle about rotated z-axis
  Real const s = x*( ci*ck+si*sj*sk) + y*( cj*sk) + z*(-si*ck+ci*sj*sk);
  Real const t = x*(-ci*sk+si*sj*ck) + y*( cj*ck) + z*( si*sk+ci*sj*ck);
  Real const u = x*( si*cj         ) + y*(-sj   ) + z*( ci*cj         );

  // transform from stream-aligned Cartesian to parabolic rotational
  // coordinates
  Real const l = r + s;
  Real const m = r - s;
  Real const n = std::atan2(u, t);

  return std::make_tuple(l, m, n);
}

std::tuple<Real, Real, Real> ParabolicStream::CalcStreamTangent(
  Real r, Real theta, Real phi, SphericalCoordsTag) const
{
  Real const ci = cos_pitch,       si = sin_pitch;
  Real const cj = cos_roll,        sj = sin_roll;
  Real const ck = cos_yaw,         sk = sin_yaw;
  Real const cq = std::cos(theta), sq = std::sin(theta);
  Real const cp = std::cos(phi),   sp = std::sin(phi);

  // transform from spherical to parabolic rotational coordinates
  auto const temp1 = CalcStreamCoords(r, theta, phi, SphericalCoordsTag{});
  Real const l = std::get<0>(temp1);
  Real const m = std::get<1>(temp1);
  Real const n = std::get<2>(temp1);

  // calculate unit tangent to coordinate surface of parabolic rotational
  // coordinates with components in stream-aligned Cartesian coordinates
  Real const temp2 = 1 / std::sqrt(l*(l+m));
  Real const ts = temp2 * -std::sqrt(l*m);
  Real const tt = temp2 * l*std::cos(n);
  Real const tu = temp2 * l*std::sin(n);

  // transform tangent from stream-aligned to global Cartesian coordinates
  Real const tx = ts*( ci*ck+si*sj*sk) + tt*(-ci*sk+si*sj*ck) + tu*( si*cj);
  Real const ty = ts*( cj*sk         ) + tt*( cj*ck         ) + tu*(-sj   );
  Real const tz = ts*(-si*ck+ci*sj*sk) + tt*( si*sk+ci*sj*ck) + tu*( ci*cj);

  // transform tangent from Cartesian to spherical coordinates
  Real const tr = tx*sq* cp + ty*sq*sp + tz* cq;
  Real const tq = tx*cq* cp + ty*cq*sp + tz*-sq;
  Real const tp = tx*   -sp + ty*   cp;


  return std::make_tuple(tr, tq, tp);
}

Primitive ParabolicStream::CalcPrimitive(
  Real r, Real theta, Real phi, SphericalCoordsTag) const
{
  // transform from spherical to parabolic rotational coordinates
  auto const temp1 = CalcStreamCoords(r, theta, phi, SphericalCoordsTag{});
  Real const l = std::get<0>(temp1);
  Real const m = std::get<1>(temp1);
  Real const n = std::get<2>(temp1);

  // calculate unit tangent to coordinate surface of parabolic rotational
  // coordinates with components in spherical coordinates
  auto const temp2 = CalcStreamTangent(r, theta, phi, SphericalCoordsTag{});
  Real const tr = std::get<0>(temp2);
  Real const tq = std::get<1>(temp2);
  Real const tp = std::get<2>(temp2);

  // calculate velocity of marginally unbound stream
  // use user-supplied coefficient instead of fixed 2.01
  Real const v = std::sqrt(vel_coeff * grav_mass / ((l + m) / 2));

  // calculate scaled displacement from injection point
  Real const lx = (l-src_l) / wid_l;
  Real const mx = (m-src_m) / wid_m;
  Real const nx = (n-src_n) / wid_n;

  // set stream properties such that stream cross section is elliptical;
  // covariant metric in parabolic rotational coordinates is diag((l+m)/(4*l),
  // (l+m)/(4*m), l*m)
  // Real const center_density =
  //     mode == InjectionMode::Area
  //   ? // current / (M_PI * wid_l * wid_n *
  //     // std::sqrt((l+m)/(4*l)) * std::sqrt(l*m) * v) //
  //     current * M_1_PI / (wid_l * wid_n * std::sqrt(m))
  //   : // current / (std::pow(M_PI, 3/2.) * wid_l * wid_m * wid_n *
  //     // std::sqrt((l+m)/(4*l)) * std::sqrt((l+m)/(4*m)) * std::sqrt(l*m))
  //     current / (std::pow(M_PI, 3/2.) * wid_l * wid_m * wid_n * (l+m)/4);
  Real const base_density =
    mode == InjectionMode::Area
    ? current * M_1_PI / (wid_l * wid_n * std::sqrt(m))
    : current / (std::pow(M_PI, 3/2.) * wid_l * wid_m * wid_n * (l+m)/4);
  Real const center_density = density_mult * base_density;

  Primitive prim;

  prim.dn = center_density * std::exp(-(lx*lx + mx*mx + nx*nx));
  prim.v1 = v*tr;
  prim.v2 = v*tq;
  prim.v3 = v*tp;
  prim.pr = prim.dn * sound_speed_sq;

  return prim;
}

}
