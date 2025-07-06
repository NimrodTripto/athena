#include <fenv.h>

#include <cmath>  // std::exp, std::hypot, std::pow, std::sqrt

#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#if COORINDATE_SYSTEM != cartesian
  #error "problem requires Cartesian coordinates"
#endif

class Conserved
{
  public:
  Real d, m1, m2, m3, e;
};

class Primitive
{
  public:
  Real d, v1, v2, v3, p;

  Conserved ToConserved(Real adiabatic_index) const;
};

class Disk
{
  protected:
  const Real aspect_ratio;
  const Real density_floor;
  const Real pressure_floor;

  public:
  explicit Disk(Real aspect_ratio, Real density_floor, Real pressure_floor);
  const Primitive CalcPrimitive(Real z) const;
};

class Stream
{
  protected:
  const Real radius;
  const Real center_density;
  const Real sound_speed_sq;
  const Real density_floor;
  const Real pressure_floor;

  public:
  explicit Stream(Real radius, Real center_density, Real sound_speed,
    Real density_floor, Real pressure_floor);
  const Primitive CalcPrimitive(Real r) const;
};

static void sc_gravity(
  MeshBlock *mb, Real t, Real dt, AthenaArray<Real> const &w,
  AthenaArray<Real> const &w_scalar, AthenaArray<Real> const &bcc,
  AthenaArray<Real> &u, AthenaArray<Real> &u_scalar);
static void bc_inflow_inner_x2(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng);
static void bc_outflow_inner_x3(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng);
static void bc_inject_outer_x3(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng);

static inline Real grav_accel(Real z);
static inline Real grav_pot(Real z);

static Real adiabatic_index;
static Disk *disk;
static Stream *stream;

void Mesh::InitUserMeshData(ParameterInput *in)
{
  // feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

  adiabatic_index = in->GetReal("hydro", "gamma");

  disk = new Disk(
    in->GetReal("disk", "aspect_ratio"),
    in->GetReal("hydro", "dfloor"),
    in->GetReal("hydro", "pfloor")
  );
  stream = new Stream(
    in->GetReal("stream", "radius"),
    in->GetReal("stream", "center_density"),
    in->GetReal("stream", "sound_speed"),
    in->GetReal("hydro", "dfloor"),
    in->GetReal("hydro", "pfloor")
  );

  EnrollUserExplicitSourceFunction(sc_gravity);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x2, bc_inflow_inner_x2);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x3, bc_outflow_inner_x3);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x3, bc_inject_outer_x3);
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *in)
{
}

void MeshBlock::ProblemGenerator(ParameterInput *in)
{
  for (int k = ks; k <= ke; k++)
    for (int j = js; j <= je; j++)
      for (int i = is; i <= ie; i++)
      {
        const Real z = pcoord->x3v(k);

        const Conserved cons =
          disk->CalcPrimitive(z).ToConserved(adiabatic_index);

        phydro->u(IDN, k, j, i) = cons.d;
        phydro->u(IM1, k, j, i) = cons.m1;
        phydro->u(IM2, k, j, i) = cons.m2;
        phydro->u(IM3, k, j, i) = cons.m3;
        phydro->u(IEN, k, j, i) = cons.e;
      }
}

void MeshBlock::UserWorkInLoop()
{
}

void Mesh::UserWorkAfterLoop(ParameterInput *in)
{
  delete disk, stream;
}

Conserved Primitive::ToConserved(Real adiabatic_index) const
{
  Conserved cons;

  cons.d = d;
  cons.m1 = d*v1;
  cons.m2 = d*v2;
  cons.m3 = d*v3;
  cons.e = d*(v1*v1+v2*v2+v3*v3)/2 + p/(adiabatic_index-1);

  return cons;
}

inline Disk::Disk(Real aspect_ratio, Real density_floor, Real pressure_floor)
:
  aspect_ratio(aspect_ratio),
  density_floor(density_floor),
  pressure_floor(pressure_floor)
{
}

const Primitive Disk::CalcPrimitive(Real z) const
{
  Primitive prim;

  const Real sound_speed_sq = aspect_ratio * aspect_ratio;

  prim.d = std::exp(-grav_pot(z)/sound_speed_sq);
  prim.v1 = 0;
  prim.v2 = 1;
  prim.v3 = 0;
  prim.p = prim.d * sound_speed_sq;

  if (prim.d < density_floor && prim.p < pressure_floor)
  {
    prim.d = density_floor;
    prim.p = pressure_floor;
  }
  else if (prim.d < density_floor)
  {
    prim.d = density_floor;
    prim.p = density_floor * sound_speed_sq;
  }
  else if (prim.p < pressure_floor)
  {
    prim.d = pressure_floor / sound_speed_sq;
    prim.p = pressure_floor;
  }

  return prim;
}

inline Stream::Stream(Real radius, Real center_density, Real sound_speed,
  Real density_floor, Real pressure_floor)
:
  radius(radius),
  center_density(center_density),
  sound_speed_sq(sound_speed * sound_speed),
  density_floor(density_floor),
  pressure_floor(pressure_floor)
{
}

const Primitive Stream::CalcPrimitive(Real r) const
{
  Primitive prim;

  prim.d = center_density * std::exp(-r*r/(radius*radius));
  prim.v1 = 0;
  prim.v2 = 0;
  prim.v3 = -std::sqrt(2);
  prim.p = prim.d * sound_speed_sq;

  if (prim.d < density_floor && prim.p < pressure_floor)
  {
    prim.d = density_floor;
    prim.p = pressure_floor;
  }
  else if (prim.d < density_floor)
  {
    prim.d = density_floor;
    prim.p = density_floor * sound_speed_sq;
  }
  else if (prim.p < pressure_floor)
  {
    prim.d = pressure_floor / sound_speed_sq;
    prim.p = pressure_floor;
  }

  return prim;
}

void sc_gravity(
  MeshBlock *mb, Real t, Real dt, AthenaArray<Real> const &w,
  AthenaArray<Real> const &w_scalar, AthenaArray<Real> const &bcc,
  AthenaArray<Real> &u, AthenaArray<Real> &u_scalar)
{
  const int is = mb->is, ie = mb->ie;
  const int js = mb->js, je = mb->je;
  const int ks = mb->ks, ke = mb->ke;

  for (int k = ks; k <= ke; k++)
  {
    const Real z = mb->pcoord->x3v(k);

    for (int j = js; j <= je; j++)
      for (int i = is; i <= ie; i++)
      {
        const Real dm3 = dt * w(IDN, k, j, i) * grav_accel(z);

        u(IM3, k, j, i) += dm3;
        if (NON_BAROTROPIC_EOS)
          u(IEN, k, j, i) += dm3 * w(IVZ, k, j, i);
      }
  }
}

void bc_inflow_inner_x2(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &u, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  for (int k = ks; k <= ke; k++)
  {
    const Real z = mb->pcoord->x3v(k);

    const Primitive disk_prim = disk->CalcPrimitive(z);

    for (int i = is; i <= ie; i++)
      for (int j = js-ng; j <= js-1; j++)
      {
        u(IDN, k, j, i) = disk_prim.d;
        u(IVX, k, j, i) = disk_prim.v1;
        u(IVY, k, j, i) = disk_prim.v2;
        u(IVZ, k, j, i) = disk_prim.v3;
        u(IPR, k, j, i) = disk_prim.p;
      }
  }
}

void bc_outflow_inner_x3(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &u, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  for (int j = js; j <= je; j++)
    for (int i = is; i <= ie; i++)
    {
      const Real z0 = mb->pcoord->x3v(ks);

      const Real d0 = u(IDN, ks, j, i);
      const Real sound_speed_sq = u(IPR, ks, j, i) / d0;

      for (int k = ks-ng; k <= ks-1; k++)
      {
        const Real z = mb->pcoord->x3v(k);

        const Real temp = grav_pot(z0) - grav_pot(z);
        const Real density = d0 * std::exp(temp/sound_speed_sq);

        u(IDN, k, j, i) = density;
        u(IVX, k, j, i) = u(IVX, ks, j, i);
        u(IVY, k, j, i) = u(IVY, ks, j, i);
        u(IVZ, k, j, i) = u(IVZ, ks, j, i);
        u(IPR, k, j, i) = density * sound_speed_sq;
      }
    }
}

void bc_inject_outer_x3(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &u, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  for (int j = js; j <= je; j++)
    for (int i = is; i <= ie; i++)
    {
      const Real x = mb->pcoord->x1v(i);
      const Real y = mb->pcoord->x2v(j);
      const Real r = std::hypot(x, y);

      const Real d0 = u(IDN, ke, j, i);
      const Primitive stream_prim = stream->CalcPrimitive(r);

      if (stream_prim.d > d0)
        for (int k = ke+1; k <= ke+ng; k++)
        {
          u(IDN, k, j, i) = stream_prim.d;
          u(IVX, k, j, i) = stream_prim.v1;
          u(IVY, k, j, i) = stream_prim.v2;
          u(IVZ, k, j, i) = stream_prim.v3;
          u(IPR, k, j, i) = stream_prim.p;
        }
      else
      {
        const Real z0 = mb->pcoord->x3v(ke);

        const Real sound_speed_sq = u(IPR, ke, j, i) / d0;

        for (int k = ke+1; k <= ke+ng; k++)
        {
          const Real z = mb->pcoord->x3v(k);

          const Real temp = grav_pot(z0) - grav_pot(z);
          const Real density = d0 * std::exp(temp/sound_speed_sq);

          u(IDN, k, j, i) = density;
          u(IVX, k, j, i) = u(IVX, ke, j, i);
          u(IVY, k, j, i) = u(IVY, ke, j, i);
          u(IVZ, k, j, i) = u(IVZ, ke, j, i);
          u(IPR, k, j, i) = density * sound_speed_sq;
        }
      }
    }
}

Real grav_accel(Real z)
{
  return -z/std::pow(1+z*z, 3/2.);
}

Real grav_pot(Real z)
{
  return 1-1/std::hypot(1, z);
}
