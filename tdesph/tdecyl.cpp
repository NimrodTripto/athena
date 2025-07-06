#include <fenv.h>

#include <cmath>      // std::cos, std::cosh, std::exp, std::fabs, std::fmax,
                      // std::fmin, std::hypot, std::log, std::log1p, std::pow,
                      // std::sin, std::sqrt, std::tanh
#include <fstream>    // std::ifstream
#include <ios>        // std::streamsize
#include <iostream>   // std::cout
#include <limits>     // std::numeric_limits
#include <ostream>    // std::endl
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string

#include "../athena.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#if GALPOT_GRAVITY
  #include "../galpot/GalPot.h"
#endif

#if COORINDATE_SYSTEM != cylindrical
  #error "problem requires cylindrical coordinates"
#endif

// false: use uniform density and pressure floors
// true:  use density and pressure of hydrostatic polytrope as floors
static const bool use_vacuum = true;

// 0: ghost zone and last physical cell have the same angular velocity
// 1: ghost zone and last physical cell have the same velocity
// 2: ghost zone and last physical cell have the same angular momentum
static const int bc_vel_mode = 1;

// false: directly copy density and pressure into ghost zone
// true:  adjust density and pressure in ghost zone to make force zero
static const int bc_zero_force = true;

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

class AxisymmetricGravity
{
  public:
  virtual Real operator()(Real r, Real z) const = 0;
  virtual Real RadialGradient(Real r, Real z) const = 0;
  virtual Real RadialGradient2(Real r, Real z) const = 0;
  virtual Real VerticalGradient(Real r, Real z) const = 0;
};

class Stream
{
  public:
  virtual Primitive CalcPrimitive(Real r, Real p, Real z) const = 0;
};

class AxisymmetricStructure
{
  public:
  virtual Primitive CalcPrimitive(Real r, Real z) const = 0;
};

class PointMassGravity : public AxisymmetricGravity
{
  protected:
  const Real mass;

  public:
  explicit PointMassGravity(Real mass);
  Real operator()(Real r, Real z) const override;
  Real RadialGradient(Real r, Real z) const override;
  Real RadialGradient2(Real r, Real z) const override;
  Real VerticalGradient(Real r, Real z) const override;
};

class HernquistGravity : public AxisymmetricGravity
{
  protected:
  const Real potential_scale;
  const Real scale_radius;

  public:
  explicit HernquistGravity(Real mass, Real scale_radius);
  Real operator()(Real r, Real z) const override;
  Real RadialGradient(Real r, Real z) const override;
  Real RadialGradient2(Real r, Real z) const override;
  Real VerticalGradient(Real r, Real z) const override;
};

class NavarroFrenkWhiteGravity : public AxisymmetricGravity
{
  protected:
  const Real potential_scale;
  const Real scale_radius;

  public:
  explicit NavarroFrenkWhiteGravity(Real virial_mass, Real virial_radius,
    Real concentration);
  Real operator()(Real r, Real z) const override;
  Real RadialGradient(Real r, Real z) const override;
  Real RadialGradient2(Real r, Real z) const override;
  Real VerticalGradient(Real r, Real z) const override;
};

class SechSquaredGravity : public AxisymmetricGravity
{
  protected:
  const Real potential_scale;
  const Real scale_radius;
  const Real inner_radius;
  const Real height_scale;

  public:
  explicit SechSquaredGravity(Real surface_density, Real scale_radius,
    Real inner_radius, Real scale_height);
  Real operator()(Real r, Real z) const override;
  Real RadialGradient(Real r, Real z) const override;
  Real RadialGradient2(Real r, Real z) const override;
  Real VerticalGradient(Real r, Real z) const override;
};

class GalaxyGravity : public AxisymmetricGravity
{
  protected:
  const NavarroFrenkWhiteGravity dark_matter_pot;
  const PointMassGravity bulge_pot;
  const SechSquaredGravity thin_stellar_disk_pot;
  const SechSquaredGravity thick_stellar_disk_pot;

  public:
  GalaxyGravity();
  Real operator()(Real r, Real z) const override;
  Real RadialGradient(Real r, Real z) const override;
  Real RadialGradient2(Real r, Real z) const override;
  Real VerticalGradient(Real r, Real z) const override;

  static const Real length_scale;
  static const Real mass_scale;
};

#if GALPOT_GRAVITY
class McMillanGravity : public AxisymmetricGravity
{
  protected:
  const GalaxyPotential *proxy;

  public:
  explicit McMillanGravity(const char *fname);
  Real operator()(Real r, Real z) const override;
  Real RadialGradient(Real r, Real z) const override;
  Real RadialGradient2(Real r, Real z) const override;
  Real VerticalGradient(Real r, Real z) const override;
  ~McMillanGravity();

  static const Real length_scale;
  Real velocity_scale;
};
#endif

class Polytrope : public AxisymmetricStructure
{
  protected:
  const Real center_radius;
  const Real center_density;
  const Real center_orbital_speed;
  const Real polytropic_index;
  const Real polytropic_const;
  const Real shear;
  const AxisymmetricGravity *const grav_pot;

  Real OrbitalSpeed(Real r) const;
  Real DensityEquation(Real r, Real z, Real density) const;
  Real equation_const;

  public:
  explicit Polytrope(Real center_radius, Real center_density,
    Real polytropic_index, Real polytropic_const, Real shear,
    const AxisymmetricGravity *grav_pot);
  Primitive CalcPrimitive(Real r, Real z) const override;
};

class ConstantSoundSpeedDisk : public AxisymmetricStructure
{
  protected:
  Real (*const midplane_density)(Real);
  Real (*const midplane_density_power_law_index)(Real);
  const Real sound_speed_sq;
  const AxisymmetricGravity *const grav_pot;

  public:
  explicit ConstantSoundSpeedDisk(Real (*midplane_density)(Real),
    Real (*midplane_density_power_law_index)(Real),
    Real sound_speed, const AxisymmetricGravity *grav_pot);
  Primitive CalcPrimitive(Real r, Real z) const override;
};

class ConstantAspectRatioDisk : public AxisymmetricStructure
{
  protected:
  const Real midplane_density;
  const Real aspect_ratio_sq;
  const Real inner_radius;
  const AxisymmetricGravity *const grav_pot;

  public:
  explicit ConstantAspectRatioDisk(Real midplane_density, Real aspect_ratio,
    Real inner_radius, const AxisymmetricGravity *grav_pot);
  Primitive CalcPrimitive(Real r, Real z) const override;
};

class SechSquaredDisk : public AxisymmetricStructure
{
  protected:
  const Real density_scale;
  const Real scale_radius;
  const Real inner_radius;
  const Real height_scale;
  const Real sound_speed_sq_scale;

  public:
  explicit SechSquaredDisk(Real surface_density, Real scale_radius,
    Real inner_radius, Real scale_height);
  Primitive CalcPrimitive(Real r, Real z) const override;
};

class StraightStream : public Stream
{
  protected:
  const Real ax, ay, az;
  const Real vx, vy, vz;
  const Real width_sq;
  const Real sound_speed_sq;
  const Real velocity_sq;
  const Real center_density;

  public:
  explicit StraightStream(Real ar, Real ap, Real az, Real vr, Real vp, Real vz,
    Real width, Real mass_flux, Real sound_speed);
  Primitive CalcPrimitive(Real r, Real p, Real z) const override;
};

class ParabolicStream : public Stream
{
  protected:
  const Real center_semilatus_rectum;
  const Real pitch;
  const Real roll;
  const Real yaw;
  const Real width;
  const Real mass_flux;
  const Real sound_speed_sq;

  public:
  explicit ParabolicStream(Real center_semilatus_rectum, Real pitch, Real roll,
    Real yaw, Real width, Real mass_flux, Real sound_speed);
  Primitive CalcPrimitive(Real r, Real p, Real z) const override;
};

static void sc_gravity(
  MeshBlock *mb, Real t, Real dt, AthenaArray<Real> const &w,
  AthenaArray<Real> const &w_scalar, AthenaArray<Real> const &bcc,
  AthenaArray<Real> &u, AthenaArray<Real> &u_scalar);
static void bc_outflow_inner_x1(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng);
static void bc_outflow_outer_x1(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng);
static void bc_outflow_inner_x3(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng);
static void bc_inject_outer_x3(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng);
static Real bc_sound_speed_sq_min_mult;
static Real bc_sound_speed_sq_min_mult_ix1;

static void enforce_floor(MeshBlock *mb);

static Real adiabatic_index;
static AxisymmetricGravity *grav_pot;
static AxisymmetricStructure *disk, *vacuum;
static Stream *stream;

static Real galaxy_midplane_density(Real r);
static Real galaxy_midplane_density_power_law_index(Real r);

template<typename T, typename F>
static T solve_brentq(const F &f, T a, T b);

void Mesh::InitUserMeshData(ParameterInput *in)
{
  // feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

  adiabatic_index = in->GetReal("hydro", "gamma");

  const std::string disk_type = in->GetString("disk", "type");
  if (disk_type == "polytrope")
  {
    grav_pot = new PointMassGravity(1);
    disk = new Polytrope(
      in->GetReal("disk", "center_radius"),
      in->GetReal("disk", "center_density"),
      in->GetReal("disk", "polytropic_index"),
      in->GetReal("disk", "polytropic_const"),
      in->GetReal("disk", "shear"),
      grav_pot
    );
  }
  else if (disk_type == "const_aspect")
  {
    grav_pot = new PointMassGravity(1);
    disk = new ConstantAspectRatioDisk(
      in->GetReal("disk", "midplane_density"),
      in->GetReal("disk", "aspect_ratio"),
      in->GetReal("disk", "inner_radius"),
      grav_pot
    );
  }
  else if (disk_type == "sech_squared")
  {
    disk = new SechSquaredDisk(
      in->GetReal("disk", "surface_density"),
      in->GetReal("disk", "scale_radius"),
      in->GetReal("disk", "inner_radius"),
      in->GetReal("disk", "scale_height")
    );
    grav_pot = new SechSquaredGravity(
      in->GetReal("disk", "surface_density"),
      in->GetReal("disk", "scale_radius"),
      in->GetReal("disk", "inner_radius"),
      in->GetReal("disk", "scale_height")
    );
  }
  #if GALPOT_GRAVITY
  else if (disk_type == "galaxy")
  {
    const Real sound_speed = in->GetReal("disk", "sound_speed");

    // grav_pot = new GalaxyGravity();
    grav_pot = new McMillanGravity(in->GetString("disk", "potential").c_str());
    disk = new ConstantSoundSpeedDisk(
      galaxy_midplane_density,
      galaxy_midplane_density_power_law_index,
      sound_speed,
      grav_pot
    );

    if (false && Globals::my_rank == 0)
    {
      const int r_samples = 1001;
      const int z_samples = 2001;
      const Real dr = 10. / (r_samples-1);
      const Real dz = 20. / (z_samples-1) * sound_speed;

      Real mass = 0;
      for (int i = 0; i < r_samples; i++)
        for (int k = 0; k < z_samples; k++)
        {
          const Real r = i*dr;
          const Real z = k*dz - 10*sound_speed;

          mass += disk->CalcPrimitive(r, z).d * r;
        }
      mass *= 2*M_PI * dr*dz;

      const std::streamsize p = std::cout.precision();
      std::cout.precision(std::numeric_limits<Real>::max_digits10);
      std::cout << "Mass: " << mass << std::endl;
      std::cout.precision(p);
    }
  }
  #endif
  else
    throw std::runtime_error(
      "### FATAL ERROR in Problem Generator\n"
      "disk type must be in "
      "[\"polytrope\", \"const_aspect\", \"sech_squared\", \"galaxy\"]");

  const std::string stream_type = in->GetString("stream", "type");
  if (stream_type == "straight")
    stream = new StraightStream(
      in->GetReal("stream", "ar"),
      in->GetReal("stream", "ap"),
      in->GetReal("stream", "az"),
      in->GetReal("stream", "vr"),
      in->GetReal("stream", "vp"),
      in->GetReal("stream", "vz"),
      in->GetReal("stream", "width"),
      in->GetReal("stream", "mass_flux"),
      in->GetReal("stream", "sound_speed")
    );
  else if (stream_type == "parabolic")
    stream = new ParabolicStream(
      in->GetReal("stream", "center_semilatus_rectum"),
      in->GetReal("stream", "pitch"),
      in->GetReal("stream", "roll"),
      in->GetReal("stream", "yaw"),
      in->GetReal("stream", "width"),
      in->GetReal("stream", "mass_flux"),
      in->GetReal("stream", "sound_speed")
    );

  vacuum = new Polytrope(
    in->GetReal("vacuum", "center_radius"),
    in->GetReal("vacuum", "center_density"),
    in->GetReal("vacuum", "polytropic_index"),
    in->GetReal("vacuum", "polytropic_const"),
    in->GetReal("vacuum", "shear"),
    grav_pot
  );

  bc_sound_speed_sq_min_mult =
    in->GetReal("bc", "sound_speed_min_mult");
  bc_sound_speed_sq_min_mult_ix1 =
    in->GetReal("bc", "sound_speed_min_mult_ix1");
  bc_sound_speed_sq_min_mult     *= bc_sound_speed_sq_min_mult;
  bc_sound_speed_sq_min_mult_ix1 *= bc_sound_speed_sq_min_mult_ix1;

  EnrollUserExplicitSourceFunction(sc_gravity);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, bc_outflow_inner_x1);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, bc_outflow_outer_x1);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x3, bc_outflow_inner_x3);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x3, bc_inject_outer_x3);
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *in)
{
}

void MeshBlock::ProblemGenerator(ParameterInput *in)
{
  for (int k = ks; k <= ke; k++)
    for (int i = is; i <= ie; i++)
    {
      const Real r = pcoord->x1v(i);
      const Real z = pcoord->x3v(k);

      const Primitive *prim;
      const Primitive disk_prim = disk->CalcPrimitive(r, z);
      if (use_vacuum)
      {
        const Primitive vacuum_prim = vacuum->CalcPrimitive(r, z);
        prim = disk_prim.d > vacuum_prim.d ? &disk_prim : &vacuum_prim;
      }
      else
        prim = &disk_prim;
      const Conserved cons = prim->ToConserved(adiabatic_index);

      for (int j = js; j <= je; j++)
      {
        phydro->w(IDN, k, j, i) = prim->d;
        phydro->w(IVX, k, j, i) = prim->v1;
        phydro->w(IVY, k, j, i) = prim->v2;
        phydro->w(IVZ, k, j, i) = prim->v3;
        phydro->w(IPR, k, j, i) = prim->p;

        phydro->u(IDN, k, j, i) = cons.d;
        phydro->u(IM1, k, j, i) = cons.m1;
        phydro->u(IM2, k, j, i) = cons.m2;
        phydro->u(IM3, k, j, i) = cons.m3;
        phydro->u(IEN, k, j, i) = cons.e;
      }
    }
}

void MeshBlock::UserWorkInLoop()
{
  if (use_vacuum)
    enforce_floor(this);
}

void Mesh::UserWorkAfterLoop(ParameterInput *in)
{
  delete grav_pot, disk, stream, vacuum;
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

inline PointMassGravity::PointMassGravity(Real mass)
:
  mass(mass)
{
}

inline Real PointMassGravity::operator()(Real r, Real z) const
{
  return -mass / std::hypot(r, z);
}

inline Real PointMassGravity::RadialGradient(Real r, Real z) const
{
  return mass * r * std::pow(r*r+z*z, -3/2.);
}

inline Real PointMassGravity::RadialGradient2(Real r, Real z) const
{
  return mass * (-2*r*r+z*z) * std::pow(r*r+z*z, -5/2.);
}

inline Real PointMassGravity::VerticalGradient(Real r, Real z) const
{
  return mass * z * std::pow(r*r+z*z, -3/2.);
}

inline HernquistGravity::HernquistGravity(Real mass, Real scale_radius)
:
  potential_scale(mass / scale_radius),
  scale_radius(scale_radius)
{
}

inline Real HernquistGravity::operator()(Real r, Real z) const
{
  const Real a = scale_radius;
  const Real d = std::hypot(r, z)/a;

  return -potential_scale / (1+d);
}

inline Real HernquistGravity::RadialGradient(Real r, Real z) const
{
  const Real a = scale_radius;
  const Real d = std::hypot(r, z)/a;
  const Real p = r/a;

  return potential_scale * p/(a*d*(1+d)*(1+d));
}

inline Real HernquistGravity::RadialGradient2(Real r, Real z) const
{
  const Real a = scale_radius;
  const Real d = std::hypot(r, z)/a;
  const Real p = r/a;

  return potential_scale / (a*a*d*(1+d)*(1+d))*(1-p*p*(1+3*d)/(d*d*(1+d)));
}

inline Real HernquistGravity::VerticalGradient(Real r, Real z) const
{
  const Real a = scale_radius;
  const Real d = std::hypot(r, z)/a;
  const Real q = z/a;

  return potential_scale * q/(a*d*(1+d)*(1+d));
}

inline NavarroFrenkWhiteGravity::NavarroFrenkWhiteGravity(Real virial_mass,
  Real virial_radius, Real concentration)
:
  potential_scale(virial_mass / (virial_radius *
    (std::log1p(concentration)/concentration - 1/(1+concentration)))),
  scale_radius(virial_radius / concentration)
{
}

inline Real NavarroFrenkWhiteGravity::operator()(Real r, Real z) const
{
  const Real a = scale_radius;
  const Real d = std::hypot(r, z)/a;

  return -potential_scale * std::log1p(d)/d;
}

inline Real NavarroFrenkWhiteGravity::RadialGradient(Real r, Real z) const
{
  const Real a = scale_radius;
  const Real d = std::hypot(r, z)/a;
  const Real p = r/a;

  return potential_scale * p/(a*d*d)*(std::log1p(d)/d-1/(1+d));
}

inline Real NavarroFrenkWhiteGravity::RadialGradient2(Real r, Real z) const
{
  const Real a = scale_radius;
  const Real d = std::hypot(r, z)/a;
  const Real p = r/a;

  return potential_scale / (a*a*d*d*d*d)*
    ((3+4*d)*p*p/((1+d)*(1+d))-d*d/(1+d)+(d*d-3*p*p)*std::log1p(d)/d);
}

inline Real NavarroFrenkWhiteGravity::VerticalGradient(Real r, Real z) const
{
  const Real a = scale_radius;
  const Real d = std::hypot(r, z)/a;
  const Real q = z/a;

  return potential_scale * q/(a*d*d)*(std::log1p(d)/d-1/(1+d));
}

inline SechSquaredGravity::SechSquaredGravity(Real surface_density,
  Real scale_radius, Real inner_radius, Real scale_height)
:
  potential_scale(4*M_PI * surface_density * scale_height),
  scale_radius(scale_radius),
  inner_radius(inner_radius),
  height_scale(scale_height * 2)
{
}

inline Real SechSquaredGravity::operator()(Real r, Real z) const
{
  const Real rd = scale_radius;
  const Real rm = inner_radius;
  const Real h = height_scale;

  return potential_scale * std::exp(-rm/r-r/rd) * std::log(std::cosh(z/h));
}

inline Real SechSquaredGravity::RadialGradient(Real r, Real z) const
{
  throw std::runtime_error(
    "### FATAL ERROR in Gravitational Potential\n"
    "radial derivative is undefined");
}

inline Real SechSquaredGravity::RadialGradient2(Real r, Real z) const
{
  throw std::runtime_error(
    "### FATAL ERROR in Gravitational Potential\n"
    "second radial derivative is undefined");
}

inline Real SechSquaredGravity::VerticalGradient(Real r, Real z) const
{
  const Real rd = scale_radius;
  const Real rm = inner_radius;
  const Real h = height_scale;

  return potential_scale * std::exp(-rm/r-r/rd) * std::tanh(z/h)/h;
}

// McMillan (2017), Table 3 [2017MNRAS.465...76M]
// v_0^2 R_0 / G
const Real GalaxyGravity::length_scale = 8.21;
const Real GalaxyGravity::mass_scale = 1.0372098091908592e11;

inline GalaxyGravity::GalaxyGravity()
:
  // McMillan (2017), Table 3 [2017MNRAS.465...76M]
  // https://github.com/PaulMcMillan-Astro/GalPot/blob/master/pot/PJM17_best.Tpot
  dark_matter_pot(
    1.286851153040796e12 / mass_scale,
    223.5191285249729 / length_scale,
    223.5191285249729 / 19.5725),
  bulge_pot(
    8.873113723354372e9 / mass_scale),
  thin_stellar_disk_pot(
    8.95679e8 / mass_scale * length_scale * length_scale,
    2.49955 / length_scale,
    0,
    0.3 / length_scale),
  thick_stellar_disk_pot(
    1.83444e8 / mass_scale * length_scale * length_scale,
    3.02134 / length_scale,
    0,
    0.9 / length_scale)
{
}

inline Real GalaxyGravity::operator()(Real r, Real z) const
{
  return
    dark_matter_pot(r, z) +
    bulge_pot(r, z) +
    thin_stellar_disk_pot(r, z) +
    thick_stellar_disk_pot(r, z);
}

inline Real GalaxyGravity::RadialGradient(Real r, Real z) const
{
  return
    dark_matter_pot.RadialGradient(r, z) +
    bulge_pot.RadialGradient(r, z) +
    thin_stellar_disk_pot.RadialGradient(r, z) +
    thick_stellar_disk_pot.RadialGradient(r, z);
}

inline Real GalaxyGravity::RadialGradient2(Real r, Real z) const
{
  return
    dark_matter_pot.RadialGradient2(r, z) +
    bulge_pot.RadialGradient2(r, z) +
    thin_stellar_disk_pot.RadialGradient2(r, z) +
    thick_stellar_disk_pot.RadialGradient2(r, z);
}

inline Real GalaxyGravity::VerticalGradient(Real r, Real z) const
{
  return
    dark_matter_pot.VerticalGradient(r, z) +
    bulge_pot.VerticalGradient(r, z) +
    thin_stellar_disk_pot.VerticalGradient(r, z) +
    thick_stellar_disk_pot.VerticalGradient(r, z);
}

#if GALPOT_GRAVITY
// McMillan (2017), Table 3 [2017MNRAS.465...76M] (units: kpc, Myr)
// const Real McMillanGravity::length_scale = 8.21;
// const Real McMillanGravity::velocity_scale = 233.1 / 977.7922216807892;

// galaxy at redshift 2
const Real McMillanGravity::length_scale = 3/8. * std::sqrt(7*4);

inline McMillanGravity::McMillanGravity(const char *fname)
:
  velocity_scale(1)
{
  std::ifstream file(fname);
  proxy = new GalaxyPotential(file);
  file.close();

  velocity_scale = std::sqrt(RadialGradient(1, 0));
}

inline McMillanGravity::~McMillanGravity()
{
  delete proxy;
}

inline Real McMillanGravity::operator()(Real r, Real z) const
{
  const Real temp = (*proxy)(r*length_scale, z*length_scale);
  return temp / (velocity_scale*velocity_scale);
}

inline Real McMillanGravity::RadialGradient(Real r, Real z) const
{
  Real temp1, temp2;
  (*proxy)(r*length_scale, z*length_scale, temp1, temp2);
  return temp1 * length_scale / (velocity_scale*velocity_scale);
}

inline Real McMillanGravity::RadialGradient2(Real r, Real z) const
{
  throw std::runtime_error(
    "### FATAL ERROR in Gravitational Potential\n"
    "second radial derivative is undefined");
}

inline Real McMillanGravity::VerticalGradient(Real r, Real z) const
{
  Real temp1, temp2;
  (*proxy)(r*length_scale, z*length_scale, temp1, temp2);
  return temp2 * length_scale / (velocity_scale*velocity_scale);
}
#endif

inline Polytrope::Polytrope(Real center_radius, Real center_density,
  Real polytropic_index, Real polytropic_const, Real shear,
  const AxisymmetricGravity *grav_pot)
:
  center_radius(center_radius),
  center_density(center_density),
  center_orbital_speed(
    std::sqrt(center_radius * grav_pot->RadialGradient(center_radius, 0))),
  polytropic_index(polytropic_index),
  polytropic_const(polytropic_const),
  shear(shear),
  equation_const(0),
  grav_pot(grav_pot)
{
  equation_const = DensityEquation(center_radius, 0, center_density);
}

inline Real Polytrope::OrbitalSpeed(Real r) const
{
  return center_orbital_speed * std::pow(r/center_radius, 1-shear);
}

Real Polytrope::DensityEquation(Real r, Real z, Real density) const
{
  const Real speed = OrbitalSpeed(r);
  const Real temp = polytropic_index - 1;

  return
    ((*grav_pot)(r, z) - speed*speed/(2*(1-shear))) / polytropic_const +
    (
      std::fabs(temp) > 1e-8
      ?
        polytropic_index/temp * std::pow(density, temp)
      :
        std::log(density) * (polytropic_index + temp/2 * std::log(density))
    )
    - equation_const;
}

Primitive Polytrope::CalcPrimitive(Real r, Real z) const
{
  Primitive prim;

  const Real density = solve_brentq(
    [=] (Real density) { return DensityEquation(r, z, density); },
    std::numeric_limits<Real>::epsilon(), center_density);

  prim.d = density;
  prim.v1 = 0;
  prim.v2 = OrbitalSpeed(r);
  prim.v3 = 0;
  prim.p = polytropic_const * std::pow(density, polytropic_index);

  return prim;
}

inline ConstantSoundSpeedDisk::ConstantSoundSpeedDisk(
  Real (*midplane_density)(Real),
  Real (*midplane_density_power_law_index)(Real),
  Real sound_speed, const AxisymmetricGravity *grav_pot)
:
  midplane_density(midplane_density),
  midplane_density_power_law_index(midplane_density_power_law_index),
  sound_speed_sq(sound_speed * sound_speed),
  grav_pot(grav_pot)
{
}

Primitive ConstantSoundSpeedDisk::CalcPrimitive(Real r, Real z) const
{
  const Real grav_pot_diff = (*grav_pot)(r, 0) - (*grav_pot)(r, z);

  const Real density =
    midplane_density(r) * std::exp(grav_pot_diff/sound_speed_sq);

  const Real orbital_speed_sq =
    sound_speed_sq * midplane_density_power_law_index(r) +
    r * grav_pot->RadialGradient(r, 0);

  Primitive prim;

  prim.d  = density;
  prim.v1 = 0;
  prim.v2 = std::sqrt(orbital_speed_sq);
  prim.v3 = 0;
  prim.p  = prim.d * sound_speed_sq;

  return prim;
}

inline ConstantAspectRatioDisk::ConstantAspectRatioDisk(Real midplane_density,
  Real aspect_ratio, Real inner_radius, const AxisymmetricGravity *grav_pot)
:
  midplane_density(midplane_density),
  aspect_ratio_sq(aspect_ratio * aspect_ratio),
  inner_radius(inner_radius),
  grav_pot(grav_pot)
{
}

Primitive ConstantAspectRatioDisk::CalcPrimitive(Real r, Real z) const
{
  const Real mp_kep_orbital_speed_sq = r * grav_pot->RadialGradient(r, 0);
  const Real sound_speed_sq = aspect_ratio_sq * mp_kep_orbital_speed_sq;

  Real grav_pot_diff;

  if (r > inner_radius)
  {
    grav_pot_diff = (*grav_pot)(r, 0) - (*grav_pot)(r, z);
  }
  else
  {
    const Real d = std::hypot(inner_radius-r, z/r*inner_radius);
    grav_pot_diff =
      (*grav_pot)(inner_radius, 0) - (*grav_pot)(inner_radius, d);
  }

  const Real density =
    midplane_density * std::exp(grav_pot_diff/sound_speed_sq);

  const Real orbital_speed_sq =
    mp_kep_orbital_speed_sq +
    (mp_kep_orbital_speed_sq + r*r*grav_pot->RadialGradient2(r, 0)) *
    (aspect_ratio_sq - grav_pot_diff/mp_kep_orbital_speed_sq);

  Primitive prim;

  prim.d  = density;
  prim.v1 = 0;
  prim.v2 = std::sqrt(orbital_speed_sq);
  prim.v3 = 0;
  prim.p  = prim.d * sound_speed_sq;

  return prim;
}

inline SechSquaredDisk::SechSquaredDisk(Real surface_density,
  Real scale_radius, Real inner_radius, Real scale_height)
:
  density_scale(surface_density / (4 * scale_height)),
  scale_radius(scale_radius),
  inner_radius(inner_radius),
  height_scale(scale_height * 2),
  sound_speed_sq_scale(2*M_PI * surface_density * scale_height)
{
}

Primitive SechSquaredDisk::CalcPrimitive(Real r, Real z) const
{
  const Real rd = scale_radius;
  const Real rm = inner_radius;
  const Real h = height_scale;

  const Real temp1 = std::exp(-rm/r-r/rd);
  const Real temp2 = std::cosh(z/h);

  Primitive prim;

  prim.d  = density_scale * temp1/(temp2*temp2);
  prim.v1 = 0;
  prim.v2 = std::sqrt(r * grav_pot->RadialGradient(r, 0));
  prim.v3 = 0;
  prim.p  = prim.d * temp1 * sound_speed_sq_scale;

  return prim;
}

inline StraightStream::StraightStream(Real ar, Real ap, Real az, Real vr,
  Real vp, Real vz, Real width, Real mass_flux, Real sound_speed)
:
  ax(ar*std::cos(ap)),
  ay(ar*std::sin(ap)),
  az(az),
  vx(vr*std::cos(ap) - vp*std::sin(ap)),
  vy(vr*std::sin(ap) + vp*std::cos(ap)),
  vz(vz),
  width_sq(width * width),
  sound_speed_sq(sound_speed * sound_speed),
  velocity_sq(vx*vx+vy*vy+vz*vz),
  center_density(mass_flux / (M_PI * width_sq * std::sqrt(velocity_sq)))
{
}

Primitive StraightStream::CalcPrimitive(Real r, Real p, Real z) const
{
  // transform from cylindrical to Cartesian coordinate system
  const Real x = r*std::cos(p);
  const Real y = r*std::sin(p);

  // calculate distance squared from stream midline
  const Real sx = ax-x, sy = ay-y, sz = az-z;

  const Real t = (sx*vx+sy*vy+sz*vz) / velocity_sq;
  const Real dx = sx-t*vx, dy = sy-t*vy, dz = sz-t*vz;
  const Real dist_sq = dx*dx+dy*dy+dz*dz;

  Primitive prim;

  // create a stream with circular cross section and uniform velocity
  prim.d  = center_density * std::exp(-dist_sq/width_sq);
  prim.v1 =  vx*std::cos(p) + vy*std::sin(p);
  prim.v2 = -vx*std::sin(p) + vy*std::cos(p);
  prim.v3 = vz;
  prim.p  = prim.d * sound_speed_sq;

  return prim;
}

inline ParabolicStream::ParabolicStream(Real center_semilatus_rectum,
  Real pitch, Real roll, Real yaw, Real width, Real mass_flux,
  Real sound_speed)
:
  center_semilatus_rectum(center_semilatus_rectum),
  pitch(pitch),
  roll(roll),
  yaw(yaw),
  width(width),
  mass_flux(mass_flux),
  sound_speed_sq(sound_speed * sound_speed)
{
}

Primitive ParabolicStream::CalcPrimitive(Real r, Real p, Real z) const
{
  const Real i = pitch;
  const Real j = roll;
  const Real k = yaw;
  const Real lc = center_semilatus_rectum;
  const Real ws = width * width;

  const Real ci = std::cos(i), si = std::sin(i);
  const Real cj = std::cos(j), sj = std::sin(j);
  const Real ck = std::cos(k), sk = std::sin(k);

  // transform from cylindrical to Cartesian coordinate system
  const Real x = r*std::cos(p);
  const Real y = r*std::sin(p);

  // transform from Cartesian to stream-plane coordinate system by rotating
  // about y-axis, then about rotated x-axis, then about rotated z-axis
  const Real s =
    x*( ci*ck+si*sj*sk) + y*( cj*sk) + z*(-si*ck+ci*sj*sk);
  const Real t =
    x*(-ci*sk+si*sj*ck) + y*( cj*ck) + z*( si*sk+ci*sj*ck);
  const Real u =
    x*( si*cj         ) + y*(-sj   ) + z*( ci*cj         );

  // transform from stream-plane to modified parabolic coordinate system whose
  // coordinates are semilatera recta of coordinate curves
  const Real l =  s+std::hypot(s, t);
  const Real m = -s+std::hypot(s, t);
  // calculate semilatus rectum difference from stream midlines
  const Real dl = l-lc;

  // calculate unit tangent to parabola
  const Real tn = std::hypot(t, l);
  const Real ts = -t/tn;
  const Real tt = l/tn;
  const Real tu = 0;

  // transform tangent to Cartesian coordinate system
  const Real tx =
    ts*( ci*ck+si*sj*sk) + tt*(-ci*sk+si*sj*ck) + tu*( si*cj);
  const Real ty =
    ts*( cj*sk         ) + tt*( cj*ck         ) + tu*(-sj   );
  const Real tz =
    ts*(-si*ck+ci*sj*sk) + tt*( si*sk+ci*sj*ck) + tu*( ci*cj);

  // transform tangent to cylindrical coordinate system
  const Real tr =  tx*std::cos(p) + ty*std::sin(p);
  const Real tp = -tx*std::sin(p) + ty*std::cos(p);

  // calculate velocity of marginally unbound stream
  const Real v = std::sqrt(2) * std::pow(s*s+t*t, -1/4.);

  Primitive prim;

  const Real center_density = mass_flux * std::sqrt(lc/(lc+m)) / (M_PI*ws*v);

  // set stream properties such that stream cross section is circular at
  // pericenter if vertical gravity and gas pressure were ignored
  prim.d  = center_density * std::exp(-(dl*dl/4+u*u)/ws);
  prim.v1 = v*tr;
  prim.v2 = v*tp;
  prim.v3 = v*tz;
  prim.p  = prim.d * sound_speed_sq;

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
    for (int i = is; i <= ie; i++)
    {
      const Real r = mb->pcoord->x1v(i);
      const Real z = mb->pcoord->x3v(k);

      for (int j = js; j <= je; j++)
      {
        const Real temp = -dt * w(IDN, k, j, i);
        const Real dm1 = temp * grav_pot->RadialGradient(r, z);
        const Real dm3 = temp * grav_pot->VerticalGradient(r, z);

        u(IM1, k, j, i) += dm1;
        u(IM3, k, j, i) += dm3;
        if (NON_BAROTROPIC_EOS)
          u(IEN, k, j, i) += dm1 * w(IVX, k, j, i) + dm3 * w(IVZ, k, j, i);
      }
    }
}

void bc_outflow_inner_x1(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  for (int k = ks; k <= ke; k++)
    for (int j = js; j <= je; j++)
    {
      const Real r0 = mb->pcoord->x1v(is);
      const Real z  = mb->pcoord->x3v(k);

      const Real d0 = w(IDN, k, j, is);
      const Real v0 = w(IVY, k, j, is);
      const Real sound_speed_sq = std::fmax(
        w(IPR, k, j, is) / d0,
        bc_sound_speed_sq_min_mult_ix1 * -(*grav_pot)(r0, z));

      for (int i = is-ng; i <= is-1; i++)
      {
        const Real r = mb->pcoord->x1v(i);

        Real grav_pot_diff = (*grav_pot)(r0, z) - (*grav_pot)(r, z);
        if (bc_vel_mode == 0)
        {
          grav_pot_diff += v0*v0/2*(r*r/(r0*r0)-1);
          w(IVY, k, j, i) = v0/r0*r;
        }
        else if (bc_vel_mode == 1)
        {
          grav_pot_diff += v0*v0*std::log(r/r0);
          w(IVY, k, j, i) = v0;
        }
        else if (bc_vel_mode == 2)
        {
          grav_pot_diff += v0*v0/2*(1-r0*r0/(r*r));
          w(IVY, k, j, i) = v0*r0/r;
        }
        else
          throw std::runtime_error(
            "### FATAL ERROR in Boundary Conditions\n"
            "bc_vel_mode must be in [0, 1, 2]");

        if (bc_zero_force)
        {
          const Real density = d0 * std::exp(grav_pot_diff/sound_speed_sq);
          w(IDN, k, j, i) = density;
          w(IPR, k, j, i) = density * sound_speed_sq;
        }
        else
        {
          w(IDN, k, j, i) = w(IDN, k, j, is);
          w(IPR, k, j, i) = w(IPR, k, j, is);
        }

        w(IVX, k, j, i) = std::fmin(w(IVX, k, j, is), 0);
        w(IVZ, k, j, i) = w(IVZ, k, j, is);
      }
    }
}

void bc_outflow_outer_x1(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  for (int k = ks; k <= ke; k++)
    for (int j = js; j <= je; j++)
    {
      const Real r0 = mb->pcoord->x1v(ie);
      const Real z  = mb->pcoord->x3v(k);

      const Real d0 = w(IDN, k, j, ie);
      const Real v0 = w(IVY, k, j, ie);
      const Real sound_speed_sq = std::fmax(
        w(IPR, k, j, ie) / d0,
        bc_sound_speed_sq_min_mult * -(*grav_pot)(r0, z));

      for (int i = ie+1; i <= ie+ng; i++)
      {
        const Real r = mb->pcoord->x1v(i);

        Real grav_pot_diff = (*grav_pot)(r0, z) - (*grav_pot)(r, z);
        if (bc_vel_mode == 0)
        {
          grav_pot_diff += v0*v0/2*(r*r/(r0*r0)-1);
          w(IVY, k, j, i) = v0/r0*r;
        }
        else if (bc_vel_mode == 1)
        {
          grav_pot_diff += v0*v0*std::log(r/r0);
          w(IVY, k, j, i) = v0;
        }
        else if (bc_vel_mode == 2)
        {
          grav_pot_diff += v0*v0/2*(1-r0*r0/(r*r));
          w(IVY, k, j, i) = v0*r0/r;
        }
        else
          throw std::runtime_error(
            "### FATAL ERROR in Boundary Conditions\n"
            "bc_vel_mode must be in [0, 1, 2]");

        if (bc_zero_force)
        {
          const Real density = d0 * std::exp(grav_pot_diff/sound_speed_sq);
          w(IDN, k, j, i) = density;
          w(IPR, k, j, i) = density * sound_speed_sq;
        }
        else
        {
          w(IDN, k, j, i) = w(IDN, k, j, ie);
          w(IPR, k, j, i) = w(IPR, k, j, ie);
        }

        w(IVX, k, j, i) = std::fmax(w(IVX, k, j, ie), 0);
        w(IVZ, k, j, i) = w(IVZ, k, j, ie);
      }
    }
}

void bc_outflow_inner_x3(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  for (int j = js; j <= je; j++)
    for (int i = is; i <= ie; i++)
    {
      const Real r  = mb->pcoord->x1v(i);
      const Real z0 = mb->pcoord->x3v(ks);

      const Real d0 = w(IDN, ks, j, i);
      const Real sound_speed_sq = std::fmax(
        w(IPR, ks, j, i) / d0,
        bc_sound_speed_sq_min_mult * -(*grav_pot)(r, z0));

      for (int k = ks-ng; k <= ks-1; k++)
      {
        const Real z = mb->pcoord->x3v(k);

        const Real grav_pot_diff = (*grav_pot)(r, z0) - (*grav_pot)(r, z);
        const Real density = d0 * std::exp(grav_pot_diff/sound_speed_sq);

        w(IDN, k, j, i) = density;
        w(IVX, k, j, i) = w(IVX, ks, j, i);
        w(IVY, k, j, i) = w(IVY, ks, j, i);
        w(IVZ, k, j, i) = std::fmin(w(IVZ, ks, j, i), 0);
        w(IPR, k, j, i) = density * sound_speed_sq;
      }
    }
}

void bc_inject_outer_x3(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  for (int j = js; j <= je; j++)
    for (int i = is; i <= ie; i++)
    {
      const Real r  = mb->pcoord->x1v(i);
      const Real p  = mb->pcoord->x2v(j);
      const Real z0 = mb->pcoord->x3v(ke);

      const Real d0 = w(IDN, ke, j, i);
      const Real sound_speed_sq = std::fmax(
        w(IPR, ke, j, i) / d0,
        bc_sound_speed_sq_min_mult * -(*grav_pot)(r, z0));

      for (int k = ke+1; k <= ke+ng; k++)
      {
        const Real z = mb->pcoord->x3v(k);

        const Real grav_pot_diff = (*grav_pot)(r, z0) - (*grav_pot)(r, z);
        const Real density = d0 * std::exp(grav_pot_diff/sound_speed_sq);
        const Primitive stream_prim = stream->CalcPrimitive(r, p, z);

        if (stream_prim.d > density && stream_prim.v3 < 0)
        {
          w(IDN, k, j, i) = stream_prim.d;
          w(IVX, k, j, i) = stream_prim.v1;
          w(IVY, k, j, i) = stream_prim.v2;
          w(IVZ, k, j, i) = stream_prim.v3;
          w(IPR, k, j, i) = stream_prim.p;
        }
        else
        {
          w(IDN, k, j, i) = density;
          w(IVX, k, j, i) = w(IVX, ke, j, i);
          w(IVY, k, j, i) = w(IVY, ke, j, i);
          w(IVZ, k, j, i) = std::fmax(w(IVZ, ke, j, i), 0);
          w(IPR, k, j, i) = density * sound_speed_sq;
        }
      }
    }
}

void enforce_floor(MeshBlock *mb)
{
  const int is = mb->is, ie = mb->ie;
  const int js = mb->js, je = mb->je;
  const int ks = mb->ks, ke = mb->ke;

  for (int k = ks; k <= ke; k++)
    for (int i = is; i <= ie; i++)
    {
      const Real r = mb->pcoord->x1v(i);
      const Real z = mb->pcoord->x3v(k);

      const Primitive vacuum_prim = vacuum->CalcPrimitive(r, z);

      for (int j = js; j <= je; j++)
      {
        Primitive prim;
        prim.d = mb->phydro->w(IDN, k, j, i);
        prim.p = mb->phydro->w(IPR, k, j, i);
        prim.v1 = mb->phydro->w(IVX, k, j, i);
        prim.v2 = mb->phydro->w(IVY, k, j, i);
        prim.v3 = mb->phydro->w(IVZ, k, j, i);

        bool modified = false;

        if (prim.d < vacuum_prim.d)
        {
          const Real r = prim.d / vacuum_prim.d;

          prim.d = vacuum_prim.d;
          prim.v1 = (1-r) * vacuum_prim.v1 + r * prim.v1;
          prim.v2 = (1-r) * vacuum_prim.v2 + r * prim.v2;
          prim.v3 = (1-r) * vacuum_prim.v3 + r * prim.v3;
          modified = true;
        }

        if (prim.p < vacuum_prim.p)
        {
          prim.p = vacuum_prim.p;
          modified = true;
        }

        if (modified)
        {
          mb->phydro->w(IDN, k, j, i) = prim.d;
          mb->phydro->w(IVX, k, j, i) = prim.v1;
          mb->phydro->w(IVY, k, j, i) = prim.v2;
          mb->phydro->w(IVZ, k, j, i) = prim.v3;
          mb->phydro->w(IPR, k, j, i) = prim.p;

          const Conserved cons = prim.ToConserved(adiabatic_index);

          mb->phydro->u(IDN, k, j, i) = cons.d;
          mb->phydro->u(IM1, k, j, i) = cons.m1;
          mb->phydro->u(IM2, k, j, i) = cons.m2;
          mb->phydro->u(IM3, k, j, i) = cons.m3;
          mb->phydro->u(IEN, k, j, i) = cons.e;
        }
      }
    }
}

#if GALPOT_GRAVITY
Real galaxy_midplane_density(Real r)
{
  const Real rd = 3/8. * 7 / McMillanGravity::length_scale;
  const Real rm = 3/8. * 4 / McMillanGravity::length_scale;

  return std::exp(-rm/r-r/rd + std::sqrt(rm/rd)*2);
}

Real galaxy_midplane_density_power_law_index(Real r)
{
  const Real rd = 3/8. * 7 / McMillanGravity::length_scale;
  const Real rm = 3/8. * 4 / McMillanGravity::length_scale;

  return rm/r-r/rd;
}
#endif

template<typename T, typename F>
T solve_brentq(const F &f, T a, T b)
{
  // maximum number of iterations
  const int max_iter = 256;
  // machine epsilon
  const T e = std::numeric_limits<T>::epsilon();

  // last and second-to-last values of b
  T c = a, d;
  // function evaluated at a, b, c
  T fa = f(a), fb = f(b), fc = fa;
  // whether last step used bisection
  bool last_bisect = true;

  // ensure zero is bracketed
  if (fa * fb > 0)
    return std::numeric_limits<T>::quiet_NaN();

  for (int i = 0; i < max_iter; i++)
  {
    // leave loop if converged
    if (fb == 0 || std::fabs(b-a) < (1+std::fabs(a)) * e)
      return b;

    // swap a, b so that b is better estimate
    if (std::fabs(fa) < std::fabs(fb))
    {
      const T temp1 =  a;  a =  b;  b = temp1;
      const T temp2 = fa; fa = fb; fb = temp2;
    }

    T x;
    if (fa != fc && fb != fc)
    {
      // perform inverse quadratic interpolation
      // x = a * fb * fc / (fa-fb) / (fa-fc) +
      //     b * fc * fa / (fb-fc) / (fb-fa) +
      //     c * fa * fb / (fc-fa) / (fc-fb);
      const T s = fb / fa, t = fa / fc;
      const T
        p = s * (t*t * (1-s) * (b-c) + (1-s*t) * (a-b)),
        q = (s-1) * (t-1) * (s*t-1);
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
    const T fx = f(x);
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
