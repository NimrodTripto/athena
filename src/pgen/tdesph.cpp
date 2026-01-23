#include <algorithm>  // for std::min
#include <array>    // std::array
#include <cmath>    // std::fmax, std::fmin, std::log10, std::pow
#include <cstring>  // std::strcmp
#include <iostream>  // std::cout
#include <limits>   // std::numeric_limits
#include <memory>   // std::make_shared, std::shared_ptr, std::unique_ptr
#include <ostream>  // std::endl
#include <sstream>  // std::stringstream
#include <string>   // std::string # #
#include <tuple>    // std::get

#include "../athena.hpp"
#include "../globals.hpp"  // Globals::my_rank
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../nr_radiation/integrators/rad_integrators.hpp"
#include "../nr_radiation/radiation.hpp"
#include "../parameter_input.hpp"

#include "pgenitrp.hpp"
#include "pgenphys.hpp"
#include "pgenuniq.hpp"

namespace
{

struct RefinementBox {
  Real x1min, x1max;
  Real x2min, x2max;
  Real x3min, x3max;
  int  level;
};

std::vector<RefinementBox> g_refine_boxes;   // filled once in InitUserMeshData

int default_refinement_level = 0; // NEW

int ProblemRefinement(MeshBlock *pmb);

using BcFunc = void (
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real t, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng);

using RadBcFunc = void (
  MeshBlock *mb, Coordinates *co, NRRadiation *rad, AthenaArray<Real> const &w,
  FaceField &b, AthenaArray<Real> &ir,
  Real t, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng);

using SrcFunc = void (
  MeshBlock *mb, Real t, Real dt, AthenaArray<Real> const &w,
  AthenaArray<Real> const &w_scalar, AthenaArray<Real> const &bcc,
  AthenaArray<Real> &u, AthenaArray<Real> &u_scalar);

using RadOpFunc = void (
  MeshBlock *mb, AthenaArray<Real> &w,
  int is, int ie, int js, int je, int ks, int ke);

bool stream_check_place = false;   // <-- NEW

void ic_gas(MeshBlock *mb, ParameterInput *in);

BcFunc bc_inner_x1, bc_outer_x1;
BcFunc bc_inner_x2, bc_outer_x2;
BcFunc bc_inner_x1_gas_dio, bc_outer_x1_gas_dio;
BcFunc bc_inner_x1_mag_dio, bc_outer_x1_mag_dio;
BcFunc bc_inner_x2_gas_dio, bc_outer_x2_gas_dio;
BcFunc bc_inner_x2_mag_dio, bc_outer_x2_mag_dio;

RadBcFunc bc_inner_x1_rad_dio, bc_outer_x1_rad_dio;
RadBcFunc bc_inner_x2_rad_dio, bc_outer_x2_rad_dio;

std::string refinement_type;

SrcFunc sc_all;
SrcFunc sc_gravity;
SrcFunc sc_stream;

void op_wrapper(MeshBlock *mb, AthenaArray<Real> &w);
RadOpFunc *op_wrapped;

RadOpFunc op_none, op_flat, op_opal, op_zhu, op_zhu_2;
Real op_flat_opc_mas_rs, op_flat_opc_mas_sc, op_flat_cutoff_temp;
std::unique_ptr<GridLinearInterpolator<Real, 2, 1> const> op_opal_table;
std::unique_ptr<GridLinearInterpolator<Real, 2, 2> const> op_zhu_table;

void uw_apply_limits(MeshBlock *mb);

[[noreturn]] void error_bad_param(char const *func, char const *param);

std::shared_ptr<Problem::AxisymmetricGravity const> grav;
std::unique_ptr<Problem::AxisymmetricGas const> disk, vacuum;
std::shared_ptr<Problem::ParabolicStream> stream;
bool point_mass_grav = false;
Problem::ParabolicStream::InjectionMode stream_mode;

Real density_floor, pressure_floor, temp_floor;
Real alf_spd_sq_ceil, snd_spd_sq_floor, snd_spd_sq_ceil;
Real density_unit_cgs, temperature_unit_cgs, inv_opc_mas_unit_cgs;

}

void Mesh::InitUserMeshData(ParameterInput *in)
{
  refinement_type = in->GetString("mesh", "refinement");
  std::string const stream_type = in->GetOrAddString("stream", "type", "none");
  Real const inf = std::numeric_limits<Real>::infinity();

  if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") != 0)
  {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]"
        << std::endl
        << "code must be configured with '--coord spherical_polar'"
        << std::endl;
    ATHENA_ERROR(msg);
  }

  // Initialize stream to nullptr by default
  stream = nullptr;

  density_floor     = in->GetOrAddReal("hydro",     "dfloor",  0);
  pressure_floor    = in->GetOrAddReal("hydro",     "pfloor",  0);
  temp_floor        = in->GetOrAddReal("hydro",     "tempfloor",  0);
  alf_spd_sq_ceil   = in->GetOrAddReal("hydro",     "vaceil",  inf);
  snd_spd_sq_floor  = in->GetOrAddReal("radiation", "csfloor", 0);
  snd_spd_sq_ceil   = in->GetOrAddReal("radiation", "csceil",  inf);
  alf_spd_sq_ceil  *= alf_spd_sq_ceil;
  snd_spd_sq_floor *= snd_spd_sq_floor;
  snd_spd_sq_ceil  *= snd_spd_sq_ceil;

  std::string const gravity_type = in->GetString("gravity", "type");
  if (gravity_type == "softened")
  {
    grav = std::make_shared<Problem::SoftenedNewtonianGravity>(
      in->GetReal("gravity", "mass"),
      in->GetReal("gravity", "soft_radius"));
    point_mass_grav = true;
  }
  else if (gravity_type == "paczynski_wiita")
  {
    grav = std::make_shared<Problem::PaczynskiWiitaGravity>(
      in->GetReal("gravity", "mass"),
      in->GetReal("gravity", "grav_radius"));
    point_mass_grav = true;
  }
  else
    error_bad_param("Mesh::InitUserMeshData", "gravity/type");

  std::string const disk_type = in->GetOrAddString("disk", "type", "none");
  if (disk_type == "none")
    disk = std_make_unique<Problem::UniformGas>(0., 0.);
  else if (disk_type == "polytrope")
    disk = std_make_unique<Problem::Polytrope>(
      grav,
      in->GetReal("disk", "radius"),
      in->GetReal("disk", "density"),
      in->GetReal("disk", "poly_index"),
      in->GetReal("disk", "poly_const"),
      in->GetReal("disk", "shear"));
  else if (disk_type == "const_aspect")
    disk = std_make_unique<Problem::ConstantAspectRatioDisk>(
      grav,
      in->GetReal("disk", "aspect"),
      in->GetReal("disk", "density"));
  else
    error_bad_param("Mesh::InitUserMeshData", "disk/type");

  std::string const vacuum_type = in->GetOrAddString("vacuum", "type", "none");
  if (vacuum_type == "none")
    vacuum = std_make_unique<Problem::UniformGas>(0., 0.);
  else if (vacuum_type == "polytrope")
    vacuum = std_make_unique<Problem::Polytrope>(
      grav,
      in->GetReal("vacuum", "radius"),
      in->GetReal("vacuum", "density"),
      in->GetReal("vacuum", "poly_index"),
      in->GetReal("vacuum", "poly_const"),
      in->GetReal("vacuum", "shear"));
  else
    error_bad_param("Mesh::InitUserMeshData", "vacuum/type");

  // std::string const stream_type = in->GetOrAddString("stream", "type", "none");
  if (stream_type == "none")
  {
    stream = nullptr;
  }
  else if (stream_type == "parabolic")
  {
    if (!point_mass_grav)
    {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [Mesh::InitUserMeshData]"
          << std::endl
          << "stream/parabolic in input file requires point-mass gravity"
          << std::endl;
      ATHENA_ERROR(msg);
    }

    std::string const stream_mode_ = in->GetString("stream", "mode");
    if (stream_mode_ == "area")
      stream_mode = Problem::ParabolicStream::InjectionMode::Area;
    else if (stream_mode_ == "volume")
      stream_mode = Problem::ParabolicStream::InjectionMode::Volume;
    else
      error_bad_param("Mesh::InitUserMeshData", "stream/mode");

    stream = std::make_shared<Problem::ParabolicStream>(
      in->GetReal("gravity", "mass"),
      in->GetReal("gravity", "grav_radius"),
      in->GetReal("stream", "pitch"),
      in->GetReal("stream", "roll"),
      in->GetReal("stream", "yaw"),
      in->GetReal("stream", "src_l"),
      in->GetReal("stream", "src_m"),
      in->GetReal("stream", "src_n"),
      in->GetReal("stream", "wid_l"),
      in->GetReal("stream", "wid_m"),
      in->GetReal("stream", "wid_n"),
      in->GetReal("stream", "current"),
      in->GetReal("stream", "sound_speed"),
      in->GetOrAddReal("stream", "density_mult", 1.0),
      in->GetOrAddReal("stream", "vel_coeff", 2.01),
      stream_mode);
  }
  else
    error_bad_param("Mesh::InitUserMeshData", "stream/type");

  if (NR_RADIATION_ENABLED)
  {
    density_unit_cgs =
      in->GetReal("radiation", "density_unit");
    temperature_unit_cgs =
      in->GetReal("radiation", "T_unit");
    inv_opc_mas_unit_cgs =
      in->GetReal("radiation", "length_unit") * density_unit_cgs;

    std::string const opacity_type = in->GetOrAddString(
      "opacity", "type", "none");
    if (opacity_type == "none")
      op_wrapped = op_none;
    else if (opacity_type == "flat")
    {
      op_wrapped = op_flat;
      op_flat_opc_mas_rs  = in->GetReal     ("opacity", "opc_mas_rs");
      op_flat_opc_mas_sc  = in->GetReal     ("opacity", "opc_mas_sc");
      op_flat_cutoff_temp = in->GetOrAddReal("opacity", "cutoff_temp", inf);
    }
    else if (opacity_type == "opal")
    {
      op_wrapped = op_opal;
      op_opal_table = std_make_unique<GridLinearInterpolator<Real, 2, 1>>(
        in->GetString("opacity", "fname").c_str());
    }
    else if (opacity_type == "zhu")
    {
      op_wrapped = op_zhu;
      op_zhu_table = std_make_unique<GridLinearInterpolator<Real, 2, 2>>(
        in->GetString("opacity", "fname").c_str());
    }
    else if (opacity_type == "zhu_2")
    {
      op_wrapped = op_zhu_2;
      op_zhu_table = std_make_unique<GridLinearInterpolator<Real, 2, 2>>(
        in->GetString("opacity", "fname").c_str());
    }
    else
      error_bad_param("MeshBlock::ProblemGenerator", "opacity/type");
  }

  if (in->GetString("mesh", "ix1_bc") == "user")
    EnrollUserBoundaryFunction(
      BoundaryFace::inner_x1, bc_inner_x1);
  if (in->GetString("mesh", "ox1_bc") == "user")
    EnrollUserBoundaryFunction(
      BoundaryFace::outer_x1, bc_outer_x1);
  if (in->GetString("mesh", "ix2_bc") == "user")
    EnrollUserBoundaryFunction(
      BoundaryFace::inner_x2, bc_inner_x2);
  if (in->GetString("mesh", "ox2_bc") == "user")
    EnrollUserBoundaryFunction(
      BoundaryFace::outer_x2, bc_outer_x2);

  if (NR_RADIATION_ENABLED)
  {
    if (in->GetString("mesh", "ix1_bc") == "user")
      EnrollUserRadBoundaryFunction(
        BoundaryFace::inner_x1, bc_inner_x1_rad_dio);
    if (in->GetString("mesh", "ox1_bc") == "user")
      EnrollUserRadBoundaryFunction(
        BoundaryFace::outer_x1, bc_outer_x1_rad_dio);
    if (in->GetString("mesh", "ix2_bc") == "user")
      EnrollUserRadBoundaryFunction(
        BoundaryFace::inner_x2, bc_inner_x2_rad_dio);
    if (in->GetString("mesh", "ox2_bc") == "user")
      EnrollUserRadBoundaryFunction(
        BoundaryFace::outer_x2, bc_outer_x2_rad_dio);
  }

  // Iterate over blocks named  custom_refinement1, custom_refinement2, …
  for (int i = 1; /*break inside*/; ++i) {
    std::ostringstream label;
    label << "custom_refinement" << i;
    if (!in->DoesParameterExist(label.str(), "level")) break;  // finished

    // std::cout << "saving block " << i << " to g_refine_boxes" << std::endl;

    RefinementBox box;
    box.x1min = in->GetReal(label.str(), "x1min");
    box.x1max = in->GetReal(label.str(), "x1max");
    box.x2min = in->GetReal(label.str(), "x2min");
    box.x2max = in->GetReal(label.str(), "x2max");
    box.x3min = in->GetReal(label.str(), "x3min");
    box.x3max = in->GetReal(label.str(), "x3max");
    box.level = in->GetInteger(label.str(), "level");

    g_refine_boxes.emplace_back(box);
  }

  if (Globals::my_rank == 0) {
    std::cout << "[AMR] boxes=" << g_refine_boxes.size() << "\n";
    for (size_t i = 0; i < g_refine_boxes.size(); ++i) {
      auto &b = g_refine_boxes[i];
      std::cout << "[AMR] box " << i
                << " x1=[" << b.x1min << "," << b.x1max << "]"
                << " x2=[" << b.x2min << "," << b.x2max << "]"
                << " x3=[" << b.x3min << "," << b.x3max << "]"
                << " level=" << b.level << "\n";
    }
  }

  // Print all blocks found
  // std::cout << "Found " << g_refine_boxes.size() << " refinement boxes:" << std::endl;
  // for (const auto &box : g_refine_boxes) {
  //   std::cout << "  x1: [" << box.x1min << ", " << box.x1max << "]"
  //             << " x2: [" << box.x2min << ", " << box.x2max << "]"
  //             << " x3: [" << box.x3min << ", " << box.x3max << "]"
  //             << " level: " << box.level
  //             << std::endl;
  // }

  // read default level for blocks not in any custom box
  default_refinement_level =
    in->GetOrAddInteger("problem", "default_level", 0);

  if (refinement_type == "adaptive") {
    std::cout<< std::endl
              << "should run refinement"
              << std::endl;
    EnrollUserRefinementCondition(ProblemRefinement); 

  }
  
  EnrollUserExplicitSourceFunction(sc_all);

}

void MeshBlock::ProblemGenerator(ParameterInput *in)
{
  ic_gas(this, in);
}

void Mesh::UserWorkInLoop()
{
}

void MeshBlock::UserWorkInLoop()
{
  uw_apply_limits(this);
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *in)
{
}

void Mesh::UserWorkAfterLoop(ParameterInput *in)
{
}

namespace
{

using Problem::Primitive;

void ic_gas(MeshBlock *mb, ParameterInput *in)
{
  int const is = mb->is, ie = mb->ie;
  int const js = mb->js, je = mb->je;
  int const ks = mb->ks, ke = mb->ke;

  for (int j = js; j <= je; ++j)
    for (int i = is; i <= ie; ++i)
    {
      Real const x1 = mb->pcoord->x1v(i);
      Real const x2 = mb->pcoord->x2v(j);

      Primitive const disk_prim = disk->CalcPrimitive(
        x1, x2, Problem::SphericalCoordsTag{});
      Primitive const vacuum_prim = vacuum->CalcPrimitive(
        x1, x2, Problem::SphericalCoordsTag{});

      Primitive const &prim =
           disk_prim.dn > vacuum_prim.dn
        && disk_prim.pr > vacuum_prim.pr ? disk_prim : vacuum_prim;

      for (int k = ks; k <= ke; ++k)
      {
        // set primitive according to disk or vacuum
        mb->phydro->w(IDN, k, j, i) = std::fmax(prim.dn, density_floor);
        mb->phydro->w(IVX, k, j, i) = prim.v1;
        mb->phydro->w(IVY, k, j, i) = prim.v2;
        mb->phydro->w(IVZ, k, j, i) = prim.v3;
        mb->phydro->w(IPR, k, j, i) = std::fmax(prim.pr, pressure_floor);
      }
    }

  // convert primitive to conserved variables
  mb->peos->PrimitiveToConserved(
    mb->phydro->w, mb->pfield->bcc, mb->phydro->u, mb->pcoord,
    is, ie, js, je, ks, ke);
}

void bc_inner_x1(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real t, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  bc_inner_x1_gas_dio(mb, co, w, b, t, dt, is, ie, js, je, ks, ke, ng);
  bc_inner_x1_mag_dio(mb, co, w, b, t, dt, is, ie, js, je, ks, ke, ng);
}

void bc_outer_x1(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real t, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  bc_outer_x1_gas_dio(mb, co, w, b, t, dt, is, ie, js, je, ks, ke, ng);
  bc_outer_x1_mag_dio(mb, co, w, b, t, dt, is, ie, js, je, ks, ke, ng);
}

void bc_inner_x2(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real t, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  bc_inner_x2_gas_dio(mb, co, w, b, t, dt, is, ie, js, je, ks, ke, ng);
  bc_inner_x2_mag_dio(mb, co, w, b, t, dt, is, ie, js, je, ks, ke, ng);
}

void bc_outer_x2(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real t, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  bc_outer_x2_gas_dio(mb, co, w, b, t, dt, is, ie, js, je, ks, ke, ng);
  bc_outer_x2_mag_dio(mb, co, w, b, t, dt, is, ie, js, je, ks, ke, ng);
}

void bc_inner_x1_gas_dio(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real t, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j)
      for (int i = is-ng; i <= is-1; ++i)
      {
        w(IDN, k, j, i) =              w(IDN, k, j, is);
        w(IVX, k, j, i) = std::fmin(0, w(IVX, k, j, is));
        w(IVY, k, j, i) =              w(IVY, k, j, is);
        w(IVZ, k, j, i) =              w(IVZ, k, j, is);
        w(IPR, k, j, i) =              w(IPR, k, j, is);
      }
}

void bc_outer_x1_gas_dio(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real t, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j)
      for (int i = ie+1; i <= ie+ng; ++i)
      {
        w(IDN, k, j, i) =              w(IDN, k, j, ie);
        w(IVX, k, j, i) = std::fmax(0, w(IVX, k, j, ie));
        w(IVY, k, j, i) =              w(IVY, k, j, ie);
        w(IVZ, k, j, i) =              w(IVZ, k, j, ie);
        w(IPR, k, j, i) =              w(IPR, k, j, ie);
      }
}

void bc_inner_x1_mag_dio(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real t, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  if (MAGNETIC_FIELDS_ENABLED)
  {
    for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= je; ++j)
        for (int i = is-ng; i <= is-1; ++i)
          b.x1f(k, j, i) = w(IVX, k, j, is) < 0 ? b.x1f(k, j, is) : 0;

    for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= je+1; ++j)
        for (int i = is-ng; i <= is-1; ++i)
          b.x2f(k, j, i) = w(IVX, k, j, is) < 0 ? b.x2f(k, j, is) : 0;

    for (int k = ks; k <= ke+1; ++k)
      for (int j = js; j <= je; ++j)
        for (int i = is-ng; i <= is-1; ++i)
          b.x3f(k, j, i) = w(IVX, k, j, is) < 0 ? b.x3f(k, j, is) : 0;
  }
}

void bc_outer_x1_mag_dio(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real t, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  if (MAGNETIC_FIELDS_ENABLED)
  {
    for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= je; ++j)
        for (int i = ie+2; i <= ie+ng+1; ++i)
          b.x1f(k, j, i) = w(IVX, k, j, ie) > 0 ? b.x1f(k, j, ie+1) : 0;

    for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= je+1; ++j)
        for (int i = ie+1; i <= ie+ng; ++i)
          b.x2f(k, j, i) = w(IVX, k, j, ie) > 0 ? b.x2f(k, j, ie) : 0;

    for (int k = ks; k <= ke+1; ++k)
      for (int j = js; j <= je; ++j)
        for (int i = ie+1; i <= ie+ng; ++i)
          b.x3f(k, j, i) = w(IVX, k, j, ie) > 0 ? b.x3f(k, j, ie) : 0;
  }
}

void bc_inner_x2_gas_dio(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real t, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  for (int i = is; i <= ie; ++i)
    for (int k = ks; k <= ke; ++k)
      for (int j = js-ng; j <= js-1; ++j)
      {
        w(IDN, k, j, i) =              w(IDN, k, js, i);
        w(IVX, k, j, i) =              w(IVX, k, js, i);
        w(IVY, k, j, i) = std::fmin(0, w(IVY, k, js, i));
        w(IVZ, k, j, i) =              w(IVZ, k, js, i);
        w(IPR, k, j, i) =              w(IPR, k, js, i);
      }
}

void bc_outer_x2_gas_dio(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real t, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  for (int i = is; i <= ie; ++i)
    for (int k = ks; k <= ke; ++k)
      for (int j = je+1; j <= je+ng; ++j)
      {
        w(IDN, k, j, i) =              w(IDN, k, je, i);
        w(IVX, k, j, i) =              w(IVX, k, je, i);
        w(IVY, k, j, i) = std::fmax(0, w(IVY, k, je, i));
        w(IVZ, k, j, i) =              w(IVZ, k, je, i);
        w(IPR, k, j, i) =              w(IPR, k, je, i);
      }
}

void bc_inner_x2_mag_dio(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real t, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  if (MAGNETIC_FIELDS_ENABLED)
  {
    for (int i = is; i <= ie; ++i)
      for (int k = ks; k <= ke; ++k)
        for (int j = js-ng; j <= js-1; ++j)
          b.x2f(k, j, i) = w(IVY, k, js, i) < 0 ? b.x2f(k, js, i) : 0;

    for (int i = is; i <= ie; ++i)
      for (int k = ks; k <= ke+1; ++k)
        for (int j = js-ng; j <= js-1; ++j)
          b.x3f(k, j, i) = w(IVY, k, js, i) < 0 ? b.x3f(k, js, i) : 0;

    for (int i = is; i <= ie+1; ++i)
      for (int k = ks; k <= ke; ++k)
        for (int j = js-ng; j <= js-1; ++j)
          b.x1f(k, j, i) = w(IVY, k, js, i) < 0 ? b.x1f(k, js, i) : 0;
  }
}

void bc_outer_x2_mag_dio(
  MeshBlock *mb, Coordinates *co, AthenaArray<Real> &w, FaceField &b,
  Real t, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  if (MAGNETIC_FIELDS_ENABLED)
  {
    for (int i = is; i <= ie; ++i)
      for (int k = ks; k <= ke; ++k)
        for (int j = je+2; j <= je+ng+1; ++j)
          b.x2f(k, j, i) = w(IVY, k, je, i) > 0 ? b.x2f(k, je+1, i) : 0;

    for (int i = is; i <= ie; ++i)
      for (int k = ks; k <= ke+1; ++k)
        for (int j = je+1; j <= je+ng; ++j)
          b.x3f(k, j, i) = w(IVY, k, je, i) > 0 ? b.x3f(k, je, i) : 0;

    for (int i = is; i <= ie+1; ++i)
      for (int k = ks; k <= ke; ++k)
        for (int j = je+1; j <= je+ng; ++j)
          b.x1f(k, j, i) = w(IVY, k, je, i) > 0 ? b.x1f(k, je, i) : 0;
  }
}

void bc_inner_x1_rad_dio(
  MeshBlock *mb, Coordinates *co, NRRadiation *rad, AthenaArray<Real> const &w,
  FaceField &b, AthenaArray<Real> &ir,
  Real t, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j)
      for (int m = 0; m < rad->nang; ++m)
        if (rad->mu(0, k, j, is, m) < 0)
          for (int i = is-ng; i <= is-1; ++i)
            for (int l = 0; l < rad->nfreq; ++l)
              rad->ir(k, j, i, rad->nang*l+m) =
                rad->ir(k, j, is, rad->nang*l+m);
        else
          for (int i = is-ng; i <= is-1; ++i)
            for (int l = 0; l < rad->nfreq; ++l)
              rad->ir(k, j, i, rad->nang*l+m) = 0;
}

void bc_outer_x1_rad_dio(
  MeshBlock *mb, Coordinates *co, NRRadiation *rad, AthenaArray<Real> const &w,
  FaceField &b, AthenaArray<Real> &ir,
  Real t, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j)
      for (int m = 0; m < rad->nang; ++m)
        if (rad->mu(0, k, j, ie, m) > 0)
          for (int i = ie+1; i <= ie+ng; ++i)
            for (int l = 0; l < rad->nfreq; ++l)
              rad->ir(k, j, i, rad->nang*l+m) =
                rad->ir(k, j, ie, rad->nang*l+m);
        else
          for (int i = ie+1; i <= ie+ng; ++i)
            for (int l = 0; l < rad->nfreq; ++l)
              rad->ir(k, j, i, rad->nang*l+m) = 0;
}

void bc_inner_x2_rad_dio(
  MeshBlock *mb, Coordinates *co, NRRadiation *rad, AthenaArray<Real> const &w,
  FaceField &b, AthenaArray<Real> &ir,
  Real t, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  for (int i = is; i <= ie; ++i)
    for (int k = ks; k <= ke; ++k)
      for (int m = 0; m < rad->nang; ++m)
        if (rad->mu(1, k, js, i, m) < 0)
          for (int j = js-ng; j <= js-1; ++j)
            for (int l = 0; l < rad->nfreq; ++l)
              rad->ir(k, j, i, rad->nang*l+m) =
                rad->ir(k, js, i, rad->nang*l+m);
        else
          for (int j = js-ng; j <= js-1; ++j)
            for (int l = 0; l < rad->nfreq; ++l)
              rad->ir(k, j, i, rad->nang*l+m) = 0;
}

void bc_outer_x2_rad_dio(
  MeshBlock *mb, Coordinates *co, NRRadiation *rad, AthenaArray<Real> const &w,
  FaceField &b, AthenaArray<Real> &ir,
  Real t, Real dt, int is, int ie, int js, int je, int ks, int ke, int ng)
{
  for (int i = is; i <= ie; ++i)
    for (int k = ks; k <= ke; ++k)
      for (int m = 0; m < rad->nang; ++m)
        if (rad->mu(1, k, je, i, m) > 0)
          for (int j = je+1; j <= je+ng; ++j)
            for (int l = 0; l < rad->nfreq; ++l)
              rad->ir(k, j, i, rad->nang*l+m) =
                rad->ir(k, je, i, rad->nang*l+m);
        else
          for (int j = je+1; j <= je+ng; ++j)
            for (int l = 0; l < rad->nfreq; ++l)
              rad->ir(k, j, i, rad->nang*l+m) = 0;
}

void sc_all(
  MeshBlock *mb, Real t, Real dt, AthenaArray<Real> const &w,
  AthenaArray<Real> const &w_scalar, AthenaArray<Real> const &bcc,
  AthenaArray<Real> &u, AthenaArray<Real> &u_scalar)
{
  // only the first function can use the primitive variables
  sc_gravity(mb, t, dt, w, w_scalar, bcc, u, u_scalar);
  sc_stream (mb, t, dt, w, w_scalar, bcc, u, u_scalar);
}

void sc_gravity(
  MeshBlock *mb, Real t, Real dt, AthenaArray<Real> const &w,
  AthenaArray<Real> const &w_scalar, AthenaArray<Real> const &bcc,
  AthenaArray<Real> &u, AthenaArray<Real> &u_scalar)
{
  int const ng = NGHOST;
  int const is = mb->is, ie = mb->ie;
  int const js = mb->js, je = mb->je;
  int const ks = mb->ks, ke = mb->ke;
  int const il = is-(is<ie)*ng, iu = ie+(is<ie)*ng;
  int const jl = js-(js<je)*ng, ju = je+(js<je)*ng;
  int const kl = ks-(ks<ke)*ng, ku = ke+(ks<ke)*ng;

  for (int k = kl; k <= ku; ++k)
    for (int j = jl; j <= ju; ++j)
      for (int i = il; i <= iu; ++i)
      {
        Real const x1 = mb->pcoord->x1v(i);
        Real const x2 = mb->pcoord->x2v(j);
        Real const x3 = mb->pcoord->x3v(k);

        Real const dn = u(IDN, k, j, i);
        Real const m1 = u(IM1, k, j, i);
        Real const m2 = u(IM2, k, j, i);
        Real const m3 = u(IM3, k, j, i);

        auto const tmp = grav->Gradient(
          x1, x2, x3, Problem::SphericalCoordsTag{});

        Real const dv1 = dt * -std::get<0>(tmp);
        Real const dv2 = dt * -std::get<1>(tmp);
        Real const dv3 = dt * -std::get<2>(tmp);

        Real const dm1 = dn * dv1;
        Real const dm2 = dn * dv2;
        Real const dm3 = dn * dv3;

        u(IM1, k, j, i) += dm1;
        u(IM2, k, j, i) += dm2;
        u(IM3, k, j, i) += dm3;

        if (NON_BAROTROPIC_EOS)
          u(IEN, k, j, i) += (m1+dm1/2)*dv1 + (m2+dm2/2)*dv2 + (m3+dm3/2)*dv3;
      }
}

void sc_stream(
  MeshBlock *mb, Real t, Real dt, AthenaArray<Real> const &w,
  AthenaArray<Real> const &w_scalar, AthenaArray<Real> const &bcc,
  AthenaArray<Real> &u, AthenaArray<Real> &u_scalar)
{
  if (stream != nullptr)
  {
  int const ng = NGHOST;
  int const is = mb->is, ie = mb->ie;
  int const js = mb->js, je = mb->je;
  int const ks = mb->ks, ke = mb->ke;
    int const il = is-(is<ie)*ng, iu = ie+(is<ie)*ng;
    int const jl = js-(js<je)*ng, ju = je+(js<je)*ng;
    int const kl = ks-(ks<ke)*ng, ku = ke+(ks<ke)*ng;

  Real const igm1 = 1 / (mb->peos->GetGamma() - 1);

    for (int k = kl; k <= ku; ++k)
      for (int j = jl; j <= ju; ++j)
      for (int i = il; i <= iu; ++i)
      {
        Real const x1 = mb->pcoord->x1v(i);
        Real const x2 = mb->pcoord->x2v(j);
        Real const x3 = mb->pcoord->x3v(k);

        Real const dn = u(IDN, k, j, i);

        Primitive const stream_prim = stream->CalcPrimitive(
          x1, x2, x3, Problem::SphericalCoordsTag{});

        Real const inject_dn =
            stream_mode == Problem::ParabolicStream::InjectionMode::Area
          ? std::fmax(0, stream_prim.dn - dn)
          : stream_prim.dn * dt;
        
        u(IDN, k, j, i) += inject_dn;
        u(IM1, k, j, i) -= inject_dn * stream_prim.v1;
        u(IM2, k, j, i) -= inject_dn * stream_prim.v2;
        u(IM3, k, j, i) -= inject_dn * stream_prim.v3;

        if (NON_BAROTROPIC_EOS)
        {
          u(IEN, k, j, i) += inject_dn * stream_prim.pr * igm1;

          // note that m1*v1^2/2 + m2*v2^2/2 >= (m1*v1+m2*v2)^2/(2*(m1+m2))
          // with equality if v1 == v2; extra injected energy appears as
          // internal energy, that is, injection is inelastic
          u(IEN, k, j, i) += inject_dn *
             (stream_prim.v1 * stream_prim.v1
            + stream_prim.v2 * stream_prim.v2
            + stream_prim.v3 * stream_prim.v3) / 2;
        }

          // magnetic field cannot be added: face-centered magnetic field at
          // half time-step is not an argument to the function

        // radiation cannot be added: radiation at half time-step is not an
        // argument to the function

          // opacity cannot be updated: face-centered magnetic field at half
          // time-step is not an argument to the function
        }
  }
}

void op_wrapper(MeshBlock *mb, AthenaArray<Real> &w)
{
  if (NR_RADIATION_ENABLED)
  {
    int const ng = NGHOST;
    int const is = mb->is, ie = mb->ie;
    int const js = mb->js, je = mb->je;
    int const ks = mb->ks, ke = mb->ke;
    int const il = is-(is<ie)*ng, iu = ie+(is<ie)*ng;
    int const jl = js-(js<je)*ng, ju = je+(js<je)*ng;
    int const kl = ks-(ks<ke)*ng, ku = ke+(ks<ke)*ng;

    op_wrapped(mb, w, il, iu, jl, ju, kl, ku);
  }
}

void op_none(
  MeshBlock *mb, AthenaArray<Real> &w,
  int is, int ie, int js, int je, int ks, int ke)
{
  if (NR_RADIATION_ENABLED)
    for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= je; ++j)
        for (int i = is; i <= ie; ++i)
          for (int l = 0; l < mb->pnrrad->nfreq; ++l)
          {
            mb->pnrrad->sigma_s (k, j, i, l) = 0;
            mb->pnrrad->sigma_a (k, j, i, l) = 0;
            mb->pnrrad->sigma_p (k, j, i, l) = 0;
            mb->pnrrad->sigma_pe(k, j, i, l) = 0;
          }
}

void op_flat(
  MeshBlock *mb, AthenaArray<Real> &w,
  int is, int ie, int js, int je, int ks, int ke)
{
  if (NR_RADIATION_ENABLED)
    for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= je; ++j)
        for (int i = is; i <= ie; ++i)
        {
          Real const density  = w(IDN, k, j, i);
          Real const gas_temp = w(IPR, k, j, i) / density;

          Real const tmp = density * (gas_temp <= op_flat_cutoff_temp);
          Real const opc_vol_rs = tmp * op_flat_opc_mas_rs;
          Real const opc_vol_sc = tmp * op_flat_opc_mas_sc;

          for (int l = 0; l < mb->pnrrad->nfreq; ++l)
          {
            mb->pnrrad->sigma_s (k, j, i, l) = opc_vol_sc;
            mb->pnrrad->sigma_a (k, j, i, l) = opc_vol_rs;
            mb->pnrrad->sigma_p (k, j, i, l) = opc_vol_rs;
            mb->pnrrad->sigma_pe(k, j, i, l) = opc_vol_rs;
          }
        }
}

void op_opal(
  MeshBlock *mb, AthenaArray<Real> &w,
  int is, int ie, int js, int je, int ks, int ke)
{
  if (NR_RADIATION_ENABLED)
  {
    // base-10 logarithm of the Planck mean free--free absorption opacity per
    // mass in cm^2/g for density 1 g/cm^3, temperature 1 K, and mass fractions
    // (X, Y) = (0.7, 0.3)
    Real const log10_opc_mas_pl_cgs_coeff = 24.130133511455732;

    // electron scattering opacity per mass in cm^2/g for mass fractions
    // (X, Y) = (0.7, 0.3) assuming hydrogen and helium are singly ionized
    Real const opc_mas_es_cgs = 0.3082425310414743;

    auto const table_min = op_opal_table->DomainMin();
    auto const table_max = op_opal_table->DomainMax();

    for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= je; ++j)
        for (int i = is; i <= ie; ++i)
        {
          Real const density  = w(IDN, k, j, i);
          Real const gas_temp = w(IPR, k, j, i) / density;

          Real const log10_T_cgs =
            std::log10(gas_temp * temperature_unit_cgs);
          Real const log10_R_cgs =
            std::log10(density * density_unit_cgs) - 3*log10_T_cgs + 18;
          std::array<Real, 2> const log10_T_log10_R_cgs = {
            std::fmax(std::get<0>(table_min),
            std::fmin(std::get<0>(table_max), log10_T_cgs)),
            std::fmax(std::get<1>(table_min),
            std::fmin(std::get<1>(table_max), log10_R_cgs))};

          auto const log10_opc_mas_tb_cgs =
            (*op_opal_table)(log10_T_log10_R_cgs);

          Real const opc_mas_tb =
            inv_opc_mas_unit_cgs * std::pow(10,
              std::get<0>(log10_opc_mas_tb_cgs));
          Real const opc_mas_pl =
            inv_opc_mas_unit_cgs * std::pow(10,
              log10_opc_mas_pl_cgs_coeff - 18 - log10_T_cgs/2 + log10_R_cgs);
          Real const opc_mas_sc =
            inv_opc_mas_unit_cgs * opc_mas_es_cgs;

          Real const opc_mas_rs = std::fmax(0, opc_mas_tb - opc_mas_sc);

          Real const opc_vol_rs = density * opc_mas_rs;
          Real const opc_vol_pl = density * opc_mas_pl;
          Real const opc_vol_sc = density * opc_mas_sc;

          for (int l = 0; l < mb->pnrrad->nfreq; ++l)
          {
            mb->pnrrad->sigma_s (k, j, i, l) = opc_vol_sc;
            mb->pnrrad->sigma_a (k, j, i, l) = opc_vol_rs;
            mb->pnrrad->sigma_p (k, j, i, l) = opc_vol_pl;
            mb->pnrrad->sigma_pe(k, j, i, l) = opc_vol_pl;
          }
        }
  }
}

void op_zhu(
  MeshBlock *mb, AthenaArray<Real> &w,
  int is, int ie, int js, int je, int ks, int ke)
{
  if (NR_RADIATION_ENABLED)
  {
    // electron scattering opacity per mass in cm^2/g for mass fractions
    // (X, Y) = (0.7, 0.3) assuming hydrogen and helium are fully ionized
    Real const opc_mas_es_cgs = 0.3382695306525382;

    // hydrogen ionization temperature
    Real const log10_T_ion_cgs = std::log10(2e4);

    auto const table_min = op_zhu_table->DomainMin();
    auto const table_max = op_zhu_table->DomainMax();

    for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= je; ++j)
        for (int i = is; i <= ie; ++i)
        {
          Real const density  = w(IDN, k, j, i);
          Real const gas_temp = w(IPR, k, j, i) / density;

          Real const log10_T_cgs =
            std::log10(gas_temp * temperature_unit_cgs);
          Real const log10_R_cgs =
            std::log10(density * density_unit_cgs) - 3*log10_T_cgs + 18;
          std::array<Real, 2> const log10_T_log10_R_cgs = {
            std::fmax(std::get<0>(table_min),
            std::fmin(std::get<0>(table_max), log10_T_cgs)),
            std::fmax(std::get<1>(table_min),
            std::fmin(std::get<1>(table_max), log10_R_cgs))};

          auto const tmp = (*op_zhu_table)(log10_T_log10_R_cgs);
          Real const opc_mas_tb_cgs = std::get<0>(tmp);
          Real const opc_mas_pl_cgs = std::get<1>(tmp);

          Real opc_mas_rs_cgs, opc_mas_sc_cgs;
          if (opc_mas_tb_cgs >= opc_mas_es_cgs)
          {
            opc_mas_rs_cgs = opc_mas_tb_cgs - opc_mas_es_cgs;
            opc_mas_sc_cgs = opc_mas_es_cgs;
          }
          else if (log10_T_cgs >= log10_T_ion_cgs)
          {
            opc_mas_rs_cgs = 0;
            opc_mas_sc_cgs = opc_mas_tb_cgs;
          }
          else
          {
            opc_mas_rs_cgs = opc_mas_tb_cgs;
            opc_mas_sc_cgs = 0;
          }

          Real const opc_mas_rs = inv_opc_mas_unit_cgs * opc_mas_rs_cgs;
          Real const opc_mas_pl = inv_opc_mas_unit_cgs * opc_mas_pl_cgs;
          Real const opc_mas_sc = inv_opc_mas_unit_cgs * opc_mas_sc_cgs;

          Real const opc_vol_rs = density * opc_mas_rs;
          Real const opc_vol_pl = density * opc_mas_pl;
          Real const opc_vol_sc = density * opc_mas_sc;

          for (int l = 0; l < mb->pnrrad->nfreq; ++l)
          {
            mb->pnrrad->sigma_s (k, j, i, l) = opc_vol_sc;
            mb->pnrrad->sigma_a (k, j, i, l) = opc_vol_rs;
            mb->pnrrad->sigma_p (k, j, i, l) = opc_vol_pl;
            mb->pnrrad->sigma_pe(k, j, i, l) = opc_vol_pl;
          }
        }
  }
}

void op_zhu_2(
  MeshBlock *mb, AthenaArray<Real> &w,
  int is, int ie, int js, int je, int ks, int ke)
{
  if (NR_RADIATION_ENABLED)
  {
    // electron scattering opacity per mass in cm^2/g for mass fractions
    // (X, Y) = (0.7, 0.3) assuming hydrogen and helium are fully ionized
    Real const opc_mas_es_cgs = 0.3382695306525382;

    // hydrogen ionization temperature
    Real const log10_T_ion_cgs = std::log10(2e4);

    auto const table_min = op_zhu_table->DomainMin();
    auto const table_max = op_zhu_table->DomainMax();

    for (int k = ks; k <= ke; ++k)
      for (int j = js; j <= je; ++j)
        for (int i = is; i <= ie; ++i)
        {
          Real const density  = w(IDN, k, j, i);
          Real const gas_temp = w(IPR, k, j, i) / density;

          Real const log10_rho_cgs =
            std::log10(density * density_unit_cgs);
          Real const log10_T_cgs =
            std::log10(gas_temp * temperature_unit_cgs);
          std::array<Real, 2> const log10_rho_log10_T_cgs = {
            std::fmax(std::get<0>(table_min),
            std::fmin(std::get<0>(table_max), log10_rho_cgs)),
            std::fmax(std::get<1>(table_min),
            std::fmin(std::get<1>(table_max), log10_T_cgs))};

          auto const tmp = (*op_zhu_table)(log10_rho_log10_T_cgs);
          Real const log10_opc_mas_tb_cgs = std::get<0>(tmp);
          Real const log10_opc_mas_pl_cgs = std::get<1>(tmp);
          Real const opc_mas_tb_cgs = std::pow(10, log10_opc_mas_tb_cgs);
          Real const opc_mas_pl_cgs = std::pow(10, log10_opc_mas_pl_cgs);

          Real opc_mas_rs_cgs, opc_mas_sc_cgs;
          if (opc_mas_tb_cgs >= opc_mas_es_cgs)
          {
            opc_mas_rs_cgs = opc_mas_tb_cgs - opc_mas_es_cgs;
            opc_mas_sc_cgs = opc_mas_es_cgs;
          }
          else if (log10_T_cgs >= log10_T_ion_cgs)
          {
            opc_mas_rs_cgs = 0;
            opc_mas_sc_cgs = opc_mas_tb_cgs;
          }
          else
          {
            opc_mas_rs_cgs = opc_mas_tb_cgs;
            opc_mas_sc_cgs = 0;
          }

          Real const opc_mas_rs = inv_opc_mas_unit_cgs * opc_mas_rs_cgs;
          Real const opc_mas_pl = inv_opc_mas_unit_cgs * opc_mas_pl_cgs;
          Real const opc_mas_sc = inv_opc_mas_unit_cgs * opc_mas_sc_cgs;

          Real const opc_vol_rs = density * opc_mas_rs;
          Real const opc_vol_pl = density * opc_mas_pl;
          Real const opc_vol_sc = density * opc_mas_sc;

          for (int l = 0; l < mb->pnrrad->nfreq; ++l)
          {
            mb->pnrrad->sigma_s (k, j, i, l) = opc_vol_sc;
            mb->pnrrad->sigma_a (k, j, i, l) = opc_vol_rs;
            mb->pnrrad->sigma_p (k, j, i, l) = opc_vol_pl;
            mb->pnrrad->sigma_pe(k, j, i, l) = opc_vol_pl;
          }
        }
  }
}

void uw_apply_limits(MeshBlock *mb)
{
  int const ng = NGHOST;
  int const is = mb->is, ie = mb->ie;
  int const js = mb->js, je = mb->je;
  int const ks = mb->ks, ke = mb->ke;
  int const il = is-(is<ie)*ng, iu = ie+(is<ie)*ng;
  int const jl = js-(js<je)*ng, ju = je+(js<je)*ng;
  int const kl = ks-(ks<ke)*ng, ku = ke+(ks<ke)*ng;

  for (int k = kl; k <= ku; ++k)
    for (int j = jl; j <= ju; ++j)
      for (int i = il; i <= iu; ++i)
      {
        bool modified = false;

        if (mb->phydro->w(IDN, k, j, i) <= density_floor)
        {
          // prevent large velocity and pressure as a result of reduced density
          mb->phydro->w(IDN, k, j, i) = density_floor;
          mb->phydro->w(IVX, k, j, i) = 0;
          mb->phydro->w(IVY, k, j, i) = 0;
          mb->phydro->w(IVZ, k, j, i) = 0;
          mb->phydro->w(IPR, k, j, i) = pressure_floor;

          modified = true;
        }

        if (MAGNETIC_FIELDS_ENABLED)
        {
          Real const b1 = mb->pfield->bcc(IB1, k, j, i);
          Real const b2 = mb->pfield->bcc(IB2, k, j, i);
          Real const b3 = mb->pfield->bcc(IB3, k, j, i);

          Real const alf_spd_sq =
            (b1*b1 + b2*b2 + b3*b3) / mb->phydro->w(IDN, k, j, i);
          if (alf_spd_sq > alf_spd_sq_ceil)
          {
            // reduce Alfven speed by increasing density
            mb->phydro->w(IDN, k, j, i) *= alf_spd_sq / alf_spd_sq_ceil;

            modified = true;
          }
        }

        Real const tmp1 = mb->phydro->w(IDN, k, j, i) * snd_spd_sq_floor;
        Real const tmp2 = mb->phydro->w(IDN, k, j, i) * snd_spd_sq_ceil;

        if (mb->phydro->w(IPR, k, j, i) < tmp1)
        {
          // apply isothermal sound speed floor
          mb->phydro->w(IPR, k, j, i) = tmp1;

          modified = true;
        }
        else if (mb->phydro->w(IPR, k, j, i) > tmp2)
        {
          // apply isothermal sound speed ceiling
          mb->phydro->w(IPR, k, j, i) = tmp2;

          modified = true;
        }
        
         // 3) temperature floor
         if (temp_floor > 0) {
          Real &rho = mb->phydro->w(IDN, k, j, i);
          Real &pr  = mb->phydro->w(IPR, k, j, i);
          // only if T = P/ρ is below the floor, clamp it
          if (pr/rho < temp_floor) {
            pr = rho * temp_floor;
            modified = true;
          }
        }
        
        if (modified)
        {
          // convert primitive to conserved variables
          mb->peos->PrimitiveToConserved(
            mb->phydro->w, mb->pfield->bcc, mb->phydro->u, mb->pcoord,
            i, i, j, j, k, k);

          // update opacity
          if (NR_RADIATION_ENABLED)
            op_wrapped(mb, mb->phydro->w, i, i, j, j, k, k);
        }
      }
}

void error_bad_param(char const *func, char const *param)
{
  std::stringstream msg;
  msg << "### FATAL ERROR in function [" << func << "]"
      << std::endl
      << "value of " << param << " in input file is unrecognized"
      << std::endl;
  ATHENA_ERROR(msg);
}

//------------------------------------------------------------------------------
//! Refine/Derefine each block to exactly the level of whichever custom_refinement
//! box (if any) it falls in.  Return +1 to refine, –1 to derefine, 0 to leave.
//------------------------------------------------------------------------------

int ProblemRefinementDummy(MeshBlock *pmb) {
  static int once = 0;
  // if (once++ < 10) {
  //   std::cout << "[AMR] ProblemRefinement called on rank " << Globals::my_rank
  //             << " level=" << pmb->loc.level << "\n";
  // }

  int lvl = pmb->loc.level;

  // 1) extents
  Real x1_lo = pmb->pcoord->x1f(pmb->is);
  Real x1_hi = pmb->pcoord->x1f(pmb->ie + 1);
  Real x2_lo = pmb->pcoord->x2f(pmb->js);
  Real x2_hi = pmb->pcoord->x2f(pmb->je + 1);
  Real x3_lo = pmb->pcoord->x3f(pmb->ks);
  Real x3_hi = pmb->pcoord->x3f(pmb->ke + 1);

  // if (once <= 10) {
  //   std::cout << "[AMR] block x1=[" << x1_lo << "," << x1_hi << "]"
  //             << " x2=[" << x2_lo << "," << x2_hi << "]"
  //             << " x3=[" << x3_lo << "," << x3_hi << "]\n";
  // }

  // 2) find highest target among overlapping boxes
  bool found = false;
  int  target = lvl;

  for (auto &box : g_refine_boxes) {
    bool overlap =
      !(x1_hi < box.x1min || x1_lo > box.x1max ||
        x2_hi < box.x2min || x2_lo > box.x2max ||
        x3_hi < box.x3min || x3_lo > box.x3max);

    // if (overlap && Globals::my_rank == 0) {
    //   std::cout << "[AMR] OVERLAP lvl=" << lvl
    //             << " block x1=[" << x1_lo << "," << x1_hi << "]"
    //             << " x2=[" << x2_lo << "," << x2_hi << "]"
    //             << " x3=[" << x3_lo << "," << x3_hi << "]"
    //             << " target=" << box.level << "\n";
    // }

    if (!overlap) {
      continue;
    }
    if (!found) {
      target = box.level;
      found  = true;
    } else {
      target = std::max(target, box.level);
    }
  }

  // // 3) if no custom box found, use the default
  // if (!found) {
  //   target = default_refinement_level;
  // }

  // 3) if no custom box found, do nothing
  if (!found) {
    std::cout << "[AMR] NO OVERLAP rank " << Globals::my_rank;
    return 0;
  }

  // 4) issue refine/derefine
  if (lvl < target)       return +1;
  else if (lvl > target)  return -1;
  else                     return  0;
}

int ProblemRefinement(MeshBlock *pmb) {
  Real x1min = 7.5;  // injection at r=7.7 
  Real x1max = 8.6;  // injection at r=7.7
  Real y1min = 1.25;  // injection at theta=1.32
  Real y1max = 1.97;  // injection at theta=1.32
  Real z1min = 1.42;  // injection at phi=1.52
  Real z1max = 1.62;  // injection at phi=1.52

  Real x2min = 8.2;
  Real x2max = 10.4;
  Real y2min = 1.97;
  Real y2max = 2.5;
  Real z2min = 1.38;
  Real z2max = 1.62;

  
  // Get block extents using face positions (not cell centers)
  Real x1_lo_1 = pmb->pcoord->x1f(pmb->is);
  Real x1_hi_1 = pmb->pcoord->x1f(pmb->ie + 1);
  Real x2_lo_1 = pmb->pcoord->x2f(pmb->js);
  Real x2_hi_1 = pmb->pcoord->x2f(pmb->je + 1);
  Real x3_lo_1 = pmb->pcoord->x3f(pmb->ks);
  Real x3_hi_1 = pmb->pcoord->x3f(pmb->ke + 1);

  Real x1_lo_2 = pmb->pcoord->x1f(pmb->is);
  Real x1_hi_2 = pmb->pcoord->x1f(pmb->ie + 1);
  Real x2_lo_2 = pmb->pcoord->x2f(pmb->js);
  Real x2_hi_2 = pmb->pcoord->x2f(pmb->je + 1);
  Real x3_lo_2 = pmb->pcoord->x3f(pmb->ks);
  Real x3_hi_2 = pmb->pcoord->x3f(pmb->ke + 1);
  
  // Check if block overlaps the box
  bool overlap_1 = !(x1_hi_1 < x1min || x1_lo_1 > x1max ||
                   x2_hi_1 < y1min || x2_lo_1 > y1max ||
                   x3_hi_1 < z1min || x3_lo_1 > z1max);

  bool overlap_2 = !(x1_hi_2 < x2min || x1_lo_2 > x2max ||
                   x2_hi_2 < y2min || x2_lo_2 > y2max ||
                   x3_hi_2 < z2min || x3_lo_2 > z2max);
  
  if (overlap_1) {
    return +1;
  }
  if (overlap_2) {
    return +1;
  }
  return 0;
}

// int ProblemRefinement(MeshBlock *pmb) {
// return -1;
// }



} // namespace

// Added MeshBlock::InitUserMeshBlockData to enroll opacity function if NR_RADIATION_ENABLED
void MeshBlock::InitUserMeshBlockData(ParameterInput *in) {
#ifdef NR_RADIATION_ENABLED
  // only register if *truly* running with radiation
  if (pnrrad && in->GetOrAddInteger("radiation","nmu",0)>0) {
    pnrrad->EnrollOpacityFunction(op_wrapper);
  }
  #endif
}