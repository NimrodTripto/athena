// src/tools/read_pressure_extrema.cpp

#include <limits>
#include <iostream>
#include <string>

#include "../athena.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " restart.rst gamma\n";
    return 1;
  }

  std::string rstfile = argv[1];
  Real gamma = std::stod(argv[2]);

  // Mimic "athena -r restart.rst"
  char *args[] = {
    const_cast<char*>("athena"),
    const_cast<char*>("-r"),
    const_cast<char*>(rstfile.c_str())
  };
  ParameterInput *pin = new ParameterInput(3, args);
  Mesh *pmesh = new Mesh(pin);

  Real pmin =  std::numeric_limits<Real>::infinity();
  Real pmax = -std::numeric_limits<Real>::infinity();

  // Loop over all meshblocks, compute pressure from conserved vars
  for (auto &pmb : pmesh->my_blocks) {
    auto &u = pmb->phydro->u;  // u(IDN..IEN, idx)
    int nc = pmb->block_size.nx1 *
             pmb->block_size.nx2 *
             pmb->block_size.nx3;
    for (int idx = 0; idx < nc; ++idx) {
      Real rho = u(IDN, idx);
      Real m1  = u(IM1, idx);
      Real m2  = u(IM2, idx);
      Real m3  = u(IM3, idx);
      Real E   = u(IEN, idx);
      Real ke  = 0.5 * (m1*m1 + m2*m2 + m3*m3) / rho;
      Real p   = (gamma - 1.0) * (E - ke);
      pmin = std::fmin(pmin, p);
      pmax = std::fmax(pmax, p);
    }
  }

  std::cout << "p_min = " << pmin << ", p_max = " << pmax << "\n";
  return 0;
}
