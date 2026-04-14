//========================================================================================
//! \file restart_scalars_load.cpp
//! \brief default initialization for optional passive-scalar restart loading
//========================================================================================

#include "restart_scalars_load.hpp"

#if NSCALARS > 0
bool rs_load_scalar[NSCALARS];

namespace {
struct DefaultRsLoadScalar {
  DefaultRsLoadScalar() {
    for (int n = 0; n < NSCALARS; ++n) {
      rs_load_scalar[n] = true;
    }
  }
} default_rs_load_scalar;
}  // namespace
#endif
