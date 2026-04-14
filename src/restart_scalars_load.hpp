#ifndef RESTART_SCALARS_LOAD_HPP_
#define RESTART_SCALARS_LOAD_HPP_
//========================================================================================
//! \file restart_scalars_load.hpp
//! \brief globals for optional passive-scalar restart loading (file may omit slabs)
//========================================================================================

#include "defs.hpp"

#if NSCALARS > 0
//! If `rs_load_scalar[n]` is false, restart data is assumed to omit that scalar slab
//! (contiguous ordering in the file matches non-omitted indices only).
extern bool rs_load_scalar[NSCALARS];
#endif

#endif  // RESTART_SCALARS_LOAD_HPP_
