// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vrtlsim_shim.h for the primary calling header

#include "Vrtlsim_shim__pch.h"
#include "Vrtlsim_shim_VX_schedule_if.h"

std::string VL_TO_STRING(const Vrtlsim_shim_VX_schedule_if* obj) {
    VL_DEBUG_IF(VL_DBG_MSGF("+                  Vrtlsim_shim_VX_schedule_if::VL_TO_STRING\n"); );
    // Body
    return (obj ? obj->name() : "null");
}
