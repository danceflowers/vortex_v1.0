// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vrtlsim_shim.h for the primary calling header

#include "Vrtlsim_shim__pch.h"
#include "Vrtlsim_shim_VX_mem_bus_if__D40_T5.h"

VL_ATTR_COLD void Vrtlsim_shim_VX_mem_bus_if__D40_T5___ctor_var_reset(Vrtlsim_shim_VX_mem_bus_if__D40_T5* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vrtlsim_shim__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+                      Vrtlsim_shim_VX_mem_bus_if__D40_T5___ctor_var_reset\n"); );
    auto &vlSelfRef = std::ref(*vlSelf).get();
    // Body
    VL_RAND_RESET_W(611, vlSelf->req_data);
    VL_RAND_RESET_W(517, vlSelf->rsp_data);
}
