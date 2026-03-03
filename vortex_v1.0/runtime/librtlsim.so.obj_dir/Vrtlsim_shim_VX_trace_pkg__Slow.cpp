// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vrtlsim_shim.h for the primary calling header

#include "Vrtlsim_shim__pch.h"
#include "Vrtlsim_shim_VX_trace_pkg.h"
#include "Vrtlsim_shim__Syms.h"

void Vrtlsim_shim_VX_trace_pkg___ctor_var_reset(Vrtlsim_shim_VX_trace_pkg* vlSelf);

Vrtlsim_shim_VX_trace_pkg::Vrtlsim_shim_VX_trace_pkg(Vrtlsim_shim__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vrtlsim_shim_VX_trace_pkg___ctor_var_reset(this);
}

void Vrtlsim_shim_VX_trace_pkg::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vrtlsim_shim_VX_trace_pkg::~Vrtlsim_shim_VX_trace_pkg() {
}
