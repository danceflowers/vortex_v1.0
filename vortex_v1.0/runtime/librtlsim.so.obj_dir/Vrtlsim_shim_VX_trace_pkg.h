// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vrtlsim_shim.h for the primary calling header

#ifndef VERILATED_VRTLSIM_SHIM_VX_TRACE_PKG_H_
#define VERILATED_VRTLSIM_SHIM_VX_TRACE_PKG_H_  // guard

#include "verilated.h"


class Vrtlsim_shim__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vrtlsim_shim_VX_trace_pkg final : public VerilatedModule {
  public:

    // INTERNAL VARIABLES
    Vrtlsim_shim__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vrtlsim_shim_VX_trace_pkg(Vrtlsim_shim__Syms* symsp, const char* v__name);
    ~Vrtlsim_shim_VX_trace_pkg();
    VL_UNCOPYABLE(Vrtlsim_shim_VX_trace_pkg);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
