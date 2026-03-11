// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vrtlsim_shim.h for the primary calling header

#ifndef VERILATED_VRTLSIM_SHIM___024UNIT_H_
#define VERILATED_VRTLSIM_SHIM___024UNIT_H_  // guard

#include "verilated.h"


class Vrtlsim_shim__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vrtlsim_shim___024unit final : public VerilatedModule {
  public:

    // INTERNAL VARIABLES
    Vrtlsim_shim__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vrtlsim_shim___024unit(Vrtlsim_shim__Syms* symsp, const char* v__name);
    ~Vrtlsim_shim___024unit();
    VL_UNCOPYABLE(Vrtlsim_shim___024unit);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
