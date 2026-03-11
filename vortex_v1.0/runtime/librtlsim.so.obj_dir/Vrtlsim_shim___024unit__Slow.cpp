// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vrtlsim_shim.h for the primary calling header

#include "Vrtlsim_shim__pch.h"
#include "Vrtlsim_shim__Syms.h"
#include "Vrtlsim_shim___024unit.h"

void Vrtlsim_shim___024unit___ctor_var_reset(Vrtlsim_shim___024unit* vlSelf);

Vrtlsim_shim___024unit::Vrtlsim_shim___024unit(Vrtlsim_shim__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vrtlsim_shim___024unit___ctor_var_reset(this);
}

void Vrtlsim_shim___024unit::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
}

Vrtlsim_shim___024unit::~Vrtlsim_shim___024unit() {
}
