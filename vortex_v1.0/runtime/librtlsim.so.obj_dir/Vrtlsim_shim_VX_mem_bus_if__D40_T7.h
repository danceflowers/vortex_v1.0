// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vrtlsim_shim.h for the primary calling header

#ifndef VERILATED_VRTLSIM_SHIM_VX_MEM_BUS_IF__D40_T7_H_
#define VERILATED_VRTLSIM_SHIM_VX_MEM_BUS_IF__D40_T7_H_  // guard

#include "verilated.h"


class Vrtlsim_shim__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vrtlsim_shim_VX_mem_bus_if__D40_T7 final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VlWide<20>/*612:0*/ req_data;
    VlWide<17>/*518:0*/ rsp_data;

    // INTERNAL VARIABLES
    Vrtlsim_shim__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vrtlsim_shim_VX_mem_bus_if__D40_T7(Vrtlsim_shim__Syms* symsp, const char* v__name);
    ~Vrtlsim_shim_VX_mem_bus_if__D40_T7();
    VL_UNCOPYABLE(Vrtlsim_shim_VX_mem_bus_if__D40_T7);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};

std::string VL_TO_STRING(const Vrtlsim_shim_VX_mem_bus_if__D40_T7* obj);

#endif  // guard
