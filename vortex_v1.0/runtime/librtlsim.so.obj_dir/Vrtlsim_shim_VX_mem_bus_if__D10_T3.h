// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vrtlsim_shim.h for the primary calling header

#ifndef VERILATED_VRTLSIM_SHIM_VX_MEM_BUS_IF__D10_T3_H_
#define VERILATED_VRTLSIM_SHIM_VX_MEM_BUS_IF__D10_T3_H_  // guard

#include "verilated.h"


class Vrtlsim_shim__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vrtlsim_shim_VX_mem_bus_if__D10_T3 final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    CData/*0:0*/ req_valid;
    VlWide<6>/*178:0*/ req_data;

    // INTERNAL VARIABLES
    Vrtlsim_shim__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vrtlsim_shim_VX_mem_bus_if__D10_T3(Vrtlsim_shim__Syms* symsp, const char* v__name);
    ~Vrtlsim_shim_VX_mem_bus_if__D10_T3();
    VL_UNCOPYABLE(Vrtlsim_shim_VX_mem_bus_if__D10_T3);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};

std::string VL_TO_STRING(const Vrtlsim_shim_VX_mem_bus_if__D10_T3* obj);

#endif  // guard
