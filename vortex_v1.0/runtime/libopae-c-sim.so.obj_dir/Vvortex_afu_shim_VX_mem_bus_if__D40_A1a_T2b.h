// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vvortex_afu_shim.h for the primary calling header

#ifndef VERILATED_VVORTEX_AFU_SHIM_VX_MEM_BUS_IF__D40_A1A_T2B_H_
#define VERILATED_VVORTEX_AFU_SHIM_VX_MEM_BUS_IF__D40_A1A_T2B_H_  // guard

#include "verilated.h"


class Vvortex_afu_shim__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vvortex_afu_shim_VX_mem_bus_if__D40_A1a_T2b final : public VerilatedModule {
  public:

    // DESIGN SPECIFIC STATE
    VlWide<21>/*648:0*/ req_data;
    VlWide<18>/*554:0*/ rsp_data;

    // INTERNAL VARIABLES
    Vvortex_afu_shim__Syms* const vlSymsp;

    // CONSTRUCTORS
    Vvortex_afu_shim_VX_mem_bus_if__D40_A1a_T2b(Vvortex_afu_shim__Syms* symsp, const char* v__name);
    ~Vvortex_afu_shim_VX_mem_bus_if__D40_A1a_T2b();
    VL_UNCOPYABLE(Vvortex_afu_shim_VX_mem_bus_if__D40_A1a_T2b);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};

std::string VL_TO_STRING(const Vvortex_afu_shim_VX_mem_bus_if__D40_A1a_T2b* obj);

#endif  // guard
