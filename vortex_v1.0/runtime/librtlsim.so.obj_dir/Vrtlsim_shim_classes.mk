# Verilated -*- Makefile -*-
# DESCRIPTION: Verilator output: Make include file with class lists
#
# This file lists generated Verilated files, for including in higher level makefiles.
# See Vrtlsim_shim.mk for the caller.

### Switches...
# C11 constructs required?  0/1 (always on now)
VM_C11 = 1
# Timing enabled?  0/1
VM_TIMING = 0
# Coverage output mode?  0/1 (from --coverage)
VM_COVERAGE = 0
# Parallel builds?  0/1 (from --output-split)
VM_PARALLEL_BUILDS = 1
# Tracing output mode?  0/1 (from --trace/--trace-fst)
VM_TRACE = 0
# Tracing output mode in VCD format?  0/1 (from --trace)
VM_TRACE_VCD = 0
# Tracing output mode in FST format?  0/1 (from --trace-fst)
VM_TRACE_FST = 0

### Object file lists...
# Generated module classes, fast-path, compile with highest optimization
VM_CLASSES_FAST += \
	Vrtlsim_shim \
	Vrtlsim_shim___024root__DepSet_hcd27c75d__0 \
	Vrtlsim_shim___024root__DepSet_hcd27c75d__1 \
	Vrtlsim_shim___024root__DepSet_hcd27c75d__2 \
	Vrtlsim_shim___024root__DepSet_hcd27c75d__3 \
	Vrtlsim_shim___024root__DepSet_h76d00735__0 \
	Vrtlsim_shim___024unit__DepSet_hace02784__0 \
	Vrtlsim_shim_VX_trace_pkg__DepSet_h9df89012__0 \
	Vrtlsim_shim_VX_schedule_if__DepSet_hb0fc5ff1__0 \
	Vrtlsim_shim_VX_fetch_if__DepSet_h43c9af56__0 \
	Vrtlsim_shim_VX_issue_sched_if__DepSet_h5127c572__0 \
	Vrtlsim_shim_VX_decode_if__DepSet_hec0ff856__0 \
	Vrtlsim_shim_VX_mem_bus_if__D40_T7__DepSet_h5f5c0e7f__0 \
	Vrtlsim_shim_VX_mem_bus_if__D4_T3__DepSet_h7eef97f1__0 \
	Vrtlsim_shim_VX_mem_bus_if__D40_T5__DepSet_hcdc2fd35__0 \
	Vrtlsim_shim_VX_mem_bus_if__D10_T3__DepSet_h8bf2585e__0 \
	Vrtlsim_shim_VX_mem_bus_if__D40_T6__DepSet_h3ed43ac6__0 \
	Vrtlsim_shim_VX_lsu_mem_if__N4_D4_T2__DepSet_h8213916b__0 \
	Vrtlsim_shim_VX_mem_bus_if__D4_T2__DepSet_hc46aa700__0 \
	Vrtlsim_shim_VX_lsu_mem_if__D10_T3__DepSet_h1be1a364__0 \
	Vrtlsim_shim_VX_result_if__Tz84__DepSet_h130f942e__0 \
	Vrtlsim_shim_VX_execute_if__Tz91__DepSet_hd01fb324__0 \
	Vrtlsim_shim_VX_result_if__Tz92__DepSet_hb0ec42a8__0 \
	Vrtlsim_shim_VX_execute_if__Tz96__DepSet_h53c33c0b__0 \
	Vrtlsim_shim_VX_result_if__Tz97__DepSet_h79d0f9db__0 \
	Vrtlsim_shim_VX_result_if__Tz101__DepSet_he763d143__0 \
	Vrtlsim_shim_VX_writeback_if__DepSet_hbec6a2fd__0 \
	Vrtlsim_shim_VX_ibuffer_if__DepSet_h9e6b7012__0 \
	Vrtlsim_shim_VX_scoreboard_if__DepSet_hb42751bf__0 \
	Vrtlsim_shim_VX_operands_if__DepSet_h63796706__0 \

# Generated module classes, non-fast-path, compile with low/medium optimization
VM_CLASSES_SLOW += \
	Vrtlsim_shim__ConstPool_0 \
	Vrtlsim_shim___024root__Slow \
	Vrtlsim_shim___024root__DepSet_hcd27c75d__0__Slow \
	Vrtlsim_shim___024root__DepSet_hcd27c75d__1__Slow \
	Vrtlsim_shim___024root__DepSet_h76d00735__0__Slow \
	Vrtlsim_shim___024unit__Slow \
	Vrtlsim_shim___024unit__DepSet_h1792a75c__0__Slow \
	Vrtlsim_shim_VX_trace_pkg__Slow \
	Vrtlsim_shim_VX_trace_pkg__DepSet_h2dacae97__0__Slow \
	Vrtlsim_shim_VX_schedule_if__Slow \
	Vrtlsim_shim_VX_schedule_if__DepSet_hb0fc5ff1__0__Slow \
	Vrtlsim_shim_VX_fetch_if__Slow \
	Vrtlsim_shim_VX_fetch_if__DepSet_h43c9af56__0__Slow \
	Vrtlsim_shim_VX_issue_sched_if__Slow \
	Vrtlsim_shim_VX_issue_sched_if__DepSet_h5127c572__0__Slow \
	Vrtlsim_shim_VX_decode_if__Slow \
	Vrtlsim_shim_VX_decode_if__DepSet_hec0ff856__0__Slow \
	Vrtlsim_shim_VX_mem_bus_if__D40_T7__Slow \
	Vrtlsim_shim_VX_mem_bus_if__D40_T7__DepSet_h5f5c0e7f__0__Slow \
	Vrtlsim_shim_VX_mem_bus_if__D4_T3__Slow \
	Vrtlsim_shim_VX_mem_bus_if__D4_T3__DepSet_h7eef97f1__0__Slow \
	Vrtlsim_shim_VX_mem_bus_if__D40_T5__Slow \
	Vrtlsim_shim_VX_mem_bus_if__D40_T5__DepSet_hcdc2fd35__0__Slow \
	Vrtlsim_shim_VX_mem_bus_if__D10_T3__Slow \
	Vrtlsim_shim_VX_mem_bus_if__D10_T3__DepSet_h8bf2585e__0__Slow \
	Vrtlsim_shim_VX_mem_bus_if__D40_T6__Slow \
	Vrtlsim_shim_VX_mem_bus_if__D40_T6__DepSet_h3ed43ac6__0__Slow \
	Vrtlsim_shim_VX_lsu_mem_if__N4_D4_T2__Slow \
	Vrtlsim_shim_VX_lsu_mem_if__N4_D4_T2__DepSet_h8213916b__0__Slow \
	Vrtlsim_shim_VX_mem_bus_if__D4_T2__Slow \
	Vrtlsim_shim_VX_mem_bus_if__D4_T2__DepSet_hc46aa700__0__Slow \
	Vrtlsim_shim_VX_lsu_mem_if__D10_T3__Slow \
	Vrtlsim_shim_VX_lsu_mem_if__D10_T3__DepSet_h1be1a364__0__Slow \
	Vrtlsim_shim_VX_result_if__Tz84__Slow \
	Vrtlsim_shim_VX_result_if__Tz84__DepSet_h130f942e__0__Slow \
	Vrtlsim_shim_VX_execute_if__Tz91__Slow \
	Vrtlsim_shim_VX_execute_if__Tz91__DepSet_hd01fb324__0__Slow \
	Vrtlsim_shim_VX_result_if__Tz92__Slow \
	Vrtlsim_shim_VX_result_if__Tz92__DepSet_hb0ec42a8__0__Slow \
	Vrtlsim_shim_VX_execute_if__Tz96__Slow \
	Vrtlsim_shim_VX_execute_if__Tz96__DepSet_h53c33c0b__0__Slow \
	Vrtlsim_shim_VX_result_if__Tz97__Slow \
	Vrtlsim_shim_VX_result_if__Tz97__DepSet_h79d0f9db__0__Slow \
	Vrtlsim_shim_VX_result_if__Tz101__Slow \
	Vrtlsim_shim_VX_result_if__Tz101__DepSet_he763d143__0__Slow \
	Vrtlsim_shim_VX_writeback_if__Slow \
	Vrtlsim_shim_VX_writeback_if__DepSet_hbec6a2fd__0__Slow \
	Vrtlsim_shim_VX_ibuffer_if__Slow \
	Vrtlsim_shim_VX_ibuffer_if__DepSet_h9e6b7012__0__Slow \
	Vrtlsim_shim_VX_scoreboard_if__Slow \
	Vrtlsim_shim_VX_scoreboard_if__DepSet_hb42751bf__0__Slow \
	Vrtlsim_shim_VX_operands_if__Slow \
	Vrtlsim_shim_VX_operands_if__DepSet_h63796706__0__Slow \

# Generated support classes, fast-path, compile with highest optimization
VM_SUPPORT_FAST += \
	Vrtlsim_shim__Dpi \

# Generated support classes, non-fast-path, compile with low/medium optimization
VM_SUPPORT_SLOW += \
	Vrtlsim_shim__Syms \

# Global classes, need linked once per executable, fast-path, compile with highest optimization
VM_GLOBAL_FAST += \
	verilated \
	verilated_dpi \
	verilated_threads \

# Global classes, need linked once per executable, non-fast-path, compile with low/medium optimization
VM_GLOBAL_SLOW += \


# Verilated -*- Makefile -*-
