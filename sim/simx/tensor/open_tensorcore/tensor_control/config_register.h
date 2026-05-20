#pragma once
#include <vector>
#include "fp_types.h"

// Global test/simulation configuration populated by command-line parsing.
struct Config {
    std::vector<PrecisionType> precisions;      // Input precisions to test.
    std::vector<PrecisionType> out_precisions;  // Output precisions to test.
    int  test_id    = 0;       // 0 runs all tests; nonzero selects one test.
    RoundingMode rm = RNE;     // Default: round to nearest-even.
    uint32_t seed   = 0;       // 0 means use the current time.
    bool show_help  = false;   // Print command-line help.
};

// Metadata carried with one 8x8 TensorCore primitive through the compute
// pipeline. It identifies the originating async operation, selects operand
// formats, and carries sparse metadata needed by the primitive datapath.
struct TensorCoreMeta {
    uint32_t wgid = 0;                             // Warpgroup ID
    uint32_t async_id = 0;                          // Async operation tracking ID.
    uint32_t k_phase_id = 0;                        // K-phase index within a macro MMA.
    uint32_t b_slot_id = 0;                         // B operand slot.
    uint32_t c_slot_id = 0;                         // C operand slot.
    uint32_t c_subtile_id = 0;                      // Output subtile index.
    PrecisionType in_prec = PREC_FP9;               // A/B input precision.
    PrecisionType out_prec = PREC_FP16;             // D writeback precision.
    PrecisionType c_prec = PREC_FP16;               // C bypass precision.
    uint8_t c_bypass_is_fp22 = 0;                   // C bypass is already FP22.
    uint8_t use_cmem_operand1 = 0;                  // Use operand1 instead of C.
    uint8_t sparse_mode = 0;                        // 0=dense, 1=2:4, 2=1:4
    uint8_t sparse_meta[16] = {};                   // 16B sparse metadata line.
    bool valid = false;                             // Metadata valid bit.
};

// Global configuration instance initialized by standalone test mains.
static Config g_cfg;
