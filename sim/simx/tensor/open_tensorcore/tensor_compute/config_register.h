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
// pipeline. It identifies the originating async operation and the DMem
// accumulation slot used when the primitive retires.
struct TensorCoreMeta {
    uint32_t wid = 0;                              // Issuing warp ID.
    uint32_t async_id = 0;                          // Async operation tracking ID.
    uint32_t accum_phase_id = 0;                    // DMem accumulation phase.
    uint32_t c_subtile_id = 0;                      // Output subtile index.
    bool valid = false;                             // Metadata valid bit.
};

// Global configuration instance initialized by standalone test mains.
static Config g_cfg;
