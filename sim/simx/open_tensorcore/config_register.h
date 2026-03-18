#pragma once
#include <vector>
#include "fp_types.h"
// =============================================================================
// Global configuration (populated from command-line arguments)
// =============================================================================
struct Config {
    std::vector<PrecisionType> precisions;
    std::vector<PrecisionType> out_precisions;
    int  test_id    = 0;       // 0 = all, 1-6 = specific test
    RoundingMode rm = RNE;
    uint32_t seed   = 0;       // 0 = use time
    bool show_help  = false;
};

struct TensorCoreMeta {
    uint32_t wid = 0;
    uint32_t async_id = 0;
    uint32_t slot_id = 0;
    uint32_t c_subtile_id = 0;
    bool valid = false;
};

static Config g_cfg;
