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

static Config g_cfg;