#pragma once

#include <array>
#include <cstdint>
#include "config_register.h"
#include "tc_add_pipe.h"

struct add4_pipe_fp22 {
struct {
    std::array<uint32_t, 4> in = {};
    bool input_valid = false;
    TensorCoreMeta meta = {};
} add_input;

struct {
    std::array<uint32_t, 4> in = {};
    uint32_t passthrough = 0;
    bool valid = false;
    TensorCoreMeta meta = {};
} r1;

struct {
    uint32_t sum0 = 0;
    uint32_t sum1 = 0;
    uint32_t passthrough = 0;
    bool valid = false;
    TensorCoreMeta meta = {};
} r2;

struct {
    uint32_t sum = 0;
    uint32_t passthrough = 0;
    bool valid = false;
    TensorCoreMeta meta = {};
} r3;

struct {
    uint32_t result = 0;
    uint32_t passthrough = 0;
    bool valid = false;
    TensorCoreMeta meta = {};
} r4;

void reset() {
    add_input.in = {};
    add_input.input_valid = false;
    add_input.meta = {};
    r1 = {};
    r2 = {};
    r3 = {};
    r4 = {};
}

bool out_valid() const {
    return r4.valid;
}

bool active() const {
    return add_input.input_valid || r1.valid || r2.valid || r3.valid || r4.valid;
}

bool in_ready(bool out_ready) const {
    const bool s4_ready = out_ready || !r4.valid;
    const bool s3_ready = s4_ready || !r3.valid;
    const bool s2_ready = s3_ready || !r2.valid;
    const bool s1_ready = s2_ready || !r1.valid;
    return s1_ready;
}

const uint32_t& out_data() const {
    return r4.result;
}

uint32_t out_passthrough() const {
    return r4.passthrough;
}

const TensorCoreMeta& out_meta() const {
    return r4.meta;
}

void tick(bool out_ready, const Config& g_cfg, uint32_t passthrough) {
    const bool s4_ready = out_ready || !r4.valid;
    const bool s3_ready = s4_ready || !r3.valid;
    const bool s2_ready = s3_ready || !r2.valid;
    const bool s1_ready = s2_ready || !r1.valid;

    if (s4_ready) {
        if (r3.valid) {
            r4.result = r3.sum;
            r4.passthrough = r3.passthrough;
            r4.valid = true;
            r4.meta = r3.meta;
        } else {
            r4 = {};
        }
    }

    if (s3_ready) {
        if (r2.valid) {
            r3.sum = fp22_add_bits(r2.sum0, r2.sum1, g_cfg.rm);
            r3.passthrough = r2.passthrough;
            r3.valid = true;
            r3.meta = r2.meta;
        } else {
            r3 = {};
        }
    }

    if (s2_ready) {
        if (r1.valid) {
            r2.sum0 = fp22_add_bits(r1.in[0], r1.in[1], g_cfg.rm);
            r2.sum1 = fp22_add_bits(r1.in[2], r1.in[3], g_cfg.rm);
            r2.passthrough = r1.passthrough;
            r2.valid = true;
            r2.meta = r1.meta;
        } else {
            r2 = {};
        }
    }

    if (s1_ready) {
        if (add_input.input_valid) {
            r1.in = add_input.in;
            r1.passthrough = passthrough;
            r1.valid = true;
            r1.meta = add_input.meta;
        } else {
            r1 = {};
        }
    }
}
};
