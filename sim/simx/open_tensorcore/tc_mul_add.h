#pragma once
#include <array>
#include <cstdint>
#include "fp_types.h"
#include "tc_mul_pipe.h"
#include "tc_add_pipe.h"
#include "tc_add4_pipe_fp22.h"
#include "fp22_to_fp16.h"
struct tc_mul_add{
std::array<mul_pipe, 8> mul_array;
std::array<add4_pipe_fp22, 2> add4_level;
add_pipe_fp22 merge_add;
add_pipe_fp22 final_add;

struct {
    uint32_t a_in[8];
    uint32_t b_in[8];
    uint32_t c_in;
    PrecisionType prec;
    bool input_valid;
    TensorCoreMeta meta;
} mul_add_input;


void reset() {
for (int i = 0; i < 8; i++) mul_array[i].reset();
for (int i = 0; i < 2; i++) add4_level[i].reset();
merge_add.reset();
final_add.reset();
r2.c_in = 0;
r2.fp22_result = 0;
r2.result = 0;
r2.valid = false;
r2.meta = {};
mul_add_input.meta = {};
mul_add_input.input_valid = false;
    }


struct {
uint32_t c_in;
uint32_t fp22_result;
uint32_t result;
bool valid;
TensorCoreMeta meta;
} r2;


bool out_valid() const{
    return r2.valid;
}

bool active() const{
    if (mul_add_input.input_valid || r2.valid) {
        return true;
    }
    for (const auto& mul : mul_array) {
        if (mul.active()) {
            return true;
        }
    }
    for (const auto& add : add4_level) {
        if (add.active()) {
            return true;
        }
    }
    return merge_add.active() || final_add.active();
}

bool in_ready(bool out_ready) const{
    bool s8_ready = true;
    s8_ready = out_ready || !r2.valid;
    bool s7_ready = true;
    s7_ready &= final_add.in_ready(s8_ready);
    bool s6_ready = true;
    s6_ready &= merge_add.in_ready(s7_ready);
    bool s5_ready = true;
    for (int i = 0; i < 2; ++i) s5_ready &= add4_level[i].in_ready(s6_ready);
    bool s4_ready = true;
    for (int i = 0; i < 8; ++i) s4_ready &= mul_array[i].in_ready(s5_ready);
    return s4_ready;
}

const uint32_t& out_data() const{
    return r2.result;
}

const uint32_t& out_fp22() const{
    return r2.fp22_result;
}

const TensorCoreMeta& out_meta() const{
    return r2.meta;
}



void tick(bool out_ready ,const Config& g_cfg ){
    
    bool s8_ready = true;
    s8_ready = out_ready || !r2.valid;
    bool s7_ready = true;
    s7_ready &= final_add.in_ready(s8_ready);
    bool s6_ready = true;
    s6_ready &= merge_add.in_ready(s7_ready);
    bool s5_ready = true;
    for (int i = 0; i < 2; ++i) s5_ready &= add4_level[i].in_ready(s6_ready);
    bool s4_ready = true;
    for (int i = 0; i < 8; ++i) s4_ready &= mul_array[i].in_ready(s5_ready);


bool r7_valid = true;
r7_valid &= final_add.out_valid();
auto out_prec = g_cfg.out_precisions.empty() ? PREC_FP16 : g_cfg.out_precisions.at(0);
if (s8_ready){
if (r7_valid){
uint32_t fp22 = final_add.out_data();
const auto& meta = final_add.out_meta();
    r2.fp22_result = fp22;
switch (out_prec) {
case PREC_FP32:
    r2.result = fp22_to_fp32(fp22);
    break;
case PREC_FP8_E4M3:
    r2.result = fp22_to_fp8_e4m3(fp22);
    break;
case PREC_FP8_E5M2:
    r2.result = fp22_to_fp8_e5m2(fp22);
    break;
case PREC_FP16:
default:
    r2.result = fp22_to_fp16(fp22);
    break;
}
r2.valid = true;
    r2.meta = meta;
    }else{
r2.valid = false;
    r2.meta = {};
    }
}


bool r6_valid = true;
r6_valid &= merge_add.out_valid();
if (s7_ready){
if (r6_valid){
    uint32_t c_fp22 = convert_c_to_fp22(merge_add.out_passthrough(), out_prec);

    final_add.add_input.a_in = merge_add.out_data();
    final_add.add_input.b_in = c_fp22;
    final_add.add_input.input_valid = true;
    final_add.add_input.meta = merge_add.out_meta();
    }else{
    final_add.add_input.input_valid = false;
    final_add.add_input.meta = {};
    }
    final_add.tick(s8_ready ,g_cfg);
}


bool r5_valid = true;
for(int i = 0; i < 2; ++i) r5_valid &= add4_level[i].out_valid();
if (s6_ready){
if (r5_valid){
    merge_add.add_input.a_in = add4_level[0].out_data();
    merge_add.add_input.b_in = add4_level[1].out_data();
    merge_add.add_input.input_valid = true;
    merge_add.add_input.meta = add4_level[0].out_meta();
    }else{
    merge_add.add_input.input_valid = false;
    merge_add.add_input.meta = {};
    }
    merge_add.tick(s7_ready ,g_cfg ,add4_level[0].out_passthrough());
}


bool r4_valid = true;
for(int i = 0; i < 8; ++i) r4_valid &= mul_array[i].out_valid();
if (s5_ready){
if (r4_valid){
    for(int i = 0; i < 2; ++i) {
    const int base = i * 4;
    add4_level[i].add_input.in[0] = mul_array[base + 0].out_data();
    add4_level[i].add_input.in[1] = mul_array[base + 1].out_data();
    add4_level[i].add_input.in[2] = mul_array[base + 2].out_data();
    add4_level[i].add_input.in[3] = mul_array[base + 3].out_data();
    add4_level[i].add_input.input_valid = true;
    add4_level[i].add_input.meta = mul_array[base].out_meta();
    add4_level[i].tick(s6_ready ,g_cfg ,mul_array[base].r3.c_in);
    }
}else{
    for(int i = 0; i < 2; ++i) {
        const int base = i * 4;
        add4_level[i].add_input.input_valid = false;
        add4_level[i].add_input.meta = {};
        add4_level[i].tick(s6_ready ,g_cfg ,mul_array[base].r3.c_in);
    }
    }
}

if (s4_ready){
if (mul_add_input.input_valid){
    for(int i = 0; i < 8; ++i) {
    mul_array[i].mul_input.a_fp9 = mul_add_input.a_in[i];
    mul_array[i].mul_input.b_fp9 = mul_add_input.b_in[i];
    mul_array[i].mul_input.input_valid = true;
    mul_array[i].mul_input.meta = mul_add_input.meta;
    mul_array[i].tick(s5_ready ,g_cfg ,mul_add_input.c_in);
    }
}else{
    for(int i = 0; i < 8; ++i) {
        mul_array[i].mul_input.input_valid = false;
        mul_array[i].mul_input.meta = {};
        mul_array[i].tick(s5_ready ,g_cfg ,mul_add_input.c_in);
    }
    }
}


}

};
