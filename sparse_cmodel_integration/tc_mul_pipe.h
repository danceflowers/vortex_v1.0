#pragma once
#include <cstdint>
#include "config_register.h"
#include "fmul_s1.h"
#include "fmul_s2.h"
#include "fmul_s3.h"
struct  mul_pipe{

struct {
    uint32_t a_fp9;
    uint32_t b_fp9;
    bool input_valid;
} mul_input;

// ===================================
//      pipeline registers
// ===================================
// register 1

struct {
fmul_s1_out s1_result;
uint32_t a_val;
uint32_t b_val;
uint32_t c_in;
bool valid;
} r1;

// register 2
struct {
fmul_s2_out s2_result;
uint32_t c_in;
bool valid;
} r2;

// register 3
struct {
uint32_t result;
uint32_t c_in;
bool valid;
} r3;

void reset(){
    r1.valid = false;
    r2.valid = false;
    r3.valid = false;
    mul_input.input_valid = false;
}

// ===================================
//              接口
// ===================================
bool out_valid() const{
    return r3.valid;
}

bool in_ready(bool out_ready) const{
    bool s3_ready = out_ready || !r3.valid;
    bool s2_ready = s3_ready || !r2.valid;
    bool s1_ready = s2_ready || !r1.valid;

    return s1_ready;
}

const uint32_t& out_data() const{
    return r3.result;
}

void tick(bool out_ready ,const Config& g_cfg ,uint32_t& c_in){
bool s3_ready = out_ready || !r3.valid;
bool s2_ready = s3_ready || !r2.valid;
bool s1_ready = s2_ready || !r1.valid;
if (s3_ready){
    if (r2.valid){
        r3.result = fmul_s3(r2.s2_result, 5, 8);
        r3.c_in = r2.c_in;
        r3.valid = true;
    }else{
        r3.valid = false;
    }
}

if (s2_ready){
    if (r1.valid){
        r2.s2_result = fmul_s2(r1.a_val, r1.b_val, 5, 8, r1.s1_result);
        r2.c_in = r1.c_in;
        r2.valid = true;
    }else{
        r2.valid = false;
    }
}

if (s1_ready){
    if (mul_input.input_valid){
    uint32_t a_padded = ((uint32_t)mul_input.a_fp9 << 4) ;
    uint32_t b_padded = ((uint32_t)mul_input.b_fp9 << 4) ;
        r1.a_val = a_padded;
        r1.b_val = b_padded;
        r1.c_in = c_in;
        r1.s1_result = fmul_s1(r1.a_val, r1.b_val, 5, 8, g_cfg.rm);
        r1.valid = true;
    }else{
        r1.valid = false;
    }
}


}
};
