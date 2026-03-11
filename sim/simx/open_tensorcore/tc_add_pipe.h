#pragma once
#include <cstdint>
#include "config_register.h"
#include "fadd_s1.h"
#include "fadd_s2.h"
struct  add_pipe{
struct {
    uint32_t a_in;
    uint32_t b_in;
    bool input_valid;
}add_input;
// ===================================
//      pipeline registers
// ===================================
// register 1

struct {
fadd_s1_out s1_result;
uint32_t c_in;
bool valid;
} r1;

// register 2
struct {
uint32_t result;
uint32_t c_in;
bool valid;
} r2;

void reset(){
    r1.valid = false;
    r2.valid = false;
    add_input.input_valid = false;
}

// ===================================
//              接口
// ===================================
bool out_valid() const{
    return r2.valid;
}

bool in_ready(bool out_ready) const{
    bool s2_ready = out_ready || !r2.valid;
    bool s1_ready = s2_ready || !r1.valid;

    return s1_ready;
}

const uint32_t& out_data() const{
    return r2.result;
}

void tick(bool out_ready ,const Config& g_cfg ,uint32_t c_in){
bool s2_ready = out_ready || !r2.valid;
bool s1_ready = s2_ready || !r1.valid;

if (s2_ready){
    if (r1.valid){
        r2.result = fadd_s2( r1.s1_result, 5, 8);
        r2.c_in = r1.c_in;
        r2.valid = true;
    }else{
        r2.valid = false;
    }
}

if (s1_ready){
    if (add_input.input_valid){
        r1.s1_result = fadd_s1(add_input.a_in, add_input.b_in, 5, 8, 8, g_cfg.rm);
        r1.c_in = c_in;
        r1.valid = true;
    }else{
        r1.valid = false;
    }
}

}
};

struct  add_pipe_fp22{

struct {
    uint32_t a_in;
    uint32_t b_in;
    bool input_valid;
}add_input;
// ===================================
//      pipeline registers
// ===================================
// register 1

struct {
fadd_s1_out s1_result;
bool valid;
} r1;

// register 2
struct {
uint32_t result;
bool valid;
} r2;

void reset(){
    r1.valid = false;
    r2.valid = false;
    add_input.input_valid = false;
}

// ===================================
//              接口
// ===================================
bool out_valid() const{
    return r2.valid;
}

bool in_ready(bool out_ready) const{
    bool s2_ready = out_ready || !r2.valid;
    bool s1_ready = s2_ready || !r1.valid;

    return s1_ready;
}

const uint32_t& out_data() const{
    return r2.result;
}

void tick(bool out_ready  ,const Config& g_cfg){
bool s2_ready = out_ready || !r2.valid;
bool s1_ready = s2_ready || !r1.valid;

if (s2_ready){
    if (r1.valid){
        r2.result = fadd_s2( r1.s1_result, 8, 14);
        r2.valid = true;
    }else{
        r2.valid = false;
    }
}

if (s1_ready){
    if (add_input.input_valid){
        r1.s1_result = fadd_s1(add_input.a_in, add_input.b_in, 8, 14, 14, g_cfg.rm);
        r1.valid = true;
    }else{
        r1.valid = false;
    }
}

}
};
