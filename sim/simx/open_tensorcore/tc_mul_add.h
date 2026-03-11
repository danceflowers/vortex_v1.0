#pragma once
#include <array>
#include <cstdint>
#include "fp_types.h"
#include "tc_mul_pipe.h"
#include "tc_add_pipe.h"
#include "fp22_to_fp16.h"
struct tc_mul_add{
std::array<mul_pipe, 8> mul_array;
std::array<add_pipe, 4> add_level0;
std::array<add_pipe, 2> add_level1;
add_pipe add_level2;
add_pipe_fp22 final_add;

struct {
    uint32_t a_in[8];
    uint32_t b_in[8];
    uint32_t c_in;
    PrecisionType prec;
    bool input_valid;
    
} mul_add_input;


void reset() {
for (int i = 0; i < 8; i++) mul_array[i].reset();
for (int i = 0; i < 4; i++) add_level0[i].reset();
for (int i = 0; i < 2; i++) add_level1[i].reset();
add_level2.reset();
final_add.reset();
r1.c_in = 0;
r1.valid = false;
r2.c_in = 0;
r2.result = 0;
r2.valid = false;
mul_add_input.input_valid = false;
    }


// register 1
struct {
    uint32_t c_in;
uint32_t a_conv_val[8];
uint32_t b_conv_val[8];
bool valid;
} r1;

struct {
uint32_t c_in;
uint32_t result;
bool valid;
} r2;


bool out_valid() const{
    return r2.valid;
}

bool in_ready(bool out_ready) const{
    bool s7_ready = true;
    s7_ready = out_ready || !r2.valid;
    bool s6_ready = true;
    s6_ready &= final_add.in_ready(s7_ready);
    bool s5_ready = true;
    s5_ready &= add_level2.in_ready(s6_ready);
    bool s4_ready = true;
    for (int i = 0; i < 2; ++i) s4_ready &= add_level1[i].in_ready(s5_ready);
    bool s3_ready = true;
    for (int i = 0; i < 4; ++i) s3_ready &= add_level0[i].in_ready(s4_ready);
    bool s2_ready = true;
    for (int i = 0; i < 8; ++i) s2_ready &= mul_array[i].in_ready(s3_ready);
    bool s1_ready = s2_ready || !r1.valid;
    return s1_ready;
}

const uint32_t& out_data() const{
    return r2.result;
}



void tick(bool out_ready ,const Config& g_cfg ){
    
    bool s7_ready = true;
    s7_ready = out_ready || !r2.valid;
    bool s6_ready = true;
    s6_ready &= final_add.in_ready(s7_ready);
    bool s5_ready = true;
    s5_ready &= add_level2.in_ready(s6_ready);
    bool s4_ready = true;
    for (int i = 0; i < 2; ++i) s4_ready &= add_level1[i].in_ready(s5_ready);
    bool s3_ready = true;
    for (int i = 0; i < 4; ++i) s3_ready &= add_level0[i].in_ready(s4_ready);
    bool s2_ready = true;
    for (int i = 0; i < 8; ++i) s2_ready &= mul_array[i].in_ready(s3_ready);
    bool s1_ready = s2_ready || !r1.valid;


bool r6_valid = true;
r6_valid &= final_add.out_valid();
if (s7_ready){
if (r6_valid){
uint32_t fp22 = final_add.out_data();
r2.result = fp22_to_fp16(fp22);
r2.valid = true;
    }else{
r2.valid = false;
    }
}


bool r5_valid = true;
r5_valid &= add_level2.out_valid();
if (s6_ready){
if (r5_valid){
    uint16_t a_fp13 = (uint16_t)(add_level2.out_data() & 0x1FFF);
    uint16_t c_fp16 = (uint16_t)(add_level2.r2.c_in & 0xFFFF);

    uint32_t a_fp22 = fp13_to_fp22(a_fp13);
    uint32_t c_fp22 = fp16_to_fp22(c_fp16);

    final_add.add_input.a_in = a_fp22;
    final_add.add_input.b_in = c_fp22;
    final_add.add_input.input_valid = true;
    }else{
    final_add.add_input.input_valid = false;
    }
    final_add.tick(out_ready ,g_cfg);
}


bool r4_valid = true;
for(int i = 0; i < 2; ++i) r4_valid &= add_level1[i].out_valid();
if (s5_ready){
if (r4_valid){
    add_level2.add_input.a_in = add_level1[0].out_data();
    add_level2.add_input.b_in = add_level1[1].out_data();
    add_level2.add_input.input_valid = true;
    }else{
    add_level2.add_input.input_valid = false;
    }
    add_level2.tick(s6_ready ,g_cfg ,add_level1[0].r2.c_in);
}


bool r3_valid = true;
for(int i = 0; i < 4; ++i) r3_valid &= add_level0[i].out_valid();
if (s4_ready){
if (r3_valid){
    for(int i = 0; i < 2; ++i) {
    add_level1[i].add_input.a_in = add_level0[i].out_data();
    add_level1[i].add_input.b_in = add_level0[i+2].out_data();
    add_level1[i].add_input.input_valid = true;
    add_level1[i].tick(s5_ready ,g_cfg ,add_level0[i].r2.c_in);
    }
}else{
    for(int i = 0; i < 2; ++i) {
        add_level1[i].add_input.input_valid = false;
        add_level1[i].tick(s5_ready ,g_cfg ,add_level0[i].r2.c_in);
    }
    }
}


bool r2_valid = true;
for(int i = 0; i < 8; ++i) r2_valid &= mul_array[i].out_valid();
if (s3_ready){
if (r2_valid){
    for(int i = 0; i < 4; ++i) {
    add_level0[i].add_input.a_in =  mul_array[i].out_data();
    add_level0[i].add_input.b_in = mul_array[i+4].out_data();
    add_level0[i].add_input.input_valid = true;
    add_level0[i].tick(s4_ready ,g_cfg ,mul_array[i].r3.c_in);
    }
}else{
    for(int i = 0; i < 4; ++i) {
        add_level0[i].add_input.input_valid = false;
        add_level0[i].tick(s4_ready ,g_cfg ,mul_array[i].r3.c_in);
    }
    }
}

if (s2_ready){
if (r1.valid){
    for(int i = 0; i < 8; ++i) {
    mul_array[i].mul_input.a_fp9 = r1.a_conv_val[i];
    mul_array[i].mul_input.b_fp9 = r1.b_conv_val[i];
    mul_array[i].mul_input.input_valid = true;
    mul_array[i].tick(s3_ready ,g_cfg ,r1.c_in);
    }
}else{
    for(int i = 0; i < 8; ++i) {
        mul_array[i].mul_input.input_valid = false;
        mul_array[i].tick(s3_ready ,g_cfg ,r1.c_in);
    }
    }
}

if (s1_ready){
if (mul_add_input.input_valid){
    for(int i = 0; i < 8; ++i) {
    r1.a_conv_val[i] = convert_to_fp9(mul_add_input.a_in[i] ,mul_add_input.prec);
    r1.b_conv_val[i] = convert_to_fp9(mul_add_input.b_in[i] ,mul_add_input.prec);
    }
    r1.c_in = mul_add_input.c_in;
    r1.valid = true;
}else{
    r1.valid = false;
}
}



}

};