// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <VX_config.h>
#include <VX_types.h>
#include <vx_intrinsics.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef XLEN_64
    #define DUMP_CSRS(i) \
        ((int64_t*)csr_mem)[i] = csr_read(VX_CSR_MPM_BASE +i)
#else
    #define DUMP_CSRS(i) \
        csr_mem[(i*2)+0] = csr_read(VX_CSR_MPM_BASE + i); \
        csr_mem[(i*2)+1] = csr_read(VX_CSR_MPM_BASE + i + (VX_CSR_MPM_BASE_H - VX_CSR_MPM_BASE))
#endif

void vx_perf_dump() {
    int core_id = vx_core_id();
    uint32_t * const csr_mem = (uint32_t*)(IO_MPM_ADDR + 64 * sizeof(uint64_t) * core_id);
    DUMP_CSRS(0);
    //DUMP_CSRS(1); reserved for exitcode
    DUMP_CSRS(2);
    DUMP_CSRS(3);
    DUMP_CSRS(4);
    DUMP_CSRS(5);
    DUMP_CSRS(6);
    DUMP_CSRS(7);
    DUMP_CSRS(8);
    DUMP_CSRS(9);
    DUMP_CSRS(10);
    DUMP_CSRS(11);
    DUMP_CSRS(12);
    DUMP_CSRS(13);
    DUMP_CSRS(14);
    DUMP_CSRS(15);
    DUMP_CSRS(16);
    DUMP_CSRS(17);
    DUMP_CSRS(18);
    DUMP_CSRS(19);
    DUMP_CSRS(20);
    DUMP_CSRS(21);
    DUMP_CSRS(22);
    DUMP_CSRS(23);
    DUMP_CSRS(24);
    DUMP_CSRS(25);
    DUMP_CSRS(26);
    DUMP_CSRS(27);
    DUMP_CSRS(28);
    DUMP_CSRS(29);
    DUMP_CSRS(30);
    DUMP_CSRS(31);
    DUMP_CSRS(32);
    DUMP_CSRS(33);
    DUMP_CSRS(34);
    DUMP_CSRS(35);
    DUMP_CSRS(36);
    DUMP_CSRS(37);
    DUMP_CSRS(38);
    DUMP_CSRS(39);
    DUMP_CSRS(40);
    DUMP_CSRS(41);
    DUMP_CSRS(42);
    DUMP_CSRS(43);
    DUMP_CSRS(44);
    DUMP_CSRS(45);
    DUMP_CSRS(46);
    DUMP_CSRS(47);
    DUMP_CSRS(48);
    DUMP_CSRS(49);
    DUMP_CSRS(50);
    DUMP_CSRS(51);
    DUMP_CSRS(52);
    DUMP_CSRS(53);
    DUMP_CSRS(54);
    DUMP_CSRS(55);
    DUMP_CSRS(56);
    DUMP_CSRS(57);
    DUMP_CSRS(58);
    DUMP_CSRS(59);
    DUMP_CSRS(60);
    DUMP_CSRS(61);
    DUMP_CSRS(62);
    DUMP_CSRS(63);
}

#ifdef __cplusplus
}
#endif
