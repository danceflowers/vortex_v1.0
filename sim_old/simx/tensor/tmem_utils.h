// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <cstdint>
#include <vector>

namespace vortex {
namespace tmem_functional {

// Shared helpers for mapping logical TMEM regions onto physical banks.
using PhysicalRow = std::vector<uint8_t>;
using BankStorage = std::vector<std::vector<PhysicalRow>>;

// Read an optional environment override for the physical bank count.
uint32_t configured_physical_bank_count(uint32_t default_banks,
                                        uint32_t min_banks,
                                        uint32_t max_banks = 256);

// Pick a stride that visits all banks before repeating.
uint32_t choose_coprime_stride(uint32_t count, uint32_t preferred);

uint32_t ceil_div(uint32_t value, uint32_t divisor);

void resize_storage(BankStorage* banks,
                    uint32_t physical_bank_count,
                    uint32_t physical_rows,
                    uint32_t bank_slice_bytes);

void reset_storage(BankStorage* banks);

bool region_query(uint32_t col_base,
                  uint32_t col_span,
                  uint32_t num_cols,
                  uint32_t col_bytes,
                  uint32_t* size_bytes);

// Compute which logical column contains a packet within a contiguous region.
bool region_packet_location(uint32_t col_base,
                            uint32_t col_span,
                            uint32_t packet_idx,
                            uint32_t num_cols,
                            uint32_t col_bytes,
                            uint32_t packet_bytes,
                            uint32_t* logical_col);

// Map one logical-line chunk to a physical bank after applying swizzle.
uint32_t line_chunk_bank(uint32_t logical_line,
                         uint32_t chunk_idx,
                         uint32_t physical_bank_count,
                         uint32_t bank_swizzle_base_stride,
                         uint32_t packet_lanes);

// Map one packet lane to the bank used for that lane's 8-byte slice.
uint32_t packet_lane_bank(uint32_t logical_col,
                          uint32_t packet_in_col,
                          uint32_t lane,
                          uint32_t physical_bank_count,
                          uint32_t physical_bank_bytes,
                          uint32_t logical_lines,
                          uint32_t bank_swizzle_base_stride,
                          uint32_t packet_lanes);

// Resolve a byte address in the logical TMEM grid to bank/row/byte indices.
void logical_byte_to_physical(uint32_t logical_col,
                              uint32_t logical_line,
                              uint32_t physical_bank_count,
                              uint32_t physical_bank_bytes,
                              uint32_t bank_swizzle_base_stride,
                              uint32_t packet_lanes,
                              uint32_t* bank,
                              uint32_t* row,
                              uint32_t* bank_byte);

} // namespace tmem_functional

namespace tmem_timing {

// Reset per-cycle packet and per-bank port budgets.
void reset_port_budgets(uint32_t read_packets_per_cycle,
                        uint32_t write_packets_per_cycle,
                        uint32_t read_ports_per_bank,
                        uint32_t write_ports_per_bank,
                        uint32_t* read_packet_budget,
                        uint32_t* write_packet_budget,
                        std::vector<uint32_t>* read_bank_budgets,
                        std::vector<uint32_t>* write_bank_budgets);

// Consume one packet's port budget, returning false when any touched bank is busy.
bool consume_packet_ports(const std::vector<bool>& touched_banks,
                          bool write,
                          uint32_t* read_packet_budget,
                          uint32_t* write_packet_budget,
                          std::vector<uint32_t>* read_bank_budgets,
                          std::vector<uint32_t>* write_bank_budgets);

// Refund a previously consumed packet budget after a request is cancelled.
void refund_packet_ports(const std::vector<bool>& touched_banks,
                         bool write,
                         uint32_t read_packets_per_cycle,
                         uint32_t write_packets_per_cycle,
                         uint32_t read_ports_per_bank,
                         uint32_t write_ports_per_bank,
                         uint32_t* read_packet_budget,
                         uint32_t* write_packet_budget,
                         std::vector<uint32_t>* read_bank_budgets,
                         std::vector<uint32_t>* write_bank_budgets);

} // namespace tmem_timing
} // namespace vortex
