// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include "tmem_utils.h"

#include <algorithm>
#include <cstdlib>
#include <numeric>

namespace vortex {
namespace tmem_functional {

// Parse VORTEX_SIMX_TMEM_BANKS while preserving a safe default on bad input.
uint32_t configured_physical_bank_count(uint32_t default_banks,
                                        uint32_t min_banks,
                                        uint32_t max_banks) {
  auto value = std::getenv("VORTEX_SIMX_TMEM_BANKS");
  if (nullptr == value || '\0' == value[0]) {
    return default_banks;
  }

  char* end = nullptr;
  auto parsed = std::strtoul(value, &end, 0);
  if (end == value
   || *end != '\0'
   || parsed < min_banks
   || parsed > max_banks) {
    return default_banks;
  }
  return static_cast<uint32_t>(parsed);
}

// Choose a bank-swizzle stride that is relatively prime to the bank count.
uint32_t choose_coprime_stride(uint32_t count, uint32_t preferred) {
  if (count <= 1) {
    return 1;
  }

  preferred = std::min<uint32_t>(std::max<uint32_t>(1, preferred), count - 1);

  for (uint32_t candidate = preferred; candidate >= 1; --candidate) {
    if (std::gcd(candidate, count) == 1) {
      return candidate;
    }
    if (candidate == 1) {
      break;
    }
  }

  for (uint32_t candidate = preferred + 1; candidate < count; ++candidate) {
    if (std::gcd(candidate, count) == 1) {
      return candidate;
    }
  }

  return 1;
}

uint32_t ceil_div(uint32_t value, uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

void resize_storage(BankStorage* banks,
                    uint32_t physical_bank_count,
                    uint32_t physical_rows,
                    uint32_t bank_slice_bytes) {
  if (nullptr == banks) {
    return;
  }
  banks->assign(physical_bank_count, std::vector<PhysicalRow>(physical_rows));
  for (auto& bank : *banks) {
    for (auto& row : bank) {
      row.assign(bank_slice_bytes, 0);
    }
  }
}

void reset_storage(BankStorage* banks) {
  if (nullptr == banks) {
    return;
  }
  for (auto& bank : *banks) {
    for (auto& row : bank) {
      std::fill(row.begin(), row.end(), 0);
    }
  }
}

bool region_query(uint32_t col_base,
                  uint32_t col_span,
                  uint32_t num_cols,
                  uint32_t col_bytes,
                  uint32_t* size_bytes) {
  if (col_span == 0 || col_base >= num_cols || (col_base + col_span) > num_cols) {
    return false;
  }
  if (size_bytes) {
    *size_bytes = col_span * col_bytes;
  }
  return true;
}

bool region_packet_location(uint32_t col_base,
                            uint32_t col_span,
                            uint32_t packet_idx,
                            uint32_t num_cols,
                            uint32_t col_bytes,
                            uint32_t packet_bytes,
                            uint32_t* logical_col) {
  uint32_t size_bytes = 0;
  if (!region_query(col_base, col_span, num_cols, col_bytes, &size_bytes)) {
    return false;
  }
  uint32_t offset = packet_idx * packet_bytes;
  if (offset + packet_bytes > size_bytes) {
    return false;
  }
  if (logical_col) {
    *logical_col = col_base + (offset / col_bytes);
  }
  return true;
}

uint32_t line_chunk_bank(uint32_t logical_line,
                         uint32_t chunk_idx,
                         uint32_t physical_bank_count,
                         uint32_t bank_swizzle_base_stride,
                         uint32_t packet_lanes) {
  auto physical_row = logical_line / 2;
  auto line_parity = logical_line % 2;
  auto base = (physical_row * bank_swizzle_base_stride) % physical_bank_count;
  return (base + line_parity * packet_lanes + chunk_idx) % physical_bank_count;
}

uint32_t packet_lane_bank(uint32_t logical_col,
                          uint32_t packet_in_col,
                          uint32_t lane,
                          uint32_t physical_bank_count,
                          uint32_t physical_bank_bytes,
                          uint32_t logical_lines,
                          uint32_t bank_swizzle_base_stride,
                          uint32_t packet_lanes) {
  auto logical_line = packet_in_col * packet_lanes + lane;
  return line_chunk_bank(logical_line % logical_lines,
                         logical_col / physical_bank_bytes,
                         physical_bank_count,
                         bank_swizzle_base_stride,
                         packet_lanes);
}

void logical_byte_to_physical(uint32_t logical_col,
                              uint32_t logical_line,
                              uint32_t physical_bank_count,
                              uint32_t physical_bank_bytes,
                              uint32_t bank_swizzle_base_stride,
                              uint32_t packet_lanes,
                              uint32_t* bank,
                              uint32_t* row,
                              uint32_t* bank_byte) {
  auto chunk_idx = logical_col / physical_bank_bytes;
  if (bank) {
    *bank = line_chunk_bank(logical_line,
                            chunk_idx,
                            physical_bank_count,
                            bank_swizzle_base_stride,
                            packet_lanes);
  }
  if (row) {
    *row = logical_line / 2;
  }
  if (bank_byte) {
    *bank_byte = logical_col % physical_bank_bytes;
  }
}

} // namespace tmem_functional

namespace tmem_timing {

void reset_port_budgets(uint32_t read_packets_per_cycle,
                        uint32_t write_packets_per_cycle,
                        uint32_t read_ports_per_bank,
                        uint32_t write_ports_per_bank,
                        uint32_t* read_packet_budget,
                        uint32_t* write_packet_budget,
                        std::vector<uint32_t>* read_bank_budgets,
                        std::vector<uint32_t>* write_bank_budgets) {
  if (read_packet_budget) {
    *read_packet_budget = read_packets_per_cycle;
  }
  if (write_packet_budget) {
    *write_packet_budget = write_packets_per_cycle;
  }
  if (read_bank_budgets) {
    std::fill(read_bank_budgets->begin(), read_bank_budgets->end(), read_ports_per_bank);
  }
  if (write_bank_budgets) {
    std::fill(write_bank_budgets->begin(), write_bank_budgets->end(), write_ports_per_bank);
  }
}

bool consume_packet_ports(const std::vector<bool>& touched_banks,
                          bool write,
                          uint32_t* read_packet_budget,
                          uint32_t* write_packet_budget,
                          std::vector<uint32_t>* read_bank_budgets,
                          std::vector<uint32_t>* write_bank_budgets) {
  auto* packet_budget = write ? write_packet_budget : read_packet_budget;
  auto* bank_budgets = write ? write_bank_budgets : read_bank_budgets;
  if (nullptr == packet_budget || nullptr == bank_budgets || 0 == *packet_budget) {
    return false;
  }
  for (uint32_t bank = 0; bank < touched_banks.size(); ++bank) {
    if (touched_banks.at(bank) && 0 == bank_budgets->at(bank)) {
      return false;
    }
  }
  --(*packet_budget);
  for (uint32_t bank = 0; bank < touched_banks.size(); ++bank) {
    if (touched_banks.at(bank)) {
      --bank_budgets->at(bank);
    }
  }
  return true;
}

void refund_packet_ports(const std::vector<bool>& touched_banks,
                         bool write,
                         uint32_t read_packets_per_cycle,
                         uint32_t write_packets_per_cycle,
                         uint32_t read_ports_per_bank,
                         uint32_t write_ports_per_bank,
                         uint32_t* read_packet_budget,
                         uint32_t* write_packet_budget,
                         std::vector<uint32_t>* read_bank_budgets,
                         std::vector<uint32_t>* write_bank_budgets) {
  auto* packet_budget = write ? write_packet_budget : read_packet_budget;
  auto* bank_budgets = write ? write_bank_budgets : read_bank_budgets;
  if (nullptr == packet_budget || nullptr == bank_budgets) {
    return;
  }
  auto max_packet_budget = write ? write_packets_per_cycle : read_packets_per_cycle;
  auto max_bank_budget = write ? write_ports_per_bank : read_ports_per_bank;
  *packet_budget = std::min<uint32_t>(*packet_budget + 1, max_packet_budget);
  for (uint32_t bank = 0; bank < touched_banks.size(); ++bank) {
    if (touched_banks.at(bank)) {
      bank_budgets->at(bank) = std::min<uint32_t>(bank_budgets->at(bank) + 1, max_bank_budget);
    }
  }
}

} // namespace tmem_timing
} // namespace vortex
