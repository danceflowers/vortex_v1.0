#pragma once

#include <vector>

#include "open_tensorcore/local_memory/amem.h"
#include "open_tensorcore/local_memory/bmem.h"
#include "open_tensorcore/local_memory/cmem.h"
#include "open_tensorcore/local_memory/meta_mem.h"

namespace tensor_mem_test_utils {

// Reference helper: fill all AMem lines from a complete packet sequence.
inline bool bulk_fill_tile_for_reference(AMem* amem,
                                         uint32_t fmt_a,
                                         const std::vector<AMem::packet_t>& packets) {
  if (nullptr == amem) {
    return false;
  }
  auto expected_packets = AMem::packet_count(fmt_a);
  if (packets.size() < expected_packets) {
    return false;
  }

  amem->clear();
  auto packets_per_line = AMem::packets_per_fill_line(fmt_a);
  for (uint32_t line = 0; line < AMem::kDepth; ++line) {
    std::vector<AMem::packet_t> line_packets;
    for (uint32_t packet = 0; packet < packets_per_line; ++packet) {
      line_packets.push_back(packets.at(line * packets_per_line + packet));
    }
    if (!amem->write_fill_line(fmt_a, line, line_packets)) {
      return false;
    }
  }
  return true;
}

// Reference helper: fill all BMem lines from a complete packet sequence.
inline bool bulk_fill_tile_for_reference(BMem* bmem,
                                         uint32_t fmt_b,
                                         const std::vector<BMem::packet_t>& packets,
                                         uint32_t sparse_mode = vortex::tensor::sparse_none) {
  if (nullptr == bmem) {
    return false;
  }
  auto expected_packets = BMem::packet_count(fmt_b, sparse_mode);
  if (packets.size() < expected_packets) {
    return false;
  }

  bmem->clear();
  auto packets_per_line = BMem::packets_per_fill_line(fmt_b, sparse_mode);
  for (uint32_t line = 0; line < BMem::kDepth; ++line) {
    std::vector<BMem::packet_t> line_packets;
    for (uint32_t packet = 0; packet < packets_per_line; ++packet) {
      line_packets.push_back(packets.at(line * packets_per_line + packet));
    }
    if (!bmem->write_fill_line(fmt_b, line, line_packets, sparse_mode)) {
      return false;
    }
  }
  return true;
}

// Reference helper: fill all CMem subtiles from a complete packet sequence.
inline bool bulk_fill_tile_for_reference(CMem* cmem,
                                         uint32_t fmt_c,
                                         const std::vector<CMem::packet_t>& packets) {
  if (nullptr == cmem) {
    return false;
  }
  auto expected_packets = CMem::packet_count(fmt_c);
  if (packets.size() < expected_packets) {
    return false;
  }

  cmem->clear();
  auto packets_per_subtile_group = CMem::packets_per_subtile(fmt_c);
  for (uint32_t subtile = 0; subtile < CMem::kDepth; ++subtile) {
    std::vector<CMem::packet_t> subtile_packets;
    auto group_base = subtile * packets_per_subtile_group;
    for (uint32_t packet = 0; packet < packets_per_subtile_group; ++packet) {
      subtile_packets.push_back(packets.at(group_base + packet));
    }
    if (!cmem->write_fill_subtile(fmt_c, subtile, subtile_packets)) {
      return false;
    }
  }
  return true;
}

// Reference helper: dump all CMem subtiles into a complete packet sequence.
inline bool bulk_dump_tile_for_reference(const CMem* cmem,
                                         uint32_t fmt_c,
                                         std::vector<CMem::packet_t>* packets) {
  if (nullptr == cmem || nullptr == packets || !cmem->valid()) {
    return false;
  }

  packets->assign(CMem::packet_count(fmt_c), CMem::packet_t{});
  auto packets_per_group = CMem::packets_per_subtile(fmt_c);
  for (uint32_t subtile = 0; subtile < CMem::kDepth; ++subtile) {
    auto base = subtile * packets_per_group;
    std::vector<CMem::packet_t> subtile_packets;
    if (!cmem->dump_subtile_packets(fmt_c, subtile, &subtile_packets)) {
      return false;
    }
    for (uint32_t packet_idx = 0; packet_idx < packets_per_group; ++packet_idx) {
      packets->at(base + packet_idx) = subtile_packets.at(packet_idx);
    }
  }
  return true;
}

// Reference helper: fill MetaMem from its single metadata packet.
inline bool bulk_fill_tile_for_reference(MetaMem* metamem,
                                         const std::vector<MetaMem::packet_t>& packets) {
  if (nullptr == metamem || packets.size() < MetaMem::packet_count()) {
    return false;
  }
  return metamem->write_fill_packet(packets.front());
}

} // namespace tensor_mem_test_utils
