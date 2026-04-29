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

#pragma once

#include <vector>
#include "tma.h"
#include "tmem.h"

namespace vortex {

TmemWindowTarget map_window_target(const TmaDescriptor& descriptor);
uint32_t infer_window_fmt(const TmaDescriptor& descriptor);
uint32_t fmt_elem_bytes(uint32_t fmt);
uint32_t meta_shadow_window_id(uint32_t window_id);
bool build_sparse_meta_window_plan(const TmaDescriptor& descriptor,
                                   uint32_t window_id,
                                   TmemWindowPlan* window);
TmemWindowPlan build_legacy_window_plan(uint32_t window_id,
                                        const TmaDescriptor& descriptor,
                                        uint32_t col_base,
                                        uint32_t col_span);
void adjust_legacy_subwindow_from_existing(const TmaDescriptor& descriptor,
                                           const TmemWindowPlan* existing_window,
                                           TmemWindowPlan* window);
bool preserve_existing_math_window(const TmemWindowPlan* existing_window,
                                   const TmaDescriptor& descriptor);
bool place_window_after_existing(const TmemAllocation& allocation,
                                 TmemWindowPlan* window);
void encode_math_window_packet(const TmemWindowPlan& window,
                               const std::vector<uint8_t>& math_bytes,
                               uint32_t packet_idx,
                               TmemPacket* packet);
void decode_math_window_packet(const TmemWindowPlan& window,
                               uint32_t packet_idx,
                               const TmemPacket& packet,
                               std::vector<uint8_t>* math_bytes);

} // namespace vortex
