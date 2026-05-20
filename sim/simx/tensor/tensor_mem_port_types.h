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

#include <cstdint>
#include "tmem.h"

namespace vortex {

// Packet-level messages between TMEM clients and TmemSystem.
//
// TensorUnit and the Core-side transfer engines issue one request per 64B TMEM
// packet. TmemSystem arbitrates all clients through the shared TMEM bank model
// and returns one matching response per request.

// One packet-level request crossing the TensorUnit/Core -> TmemSystem boundary.
// arbitration_age orders all clients inside the shared TMEM request FIFO.
struct TensorMemPortReq {
  enum class AccessType : uint8_t {
    Read = 0,
    Write,
  };

  uint64_t request_id = 0;          // Globally unique request/response key.
  uint64_t arbitration_age = 0;     // Lower values win shared-port arbitration.
  AccessType access_type = AccessType::Read;  // Read or write transaction.
  Tmem::PortRequestDesc port_request = {};    // TMEM packet address descriptor.
  TmemPacket write_packet = {};               // Write payload; ignored for reads.
};

// Response for one granted TMEM packet request after the SRAM access completes.
struct TensorMemPortRsp {
  uint64_t request_id = 0;          // Matching request ID.
  TensorMemPortReq::AccessType access_type = TensorMemPortReq::AccessType::Read;
  TmemPacket read_packet = {};      // Read payload; zero for writes.
};

// Completion notice for an architecture-visible async tensor macro operation.
// Core listens to this stream to wake warps blocked on waits or fences.
struct TensorAsyncOpCompletion {
  uint32_t async_id = 0;            // Completed async operation ID.
  uint64_t tx_bound_mbar = 0;       // Optional TMA complete_tx target.
  uint32_t tx_bytes = 0;            // Optional completed TMA byte count.
  uint32_t payload_size_bytes = 0;  // Optional completed payload size.
};

} // namespace vortex
