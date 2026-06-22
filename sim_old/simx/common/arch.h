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

#include <string>
#include <sstream>

#include <cstdlib>
#include <stdio.h>
#include "types.h"

namespace vortex {

class Arch {
private:
  uint16_t num_threads_;
  uint16_t num_warps_;
  uint16_t warpgroup_size_;
  uint16_t num_cores_;
  uint16_t num_clusters_;
  uint16_t socket_size_;
  uint16_t num_barriers_;
  uint64_t local_mem_base_;

public:
  Arch(uint16_t num_threads, uint16_t num_warps, uint16_t num_cores)   
    : num_threads_(num_threads)
    , num_warps_(num_warps)
    , warpgroup_size_(4)
    , num_cores_(num_cores)
    , num_clusters_(NUM_CLUSTERS)
    , socket_size_(SOCKET_SIZE)
    , num_barriers_(NUM_BARRIERS)
    , local_mem_base_(LMEM_BASE_ADDR)
  {}

  uint16_t num_barriers() const {
    return num_barriers_;
  }

  uint64_t local_mem_base() const {
    return local_mem_base_;
  }

  uint16_t num_threads() const {
    return num_threads_;
  }

  uint16_t num_warps() const {
    return num_warps_;
  }

  uint16_t warpgroup_size() const {
    return warpgroup_size_;
  }

  uint16_t num_warpgroups() const {
    return (num_warps_ + warpgroup_size_ - 1) / warpgroup_size_;
  }

  uint16_t warpgroup_id(uint32_t wid) const {
    return wid / warpgroup_size_;
  }

  uint16_t warpgroup_lane(uint32_t wid) const {
    return wid % warpgroup_size_;
  }

  uint16_t warpgroup_first_wid(uint32_t wgid) const {
    return wgid * warpgroup_size_;
  }

  uint16_t warpgroup_leader(uint32_t wgid) const {
    return warpgroup_first_wid(wgid);
  }

  uint16_t num_cores() const {
    return num_cores_;
  }

  uint16_t num_clusters() const {
    return num_clusters_;
  }

  uint16_t socket_size() const {
    return socket_size_;
  }

};

}
