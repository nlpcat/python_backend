// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <unistd.h>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace triton { namespace backend { namespace python {

namespace bi = boost::interprocess;

struct SharedMemoryControl {
  bool allocated;
  size_t capacity;
  bool has_grown;
  char handle[64];
};

class SharedMemory {
  size_t* capacity_;
  off_t* offset_;
  char* shm_addr_;
  SharedMemoryControl* shm_control_block_;
  bi::managed_shared_memory::handle_t control_block_handle_;

  // Current capcity, local to each process.
  size_t current_capacity_;

  // Amount of bytes to grow the shared memory when the pool is completely used.
  int64_t shm_growth_bytes_;

  // List of old shared memory addresses that should be deallocated.
  // First element of the pair is size and second element is the address.
  std::vector<std::pair<size_t, char*>> old_shm_addresses_;
  void UpdateSharedMemory();

 public:
  SharedMemory(
      void* shm_addr, SharedMemoryControl* shm_control_block,
      bi::managed_shared_memory::handle_t control_block_handle);
  void MapOffset(char** shm_addr, size_t byte_size, off_t offset);
  void Map(char** shm_addr, size_t byte_size, off_t& offset);
  void SetOffset(off_t offset);
  bi::managed_shared_memory::handle_t Handle();
  ~SharedMemory();
};
class SharedMemoryManager {
  std::string shm_control_key_;
  std::string shm_data_key_;
  int64_t shm_growth_bytes_;
  int64_t shm_default_bytes_;
  bi::managed_shared_memory shm_data_;
  bi::managed_shared_memory shm_control_;

 public:
  std::string ShmControlKey();
  std::string ShmDataKey();

  std::unique_ptr<SharedMemory> GetRegionFromHandle(
      bi::managed_shared_memory::handle_t handle);
  std::unique_ptr<SharedMemory> Region(const int64_t byte_size);
  SharedMemoryManager(
      const std::string& shm_control_key, const std::string& shm_data_key,
      std::size_t default_byte_size, std::size_t growth_byte_size);
};

}}}  // namespace triton::backend::python
