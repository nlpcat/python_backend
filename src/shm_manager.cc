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

#include "shm_manager.h"
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <sstream>
#include <string>
#include "pb_utils.h"

namespace triton { namespace backend { namespace python {

namespace bi = boost::interprocess;

SharedMemory::SharedMemory(
    void* shm_addr, SharedMemoryControl* shm_control_block,
    bi::managed_shared_memory::handle_t control_block_handle)
{
  control_block_handle_ = control_block_handle;
  shm_addr_ = static_cast<char*>(shm_addr);
  shm_control_block_ = shm_control_block;

  capacity_ = (size_t*)shm_addr_;
  *capacity_ = shm_control_block_->capacity;
  current_capacity_ = *capacity_;

  // Set offset address
  offset_ = (off_t*)((char*)shm_addr_ + sizeof(size_t));

  *offset_ = 0;
  *offset_ += sizeof(off_t);
  *offset_ += sizeof(size_t);
}

SharedMemory::~SharedMemory() {}

void
SharedMemory::Map(char** shm_addr, size_t byte_size, off_t& offset)
{
  // What to do for growing the shared memory?
  // while (*offset_ + byte_size >= *capacity_) {
  // // Increase the shared memory pool size by one page size.
  // *capacity_ = *offset_ + byte_size + shm_growth_bytes_;
  // if (ftruncate(shm_fd_, *capacity_) == -1) {
  // std::unique_ptr<PythonBackendError> err =
  // std::make_unique<PythonBackendError>();
  // err->error_message =
  // ("Failed to increase the shared memory pool size for key '" +
  // shm_key_ + "' to " + std::to_string(*capacity_) + " bytes");
  // throw PythonBackendException(std::move(err));
  // }
  // }

  // UpdateSharedMemory();

  *shm_addr = shm_addr_ + *offset_;
  offset = *offset_;

  *offset_ += byte_size;
}

void
SharedMemory::UpdateSharedMemory()
{
  // if (current_capacity_ != *capacity_) {
  //   void* map_addr =
  //       mmap(NULL, *capacity_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_,
  //       0);

  //   if (map_addr == MAP_FAILED) {
  //     std::unique_ptr<PythonBackendError> err =
  //         std::make_unique<PythonBackendError>();
  //     err->error_message =
  //         ("unable to process address space or shared-memory descriptor: " +
  //          std::to_string(shm_fd_));
  //     throw PythonBackendException(std::move(err));
  //   }

  //   old_shm_addresses_.push_back({current_capacity_, shm_addr_});
  //   current_capacity_ = *capacity_;
  //   shm_addr_ = (char*)map_addr;
  // }
}

void
SharedMemory::MapOffset(char** shm_addr, size_t byte_size, off_t offset)
{
  // Update shared memory pointer and capacity if necessary.
  // UpdateSharedMemory();
  *shm_addr = shm_addr_ + offset;
}

void
SharedMemory::SetOffset(off_t offset)
{
  *offset_ = offset;
}

bi::managed_shared_memory::handle_t
SharedMemory::Handle()
{
  return control_block_handle_;
}

SharedMemoryManager::SharedMemoryManager(
    const std::string& shm_control_key, const std::string& shm_data_key,
    std::size_t default_byte_size, std::size_t growth_byte_size)
{
  shm_control_key_ = shm_control_key;
  shm_data_key_ = shm_data_key;
  shm_growth_bytes_ = growth_byte_size;
  shm_default_bytes_ = default_byte_size;
  shm_data_ = bi::managed_shared_memory(
      bi::open_or_create, shm_data_key_.c_str(), default_byte_size);
  shm_control_ = bi::managed_shared_memory(
      bi::open_or_create, shm_control_key_.c_str(), default_byte_size);
}

std::unique_ptr<SharedMemory>
SharedMemoryManager::Region(const int64_t byte_size)
{
  void* ptr = shm_data_.allocate(byte_size);
  void* shm_control = shm_control_.allocate(sizeof(SharedMemoryControl));
  SharedMemoryControl* shared_memory_control =
      static_cast<SharedMemoryControl*>(shm_control);
  shared_memory_control->capacity = byte_size;
  bi::managed_shared_memory::handle_t handle =
      shm_data_.get_handle_from_address(ptr);
  std::stringstream s;
  s << handle;
  strcpy(shared_memory_control->handle, s.str().c_str());
  return std::make_unique<SharedMemory>(
      ptr, shared_memory_control,
      shm_control_.get_handle_from_address(shm_control));
}

std::unique_ptr<SharedMemory>
SharedMemoryManager::GetRegionFromHandle(
    bi::managed_shared_memory::handle_t handle)
{
  SharedMemoryControl* shared_memory_control =
      static_cast<SharedMemoryControl*>(
          shm_control_.get_address_from_handle(handle));
  bi::managed_shared_memory::handle_t handle_data = 0;
  std::stringstream s;
  s << shared_memory_control->handle;
  s >> handle_data;
  void* ptr = shm_data_.get_address_from_handle(handle_data);
  return std::make_unique<SharedMemory>(ptr, shared_memory_control, handle);
}

std::string
SharedMemoryManager::ShmControlKey()
{
  return shm_control_key_;
}

std::string
SharedMemoryManager::ShmDataKey()
{
  return shm_data_key_;
}

}}}  // namespace triton::backend::python
