/**
 * Startup read trace: record the guest's reads of the read-only game files, and replay them ahead
 * of the guest on a background thread. See startup_trace.cpp for the design and the file format.
 *
 * Everything here is inert unless one of the cvars sa2_io_trace_record / sa2_io_preload is on.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

#include <rex/filesystem.h>

namespace rex::filesystem {

class HostPathDevice;

namespace startup_trace {

// Called once a read-only HostPathDevice finished mounting. The first one becomes the "game root".
void OnMount(HostPathDevice* device);

// Called when the guest opens a read-only file on `device` without write access.
// Returns a tag (> 0) if the file is tracked, 0 if tracing is inactive for it.
uint32_t OnOpen(HostPathDevice* device, const std::string& path);

// Called for every guest read of a tracked file, before any cache. If the preload store has the
// exact range, copies it into `buffer`, sets *out_bytes_read and returns true.
bool OnRead(uint32_t tag, uint64_t offset, std::span<uint8_t> buffer, size_t* out_bytes_read);

// A pre-opened read handle for the file at `host_path`, or nullptr.
std::unique_ptr<FileHandle> TakePreopened(HostPathDevice* device, const std::string& path);

// Test aid (sa2_io_preload_verify): true if every preload hit should be compared with the SD.
bool VerifyEnabled();
// Result of one such comparison. Mismatches are logged at once and counted in the final report.
void ReportVerify(uint32_t tag, uint64_t offset, size_t length, bool equal);

}  // namespace startup_trace
}  // namespace rex::filesystem
