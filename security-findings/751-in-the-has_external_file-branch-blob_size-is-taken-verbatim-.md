# Security finding #751: In the `!has_external_file` branch, `blob_size` is taken verbatim f…

**Summary:** In the `!has_external_file` branch, `blob_size` is taken verbatim f…

**CWE IDs:** CWE-789: Memory Allocation with Excessive Size Value
**Severity / Impact:** Denial of service via memory exhaustion: a crafted .bin cache with a large `size` (e.g. tens of GB, still below vector::max_size) forces a huge heap allocation before any read validates it, causing memory pressure / OOM. The subsequent read failure and any bad_alloc are caught by the try/catch in Deserialize (line 335), so this is a resource-exhaustion DoS rather than a crash, affecting any process loading an attacker-influenced EP context cache.
**Affected location:** `targets/onnxruntime/onnxruntime/core/providers/openvino/ov_bin_manager.cc:425` — `BinManager::DeserializeImpl()`
**Validated for repos:** onnxruntime
**Trust boundary:** EP context cache .bin file (BSON blob metadata) → ORT/OpenVINO EP deserialization

## Description / Root cause
In the `!has_external_file` branch, `blob_size` is taken verbatim from attacker-controlled BSON (line 412, uint64_t) and passed straight to `container.data.resize(blob_size)` (line 425) with no upper-bound check against the actual stream length. Unlike GetNativeBlob (lines 177-180), which enforces `blob_offset < bin_size && blob_size <= bin_size && blob_offset <= bin_size - blob_size` against the real byte size, DeserializeImpl performs no such validation before allocating. The preceding `stream.seekg(blob_offset)`+`stream.good()` check (422-423) does not detect an out-of-range offset for typical istream implementations (seeking past EOF leaves good()==true; only the later read fails), so the oversized allocation at line 425 is attempted regardless.

**Validator analysis:** The flaw is real: DeserializeImpl at ov_bin_manager.cc:425 calls container.data.resize(blob_size) where blob_size comes entirely from BSON metadata (line 412, uint64_t, no cap). The stream.seekg + stream.good() check at lines 422-423 does not validate blob_size against any stream extent — it only checks that the seek didn't fail. GetNativeBlob (lines 177-180) has the correct tripartite guard (blob_offset < bin_size && blob_size <= bin_size && blob_offset <= bin_size - blob_size) but DeserializeImpl has no equivalent. The try/catch at Deserialize() lines 333-337 catches std::exception (including std::bad_alloc), so this results in an ORT exception being thrown rather than a process crash, making the impact a DoS (resource exhaustion / process abort with a cache-corruption error) rather than RCE — the vuln_type CWE-789 and DoS impact characterization are accurate. The proposed fix is correct: seek to end of stream, capture the size, and apply the same tripartite bound check before the resize. The chromiumWebnn entry surface cannot craft the .bin file from web content, so the full chain is not web-reachable.

## Exploit / Proof of Concept
An attacker who can craft/replace the `<model>_OpenVINOExecutionProvider.bin` context-cache file sets a blob entry's `size` field (BSONFields::kSize) to a large value such as 0x0000_0080_0000_0000 (~512 GB) or any value large enough to exhaust memory but below vector::max_size. During DeserializeImpl the code reaches line 425 and calls container.data.resize(blob_size), attempting the allocation before the read at line 426 (which would fail) can reject it, exhausting host memory.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Targets: ov_bin_manager.cc:425 — container.data.resize(blob_size) with attacker-controlled blob_size
// from BSON metadata, no upper-bound check against actual stream size.
// Pre-fix: this test causes a bad_alloc or memory exhaustion (resize with huge blob_size).
// Post-fix: ORT_ENFORCE rejects oversized blob_size before resize.

// TODO: Determine actual gtest target for onnxruntime openvino EP unit tests.
// The component lives under onnxruntime/core/providers/openvino/; likely linked
// into onnxruntime_test_utils or a provider-specific test binary. Adjust target name.

#include <gtest/gtest.h>
#include <sstream>
#include <cstdint>
#include <nlohmann/json.hpp>

// TODO: Include BinManager header once the correct include path is confirmed.
// #include "core/providers/openvino/ov_bin_manager.h"

namespace onnxruntime {
namespace openvino_ep {
namespace test {

// Helper: build a minimal valid .bin byte stream with a crafted blob_size
// that exceeds the actual stream content length.
static std::string BuildCraftedBin(uint64_t crafted_blob_size) {
  // TODO: Replicate the header_t layout and BSON structure produced by
  // BinManager::Serialize. Specifically:
  //   1. Write header_t with correct magic, version, header_size, bson_start_offset, bson_size.
  //   2. Write BSON containing BSONFields::kBlobMetadata with one entry:
  //      { "test_blob": { "data_offset": <valid small offset>, "size": crafted_blob_size } }
  //   3. Write a few bytes of actual blob data (much smaller than crafted_blob_size).
  // Use nlohmann::json::to_bson() to produce the BSON bytes.
  // Return the full byte stream as a std::string for use with std::istringstream.
  // TODO: Fill in once header_t definition and BSONFields constants are confirmed from source.
  (void)crafted_blob_size;
  return {};
}

// Test: DeserializeImpl must reject a blob_size that exceeds the actual stream length.
// Pre-fix behavior: resize(crafted_blob_size) with e.g. 0x80000000 bytes on a tiny stream.
// Post-fix behavior: ORT_ENFORCE fires before resize, throwing an onnxruntime::OnnxRuntimeException.
TEST(BinManagerDeserialize, RejectOversizedBlobSizeInEmbeddedStream) {
  // Craft a tiny .bin stream but advertise a blob_size of 2 GB.
  constexpr uint64_t kCraftedBlobSize = 0x80000000ULL;  // 2 GB
  std::string raw = BuildCraftedBin(kCraftedBlobSize);
  // TODO: Remove the skip once BuildCraftedBin is implemented.
  if (raw.empty()) {
    GTEST_SKIP() << "BuildCraftedBin not yet implemented — skeleton only";
  }

  std::istringstream stream(raw, std::ios::binary);

  // TODO: Construct a real BinManager with default options.
  // BinManager manager(/* options */);
  // EXPECT_THROW(manager.Deserialize(stream, nullptr), onnxruntime::OnnxRuntimeException);
  // TODO: Verify the exception message contains "out of bounds" or similar
  // once the fix's ORT_ENFORCE message is known.
}

}  // namespace test
}  // namespace openvino_ep
}  // namespace onnxruntime
```
**Build / run:** TODO: Build target is likely 'onnxruntime_test_utils' or a provider-specific test binary for the openvino EP. Run with --gtest_filter='BinManagerDeserialize.RejectOversizedBlobSizeInEmbeddedStream'. Expected pre-fix failure: std::bad_alloc or OOM during container.data.resize(0x80000000) at ov_bin_manager.cc:425 (caught as ORT exception). Expected post-fix: ORT_ENFORCE fires at the new bounds check before resize, throwing cleanly.

## Suggested fix
Before allocating, determine the real stream extent and validate, mirroring GetNativeBlob's check. E.g. capture `stream.seekg(0, std::ios::end); auto end = stream.tellg();` then require `blob_offset <= end && blob_size <= (uint64_t)end && blob_offset <= (uint64_t)end - blob_size` (this also guards blob_offset+blob_size wraparound) via ORT_ENFORCE, and reject before `container.data.resize(blob_size)`. Optionally cap blob_size to a sane maximum.

## Models used

| Role | Provider / model |
| --- | --- |
| Idea | `claude-4-6-sonnet` |
| Tech lead | `claude-4-6-sonnet` |
| Researcher | `claude-4-8-opus` |
| Validator | `claude-4-6-sonnet` |


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #751.
