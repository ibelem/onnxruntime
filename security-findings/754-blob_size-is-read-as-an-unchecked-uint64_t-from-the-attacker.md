# Security finding #754: `blob_size` is read as an unchecked `uint64_t` from the attacker-su…

**Summary:** `blob_size` is read as an unchecked `uint64_t` from the attacker-su…

**CWE IDs:** CWE-789: Memory Allocation with Excessive Size Value
**Severity / Impact:** A crafted embedded EPContext blob whose BSON declares an enormous per-blob `size` forces an oversized `std::vector` allocation, causing memory-exhaustion DoS of the host process. Reachable from web content via the same embed_mode EPContext load path.
**Affected location:** `targets/onnxruntime/onnxruntime/core/providers/openvino/ov_bin_manager.cc:425` — `BinManager::DeserializeImpl()`
**Validated for repos:** onnxruntime
**Trust boundary:** web content → WebNN → ORT → OpenVINO EP embedded EPContext blob → BSON blob_metadata `size` field

## Description / Root cause
`blob_size` is read as an unchecked `uint64_t` from the attacker-supplied BSON metadata at line 412 and used directly at line 425 `container.data.resize(blob_size)` with no upper bound and no validation that `blob_offset + blob_size` lies within the stream. `blob_offset` (line 411) is likewise used unchecked for `stream.seekg` at line 422.

**Validator analysis:** The vulnerability type (CWE-789: Memory Allocation with Excessive Size Value) is accurately categorized. The flaw is confirmed at ov_bin_manager.cc:425: blob_size is read from attacker-controlled BSON at line 412 as a uint64_t with no cap, then used directly for container.data.resize(blob_size) at line 425. The seek-bounds check via TensorStreamBuf::seekoff (lines 49-53) only gates seekg success, not the resize — a huge allocation is attempted before any read. The post-read ORT_ENFORCE(stream.good()) at line 427 does not prevent the allocation. blob_offset is similarly unchecked before the seekg at line 422. The proposed fix (compute stream length once; enforce blob_offset <= len && blob_size <= len - blob_offset before resize) is correct and sufficient. A per-blob size cap (e.g., 512 MB) would add defense-in-depth. The impact claim (memory exhaustion DoS of the host process) is accurate.

## Exploit / Proof of Concept
In the embedded blob, craft valid header + BSON such that `blob_metadata` contains an entry with `size` = a very large value (and `has_external_file` false, i.e. embedded stream). DeserializeImpl reaches line 425 and calls `resize(blob_size)`, attempting a huge allocation before any read succeeds.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.

// Regression test for CWE-789 in BinManager::DeserializeImpl (ov_bin_manager.cc:425)
// A crafted embedded EPContext binary blob declares blob_size = UINT64_MAX in its BSON
// metadata. Pre-fix: container.data.resize(UINT64_MAX) triggers OOM / std::bad_alloc.
// Post-fix: ORT_ENFORCE fires before resize because blob_size > stream_length.

#include <gtest/gtest.h>
#include <sstream>
#include <vector>
#include <cstdint>
#include <nlohmann/json.hpp>

// Include the BinManager header under test.
// TODO: Adjust the include path to match the actual build tree.
#include "onnxruntime/core/providers/openvino/ov_bin_manager.h"
#include "onnxruntime/core/providers/openvino/ov_shared_context.h"

namespace onnxruntime {
namespace openvino_ep {
namespace {

// Helper: build a minimal valid binary stream containing a header, BSON
// metadata whose blob_metadata_map entry has size = giant_size,
// and no actual blob data (embedded mode, no external_bin_path).
std::string BuildCraftedBin(uint64_t giant_blob_size) {
  // Magic / version / header constants copied from ov_bin_manager.cc.
  constexpr uint64_t kMagicNumber = 0x4E49425F5045564FULL;
  constexpr uint64_t kBinVersion = 1;  // BinVersion::v1

  // Build BSON metadata.
  nlohmann::json j;
  j["version"] = "1.0.0";
  j["producer"] = "test";
  // blob_metadata_map: one entry with enormous size.
  j["blob_metadata_map"]["crafted_blob"]["data_offset"] = uint64_t{0};
  j["blob_metadata_map"]["crafted_blob"]["size"] = giant_blob_size;

  std::vector<uint8_t> bson_bytes = nlohmann::json::to_bson(j);

  // header_t layout: magic(8) version(8) header_size(8) bson_start_offset(8) bson_size(8) = 40 bytes.
  constexpr uint64_t kHeaderSize = 40;
  uint64_t bson_start = kHeaderSize;
  uint64_t bson_size = bson_bytes.size();

  std::string buf;
  buf.resize(kHeaderSize + bson_size);
  auto write64 = [&](size_t offset, uint64_t v) {
    std::memcpy(&buf[offset], &v, 8);
  };
  write64(0,  kMagicNumber);
  write64(8,  kBinVersion);
  write64(16, kHeaderSize);
  write64(24, bson_start);
  write64(32, bson_size);
  std::memcpy(&buf[kHeaderSize], bson_bytes.data(), bson_bytes.size());
  return buf;
}

// TODO: Confirm ORT_ENFORCE throws onnxruntime::OnnxRuntimeException (or a
// std::exception subclass) — replace ASSERT_ANY_THROW with the precise type
// once confirmed against the build.
TEST(BinManagerDeserializeImplTest, OversizedBlobSizeIsRejected) {
  // Arrange: craft a stream whose BSON claims blob_size = UINT64_MAX.
  std::string raw = BuildCraftedBin(std::numeric_limits<uint64_t>::max());
  std::istringstream stream(raw, std::ios::binary);

  BinManager mgr;  // No external bin path => embedded mode.
  // TODO: SharedContext requires a bin_path in its constructor;
  //       pass an empty path or a valid SharedContext factory if needed.
  std::shared_ptr<SharedContext> ctx{};

  // Act + Assert: pre-fix this throws std::bad_alloc or crashes;
  //               post-fix it throws ORT_ENFORCE before reaching resize().
  ASSERT_ANY_THROW(mgr.Deserialize(stream, ctx));
}

}  // namespace
}  // namespace openvino_ep
}  // namespace onnxruntime
```
**Build / run:** Build target: ov_openvino_ep_unit_tests (or the nearest openvino provider gtest target — check CMakeLists.txt under onnxruntime/core/providers/openvino). Run with: --gtest_filter=BinManagerDeserializeImplTest.OversizedBlobSizeIsRejected. Expected pre-fix failure: std::bad_alloc or SIGKILL (OOM) during container.data.resize(UINT64_MAX) at ov_bin_manager.cc:425. Expected post-fix behaviour: ORT_ENFORCE fires immediately with a bounds-check error before any allocation.

## Suggested fix
Validate each `blob_size`/`blob_offset` against the total stream length and a reasonable cap before resizing: compute stream length once and `ORT_ENFORCE(blob_offset <= len && blob_size <= len - blob_offset, ...)`. Prefer reading directly into a bounded buffer sized to the verified remaining bytes rather than to the attacker-declared size.

## Models used

| Role | Provider / model |
| --- | --- |
| Idea | `claude-4-6-sonnet` |
| Tech lead | `claude-4-6-sonnet` |
| Researcher | `claude-4-8-opus` |
| Validator | `claude-4-6-sonnet` |


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #754.
