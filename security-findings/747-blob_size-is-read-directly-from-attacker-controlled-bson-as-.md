# Security finding #747: `blob_size` is read directly from attacker-controlled BSON as a uin…

**Summary:** `blob_size` is read directly from attacker-controlled BSON as a uin…

**CWE IDs:** CWE-789: Memory Allocation with Excessive Size Value
**Severity / Impact:** Memory-exhaustion denial of service. A single `size` of e.g. 0xFFFFFFFF (or several multi-GiB entries) forces the process to attempt to commit gigabytes before any read occurs. On overcommit systems the memory is actually committed (OOM-kill / thrash); elsewhere it throws bad_alloc. Affects any consumer loading an untrusted ep-context cache via the OV EP.
**Affected location:** `targets/onnxruntime/onnxruntime/core/providers/openvino/ov_bin_manager.cc:425` — `BinManager::DeserializeImpl()`
**Validated for repos:** onnxruntime
**Trust boundary:** BSON `size` field inside an attacker-written .bin ep-context-cache file → ORT OpenVINO EP deserialization

## Description / Root cause
`blob_size` is read directly from attacker-controlled BSON as a uint64_t (line 412) and passed straight to `container.data.resize(blob_size)` (line 425) with no upper bound and no comparison against the actual stream length. The only guards are the header/version checks and a post-read `stream.good()` — neither caps the requested allocation size. Each blob entry in `blob_metadata_map` triggers its own `resize`, so many large entries compound the effect.

**Validator analysis:** The vuln type (CWE-789) is accurate: blob_size is read from BSON at line 412 as a uint64_t with no upper bound enforced, then used directly as the argument to std::vector::resize() at line 425. No check against the actual stream length occurs before the allocation. Multiple large entries in blob_metadata_map each trigger their own unchecked resize, compounding the effect. There is also a similar unchecked allocation for the BSON data itself at line 357 (header.bson_size from the file header), but that is not the cited line. The impact (memory-exhaustion DoS / bad_alloc) is accurate. The proposed fix is correct and sufficient: compute stream_size once (seekg to end, subtract start), then enforce `blob_offset <= stream_size && blob_size <= stream_size - blob_offset` (overflow-safe, no addition) before each resize(). The same pattern should be applied to header.bson_size at line 357.

## Exploit / Proof of Concept
Craft a .bin with a valid header/magic/version and a BSON `blob_metadata` object whose entry sets `data_offset` inside the file but `size` = 4 GiB (or a dozen entries of 1 GiB each). On load, line 425 calls `std::vector::resize(4GiB)` per entry before validating that the file even contains that many bytes, exhausting/committing process memory.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-789 in BinManager::DeserializeImpl
// (targets/onnxruntime/onnxruntime/core/providers/openvino/ov_bin_manager.cc:425)
//
// The flaw: blob_size read from BSON is passed directly to container.data.resize(blob_size)
// with no check that blob_size <= (stream_size - blob_offset).
// Pre-fix: resize(0xFFFFFFFF) (or similar) OOM-kills / throws bad_alloc.
// Post-fix: ORT_ENFORCE fires before the allocation.
//
// Harness: ORT unit tests for the OpenVINO EP.
// TODO: Locate the exact gtest binary name for the ORT OV EP tests.
//       Likely target: onnxruntime_test_providers_openvino or similar.
//       Check CMakeLists.txt under onnxruntime/test/providers/openvino/.

#include "gtest/gtest.h"
#include <sstream>
#include <cstdint>
#include <cstring>

// TODO: Replace with the correct include path for BinManager
// #include "core/providers/openvino/ov_bin_manager.h"
// #include "core/providers/openvino/ov_shared_context.h"

namespace onnxruntime {
namespace openvino_ep {

// Helper: build a minimal valid .bin stream with one blob whose BSON 'size'
// far exceeds the actual stream length.
static std::string BuildMaliciousBin() {
    // TODO: Construct a binary stream matching the header_t layout
    // (magic, version, header_size, bson_start_offset, bson_size)
    // followed by a BSON document whose blob_metadata entry has
    // data_offset=<within stream> and size=0xFFFFFFFF (4 GiB).
    //
    // The BSON can be built with nlohmann::json::to_bson():
    //   nlohmann::json j;
    //   j["version"] = "<current BSONFields::kCurrentBsonVersion>";
    //   j["blob_metadata"]["crafted_blob"]["data_offset"] = <small valid offset>;
    //   j["blob_metadata"]["crafted_blob"]["size"] = (uint64_t)0xFFFFFFFF;
    //   auto bson = nlohmann::json::to_bson(j);
    // Then prepend a header_t pointing to this BSON.
    //
    // TODO: fill in kMagicNumber, BinVersion::current, kCurrentBsonVersion
    //        from ov_bin_manager.h / ov_bin_manager.cc constants.
    return ""; // placeholder
}

TEST(BinManagerDeserializeTest, RejectsOversizedBlobSize) {
    // This test encodes the fix: an oversized blob_size must be rejected
    // before any allocation, not after.
    std::string bin_data = BuildMaliciousBin();
    // TODO: skip test if BuildMaliciousBin() returns empty (not yet implemented).
    if (bin_data.empty()) { GTEST_SKIP() << "Malicious .bin builder not yet implemented — see TODOs."; }

    std::istringstream ss(bin_data);

    // TODO: instantiate BinManager with a dummy path.
    // BinManager mgr("");
    // EXPECT_THROW(mgr.Deserialize(ss, nullptr), onnxruntime::OnnxRuntimeException)
    //     << "Expected ORT_ENFORCE to fire for blob_size exceeding stream bounds";
    //
    // Pre-fix: the above throws std::bad_alloc (or OOM-kills) instead of OnnxRuntimeException.
    // Post-fix: ORT_ENFORCE(blob_size <= stream_size - blob_offset) fires cleanly.
    GTEST_SKIP() << "Skeleton only — see TODOs above for symbol names and binary layout.";
}

} // namespace openvino_ep
} // namespace onnxruntime
```
**Build / run:** Build target: locate the ORT OV EP test binary (likely 'onnxruntime_test_providers_openvino' or similar under onnxruntime/test/providers/openvino/CMakeLists.txt). Run: ./onnxruntime_test_providers_openvino --gtest_filter=BinManagerDeserializeTest.RejectsOversizedBlobSize. Pre-fix failure mode: std::bad_alloc or process OOM-killed when resize(0xFFFFFFFF) is attempted at ov_bin_manager.cc:425. Post-fix: ORT_ENFORCE fires with 'blob exceeds stream bounds' before the allocation.

## Suggested fix
Before line 425, validate `blob_size` against the real stream size: after determining stream length (seek to end / stat), enforce `blob_offset <= stream_size` and `blob_size <= stream_size - blob_offset` (overflow-safe, no addition), and reject entries exceeding a sane maximum. E.g. `ORT_ENFORCE(blob_offset <= stream_size && blob_size <= stream_size - blob_offset, "blob exceeds stream bounds");` then `resize`. This bounds the allocation to bytes that actually exist in the file.

## Models used

| Role | Provider / model |
| --- | --- |
| Idea | `claude-4-6-sonnet` |
| Tech lead | `claude-4-6-sonnet` |
| Researcher | `claude-4-8-opus` |
| Validator | `claude-4-6-sonnet` |


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #747.
