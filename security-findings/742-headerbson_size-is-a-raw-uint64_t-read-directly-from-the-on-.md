# Security finding #742: header.bson_size is a raw uint64_t read directly from the on-disk h…

**Summary:** header.bson_size is a raw uint64_t read directly from the on-disk h…

**CWE IDs:** CWE-789: Memory Allocation with Excessive Size Value / CWE-770
**Severity / Impact:** A crafted header with bson_size = 0xFFFFFFFFFFFFFFFF (or any value far larger than the file) causes an immediate huge value-initialized allocation. The vector constructor zero-fills, committing every page, so a moderately large value forces real memory commit → memory exhaustion / OOM DoS; an enormous value throws std::bad_alloc. The bad_alloc is caught by the try/catch in Deserialize (line 335) and re-thrown as an ORT error (so not an uncaught crash), but the uncontrolled allocation itself is a memory-exhaustion DoS affecting any process loading an attacker-supplied context cache.
**Affected location:** `targets/onnxruntime/onnxruntime/core/providers/openvino/ov_bin_manager.cc:357` — `BinManager::DeserializeImpl()`
**Validated for repos:** onnxruntime
**Trust boundary:** attacker-crafted EP-context .bin file header (raw uint64_t header.bson_size) consumed via ONNX OpenVINO EP context cache load, reachable from WebNN → ORT → OV EP

## Description / Root cause
header.bson_size is a raw uint64_t read directly from the on-disk header at line 344 and used verbatim as the size argument to `std::vector<uint8_t> bson_data(header.bson_size)` at line 357. Unlike the plugin counterpart there is no upper-bound check that bson_size <= remaining stream/file size before the allocation. The header validation (lines 346-348) only checks magic/version/header_size, never bson_size.

**Validator analysis:** The vuln type (CWE-789/CWE-770 — allocation with excessive size) is accurate. The impact (memory-exhaustion DoS; not an uncaught crash because std::bad_alloc is caught at ov_bin_manager.cc:333-338) is correctly stated. The proposed fix is correct and sufficient: compute total stream length (seekg to end / tellg), then assert `header.bson_start_offset + header.bson_size <= file_size` using overflow-safe arithmetic before line 357, mirroring the plugin's `IsWithinBounds` pattern. An additional sane-maximum cap (e.g., 512 MiB) would add defence-in-depth. The fix needs overflow-safe addition to avoid wrapping (use checked add or cast to unsigned 128-bit / compare separately: `header.bson_start_offset <= file_size && header.bson_size <= file_size - header.bson_start_offset`).

## Exploit / Proof of Concept
Supply an EP context .bin whose header passes the magic/version/header_size checks but sets bson_size to a value near SIZE_MAX or many GB. When DeserializeImpl reaches line 357 it attempts to allocate/zero that many bytes before any read of actual BSON content, exhausting memory or aborting the load.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-789 in BinManager::DeserializeImpl
// (targets/onnxruntime/onnxruntime/core/providers/openvino/ov_bin_manager.cc:357)
//
// A crafted header passes magic/version/header_size checks but sets bson_size
// to a value far larger than the file; pre-fix this causes a huge allocation;
// post-fix it must throw an ORT exception before allocation.
//
// Build target: onnxruntime_test_all (or the OV EP unit test target in the ORT
// repo — locate via cmake/onnxruntime_unittests.cmake or similar).
// Run: --gtest_filter=BinManagerDeserializeTest.RejectOversizedBsonSize
// Expected pre-fix behaviour: std::bad_alloc caught and re-thrown as OrtException
//   (memory exhaustion on large-but-allocable sizes).
// Expected post-fix behaviour: OrtException thrown BEFORE allocation attempt.

#include <sstream>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "gtest/gtest.h"

// TODO: Replace with the actual include path once confirmed from the ORT build
// system.  The BinManager class is in:
//   onnxruntime/core/providers/openvino/ov_bin_manager.h
// and its constructor requires a SessionContext or equivalent — adapt as needed.
// #include "core/providers/openvino/ov_bin_manager.h"

// Magic number from ov_bin_manager.cc ("OVEP_BIN" little-endian)
static constexpr uint64_t kMagicNumber = 0x4E49425F5045564FULL;

// Version 1
static constexpr uint64_t kVersion1 = 1;

// Minimal header_t layout (must match struct header_t in ov_bin_manager.cc)
struct TestHeader {
  uint64_t magic;
  uint64_t version;
  uint64_t header_size;
  uint64_t bson_start_offset;
  uint64_t bson_size;
};

namespace {

// Build a minimal well-formed stream whose header passes magic/version/
// header_size checks but carries a bson_size far larger than the file.
std::istringstream MakeCraftedStream(uint64_t evil_bson_size) {
  TestHeader hdr{};
  hdr.magic            = kMagicNumber;
  hdr.version          = kVersion1;
  hdr.header_size      = sizeof(TestHeader);
  hdr.bson_start_offset = sizeof(TestHeader);  // BSON immediately after header
  hdr.bson_size        = evil_bson_size;        // << crafted oversized value

  std::string buf(sizeof(TestHeader), '\0');
  std::memcpy(buf.data(), &hdr, sizeof(TestHeader));
  return std::istringstream(std::move(buf), std::ios::binary);
}

}  // namespace

// TODO: Instantiate BinManager with a dummy SessionContext / model path as
// required by its constructor.  The exact constructor signature must be verified
// by reading ov_bin_manager.h.
TEST(BinManagerDeserializeTest, RejectOversizedBsonSize) {
  // bson_size = 4 GiB — large enough to exhaust memory on most systems;
  // small enough that a 64-bit allocator may actually attempt it.
  auto stream = MakeCraftedStream(0xFFFFFFFFULL);

  // TODO: Replace with the real BinManager construction:
  //   onnxruntime::openvino_ep::BinManager bin_mgr(...);
  //   EXPECT_THROW(bin_mgr.Deserialize(stream, nullptr), onnxruntime::OnnxRuntimeException);
  //
  // Pre-fix: std::bad_alloc is caught inside Deserialize and re-thrown as
  //   OnnxRuntimeException (ov_bin_manager.cc:335-337), confirming memory exhaustion.
  // Post-fix: an OnnxRuntimeException is thrown BEFORE reaching line 357,
  //   verifying the bounds check fires with zero allocation.

  // Placeholder assertion so the test structure compiles:
  EXPECT_TRUE(true) << "TODO: wire up BinManager and assert OnnxRuntimeException";
}

TEST(BinManagerDeserializeTest, RejectMaxUint64BsonSize) {
  // bson_size = SIZE_MAX — guaranteed to produce std::bad_alloc; confirms the
  // pre-fix code path catches and re-throws rather than crashing.
  auto stream = MakeCraftedStream(0xFFFFFFFFFFFFFFFFULL);

  // TODO: Same as above — replace placeholder with:
  //   onnxruntime::openvino_ep::BinManager bin_mgr(...);
  //   EXPECT_THROW(bin_mgr.Deserialize(stream, nullptr), onnxruntime::OnnxRuntimeException);
  EXPECT_TRUE(true) << "TODO: wire up BinManager and assert OnnxRuntimeException";
}
```
**Build / run:** Build target (locate in cmake/onnxruntime_unittests.cmake or equivalent ORT build file — likely `onnxruntime_test_all` or a dedicated OV EP test binary). Run with: --gtest_filter=BinManagerDeserializeTest.*. Pre-fix with a 4 GiB bson_size: expect OnnxRuntimeException from the re-throw at ov_bin_manager.cc:336. Pre-fix with SIZE_MAX: std::bad_alloc caught and re-thrown as OnnxRuntimeException. Post-fix: OnnxRuntimeException thrown before line 357 (no allocation attempt). Enable ASan to confirm no actual large allocation occurs post-fix.

## Suggested fix
Before line 357, compute the actual stream length (seekg to end / tellg, or the known file size) and ORT_ENFORCE(header.bson_start_offset + header.bson_size <= file_size) with overflow-safe arithmetic, and cap bson_size to a sane maximum. Mirror the plugin's IsWithinBounds guard for the BSON region.

## Models used

| Role | Provider / model |
| --- | --- |
| Idea | `claude-4-6-sonnet` |
| Tech lead | `claude-4-6-sonnet` |
| Researcher | `claude-4-8-opus` |
| Validator | `claude-4-6-sonnet` |


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #742.
