# Security finding #748: `header.bson_size` is taken verbatim from the untrusted file header…

**Summary:** `header.bson_size` is taken verbatim from the untrusted file header…

**CWE IDs:** CWE-789: Memory Allocation with Excessive Size Value
**Severity / Impact:** Memory-exhaustion DoS at the very start of deserialization: an attacker-chosen `bson_size` of several GiB forces an oversized allocation before the BSON is even read. Same threat surface as the blob case, triggered earlier and unconditionally.
**Affected location:** `targets/onnxruntime/onnxruntime/core/providers/openvino/ov_bin_manager.cc:357` — `BinManager::DeserializeImpl()`
**Validated for repos:** onnxruntime
**Trust boundary:** `header.bson_size` field in the attacker-written .bin ep-context-cache header → ORT OpenVINO EP deserialization

## Description / Root cause
`header.bson_size` is taken verbatim from the untrusted file header and used to size a heap buffer `std::vector<uint8_t> bson_data(header.bson_size)` (line 357) before any check that the file actually contains that many bytes. The header validation (lines 346-348) checks magic/version/header_size but never bounds `bson_size` or `bson_start_offset` against the file length.

**Validator analysis:** The vulnerability type (CWE-789, excessive allocation from an attacker-controlled size field) is accurately diagnosed. The impact (memory-exhaustion DoS) is correct: a crafted `bson_size` of e.g. 0xFFFFFFFFFFFFFFFF will cause `std::vector<uint8_t>` to attempt a multi-GiB allocation before any I/O validation, throwing `std::bad_alloc` — DoS, not code execution. The proposed fix is correct and sufficient: read the stream size (via `seekg`/`tellg`) immediately after validating the header, then enforce `header.bson_start_offset <= stream_size` and `header.bson_size <= stream_size - header.bson_start_offset` before allocating `bson_data`. The Plugin EP's implementation (line 248 in the plugin's `ov_bin_manager.cc`) serves as a direct reference implementation for the correct pattern. One improvement over the proposed fix: also add an absolute sane cap (e.g. 64 MiB) to catch files where `bson_size` is plausible but unreasonably large.

## Exploit / Proof of Concept
Provide a .bin with valid magic/version/header_size but `bson_size` set to a huge 64-bit value; line 357 allocates that many bytes immediately, and `stream.read` (line 358) attempts the oversized read.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes: BinManager::DeserializeImpl (ov_bin_manager.cc:357) must reject a
// header whose bson_size exceeds the actual stream length rather than attempting
// an oversized allocation.  Pre-fix: std::bad_alloc or OOM crash. Post-fix:
// ORT_ENFORCE fires with "BSON size out of bounds" (or equivalent) error.

#include "gtest/gtest.h"
#include <sstream>
#include <cstdint>
#include <cstring>

// TODO: replace with the actual include path once confirmed from the build tree
// #include "core/providers/openvino/ov_bin_manager.h"
// #include "core/providers/openvino/ov_shared_context.h"

namespace onnxruntime {
namespace openvino_ep {
namespace test {

// Mirror of the private header_t in ov_bin_manager.cc — keep in sync.
struct header_t {
  uint64_t magic;         // 0x4E49425F5045564FULL
  uint64_t version;       // 1
  uint64_t header_size;   // sizeof(header_t) == 40
  uint64_t bson_start_offset;
  uint64_t bson_size;
};

static constexpr uint64_t kMagicNumber = 0x4E49425F5045564FULL;
static constexpr uint64_t kVersion     = 1;

// Build a minimal .bin stream: valid header but bson_size >> actual stream bytes.
static std::istringstream BuildCraftedStream(uint64_t bson_size_value) {
  header_t hdr{};
  hdr.magic              = kMagicNumber;
  hdr.version            = kVersion;
  hdr.header_size        = sizeof(header_t);
  hdr.bson_start_offset  = sizeof(header_t); // immediately after header
  hdr.bson_size          = bson_size_value;  // attacker-controlled

  std::string buf(sizeof(header_t), '\0');
  std::memcpy(buf.data(), &hdr, sizeof(hdr));
  // No actual BSON bytes follow — stream ends after the 40-byte header.
  return std::istringstream(buf, std::ios::binary);
}

// TODO: instantiate BinManager with a real (or stub) bin_path once headers are
// confirmed.  Adjust shared_context arg to match the public API.
TEST(BinManagerDeserialize, RejectsBsonSizeLargerThanStream) {
  // bson_size = 2 GiB — vastly exceeds the 40-byte test stream.
  auto stream = BuildCraftedStream(static_cast<uint64_t>(2) * 1024 * 1024 * 1024);

  // TODO: construct BinManager properly; adjust path arg as needed.
  // BinManager mgr(std::filesystem::path{});
  // EXPECT_THROW(mgr.Deserialize(stream, nullptr), onnxruntime::OnnxRuntimeException);
  //
  // Pre-fix behaviour: throws std::bad_alloc (or process OOM) at ov_bin_manager.cc:357.
  // Post-fix behaviour: ORT_ENFORCE fires before the vector<uint8_t> construction.
  //
  // TODO: once headers/symbols are confirmed, remove the stub and uncomment above.
  GTEST_SKIP() << "TODO: wire up BinManager headers and remove this skip";
}

TEST(BinManagerDeserialize, RejectsBsonStartOffsetPlusSizeOverflow) {
  // bson_start_offset == sizeof(header_t), bson_size == UINT64_MAX-sizeof(header_t)+1
  // The sum wraps, exercising the overflow branch of the proposed bounds check.
  uint64_t crafted_size = std::numeric_limits<uint64_t>::max() - sizeof(header_t) + 1;
  auto stream = BuildCraftedStream(crafted_size);

  // TODO: same as above.
  GTEST_SKIP() << "TODO: wire up BinManager headers and remove this skip";
}

}  // namespace test
}  // namespace openvino_ep
}  // namespace onnxruntime
```
**Build / run:** Build target: onnxruntime_test_all (or the openvino-EP-specific test target, e.g. onnxruntime_providers_openvino_unittests — confirm from CMakeLists.txt in onnxruntime/test/providers/openvino/). Filter: --gtest_filter='BinManagerDeserialize.*'. Expected pre-fix failure: std::bad_alloc or process OOM when header.bson_size is large; with ASan: heap-allocation-failure or ENOMEM abort at ov_bin_manager.cc:357. Post-fix: ORT_ENFORCE exception thrown before any allocation.

## Suggested fix
After reading the header, compute the actual stream length and enforce `header.bson_start_offset <= stream_size` and `header.bson_size <= stream_size - header.bson_start_offset` (and an absolute sane cap) before allocating `bson_data`, so the allocation cannot exceed the file's real size.

## Models used

| Role | Provider / model |
| --- | --- |
| Idea | `claude-4-6-sonnet` |
| Tech lead | `claude-4-6-sonnet` |
| Researcher | `claude-4-8-opus` |
| Validator | `claude-4-6-sonnet` |


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #748.
