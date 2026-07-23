# Security finding #743: Per-blob blob_size (line 412) and blob_offset (line 411) are taken …

**Summary:** Per-blob blob_size (line 412) and blob_offset (line 411) are taken …

**CWE IDs:** CWE-789: Memory Allocation with Excessive Size Value / CWE-770
**Severity / Impact:** A malicious .bin can declare a blob with blob_size = 0xFFFFFFFFFFFFFFFF (or many GB), forcing container.data.resize() to attempt a huge value-initialized allocation for every blob entry — memory-exhaustion DoS, or std::bad_alloc (converted to ORT error by the line-335 catch). Attacker also controls blob_offset with no bound, so reads occur from arbitrary stream positions. Affects any consumer loading an attacker-supplied context cache.
**Affected location:** `targets/onnxruntime/onnxruntime/core/providers/openvino/ov_bin_manager.cc:425` — `BinManager::DeserializeImpl()`
**Validated for repos:** chromiumWebnn, onnxruntime
**Trust boundary:** attacker-controlled BSON blob metadata (blob_size / blob_offset as uint64_t) inside the EP-context .bin, reachable from WebNN → ORT → OV EP

## Description / Root cause
Per-blob blob_size (line 412) and blob_offset (line 411) are taken directly from the attacker-controlled BSON and used at lines 422-426 as `stream.seekg(blob_offset)`, `container.data.resize(blob_size)` and `stream.read(..., blob_size)` with no offset+size <= file_size validation. The plugin counterpart explicitly calls `OVEP_ENFORCE(IsWithinBounds(blob_offset, blob_size, data.size()), ...)` (plugin ov_bin_manager.cc:304-305); the ORT EP version omits this guard entirely.

**Validator analysis:** The vuln type (CWE-789/CWE-770) and impact (memory-exhaustion DoS) are accurate. A crafted ONNX model embedded in compiled_model_data can declare BSON blob entries with blob_size = 0xFFFFFFFFFFFFFFFF; container.data.resize(blob_size) at line 425 will attempt an enormous allocation for each blob entry in the loop. std::bad_alloc is caught at line 335 and converted to ORT error, so the process does not crash, but OOM conditions can exhaust available memory on the dispatch host. blob_offset is also unconstrained, so reads occur from arbitrary stream positions (line 426) even if the resize succeeds. The proposed fix is correct: compute file_size once from the stream, then inside the loop enforce blob_offset <= file_size && blob_size <= file_size - blob_offset (overflow-safe) before resize, matching the plugin's IsWithinBounds pattern. This is sufficient provided overflow-safe arithmetic (SafeInt or equivalent) is used for the sum.

## Exploit / Proof of Concept
Craft valid BSON blob metadata whose kSize field is enormous (and/or kDataOffset beyond EOF). During the loop at lines 410-433, resize(blob_size) triggers an uncontrolled allocation before/independent of validating the value against the real file size; multiple such entries multiply the effect.

## Reproduction (steps)
```
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>WebNN OV EP BinManager DoS PoC</title>
</head>
<body>
<pre id="log"></pre>
<script>
// Proof-of-concept: craft an ONNX model buffer containing an EPContext node
// whose EP_CACHE_CONTEXT attribute embeds a BSON blob with blob_size = 2^32-1.
// When LoadCompiledGraph sends this to the dispatch context, ORT calls
// BinManager::Deserialize, which hits container.data.resize(blob_size) at
// ov_bin_manager.cc:425 without bounds checking, causing OOM.
//
// NOTE: A fully working repro requires:
//   1. An ORT build with the OpenVINO EP loaded.
//   2. A Chromium build with WebNN + ORT dispatch context enabled.
//   3. A crafted ONNX model (binary) whose EPContext node EP_CACHE_CONTEXT
//      attribute contains a BSON payload with kSize = 0xFFFFFFFF (or larger).
//
// This page demonstrates the WebNN API entry path only; the binary payload
// must be supplied as a pre-compiled compiled_model_data blob (steps below).

const log = document.getElementById('log');
function print(msg) { log.textContent += msg + '\n'; }

// Step 1: Construct a minimal WebNN context targeting the ORT/OpenVINO EP.
if (!navigator.ml) {
  print('WebNN not available — run Chromium with --enable-features=WebMachineLearningNeuralNetwork');
} else {
  navigator.ml.createContext({ deviceType: 'gpu' }).then(async ctx => {
    print('Context created: ' + ctx);
    // Step 2: In a real exploit, a compiled model (ONNX with EPContext node
    // containing crafted BSON blob_size) would be loaded via LoadCompiledGraph
    // Mojo call. The renderer can tamper with compiled_model_data returned by
    // the compiler context before re-sending it to the dispatch context.
    //
    // The crafted BSON structure that triggers the flaw:
    //   { blobMetadata: { 'blob0': { kDataOffset: 0, kSize: 4294967295 } } }
    // When deserialized, container.data.resize(4294967295) runs unchecked.
    print('Repro requires a compromised renderer sending crafted compiled_model_data.');
    print('See steps in reproductionKind=steps for the full attack chain.');
  }).catch(e => print('Error: ' + e));
}
</script>
</body>
</html>
```

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for: ov_bin_manager.cc:412,425
// BinManager::DeserializeImpl does not check blob_offset + blob_size <= file_size
// before container.data.resize(blob_size), allowing attacker-controlled BSON to
// trigger an enormous allocation (CWE-789/CWE-770).
//
// Assertion: DeserializeImpl must throw / enforce-fail when blob_size exceeds
// the stream length, not attempt the allocation.
// Pre-fix: resize(blob_size) is called unconditionally → std::bad_alloc or OOM.
// Post-fix: ORT_ENFORCE(blob_size <= file_size - blob_offset, ...) fires first.

#include <gtest/gtest.h>
#include <sstream>
#include <cstdint>

// TODO: Replace with the actual include path once confirmed.
// #include "core/providers/openvino/ov_bin_manager.h"
// #include "core/providers/openvino/ov_shared_context.h"

namespace onnxruntime {
namespace openvino_ep {
namespace test {

// Helper: build a minimal valid .bin stream whose BSON blob_size is oversized.
// The stream has a valid header + BSON but declares blob_size = 0xFFFFFFFF.
std::istringstream BuildCraftedBinWithOverflowBlobSize() {
  // TODO: Generate a real BSON payload using nlohmann::json BSON encoding,
  // with BSONFields::kBlobMetadata containing:
  //   { "blob0": { kDataOffset: 0, kSize: 4294967295 } }
  // then prepend the BinManager header_t with correct magic, version, bson offsets.
  // This requires reading the exact layout from ov_bin_manager.cc:~30-100 and
  // BSONFields constants — left as TODO for a compiler-verified implementation.
  return std::istringstream(""); // placeholder
}

TEST(BinManagerDeserializeTest, RejectsOversizedBlobSize) {
  // TODO: Instantiate BinManager and SharedContext as done in existing ORT EP tests.
  // Example (adjust to actual API):
  //   auto bin_manager = BinManager(std::nullopt /*no external path*/);
  //   auto shared_ctx = std::make_shared<SharedContext>(...);
  //   auto crafted_stream = BuildCraftedBinWithOverflowBlobSize();
  //
  // Pre-fix: this call exhausts memory or throws std::bad_alloc wrapped in ORT error.
  // Post-fix: ORT_ENFORCE fires with "offset+size out of bounds" before resize.
  //
  // EXPECT_THROW(bin_manager.Deserialize(crafted_stream, shared_ctx),
  //              onnxruntime::OnnxRuntimeException);
  GTEST_SKIP() << "TODO: construct crafted BSON stream with oversized blob_size; "
                  "see ov_bin_manager.cc:412,425";
}

TEST(BinManagerDeserializeTest, RejectsOversizedBlobOffset) {
  // Same as above but kDataOffset = UINT64_MAX, kSize = 1.
  // Pre-fix: seekg(UINT64_MAX) may silently succeed or no-op, then stream.good()
  // check at line 423 may or may not catch it; resize(1) succeeds but read is
  // from wrong position.
  // Post-fix: ORT_ENFORCE(blob_offset <= file_size) fires.
  GTEST_SKIP() << "TODO: construct crafted BSON stream with out-of-bounds blob_offset; "
                  "see ov_bin_manager.cc:411,422";
}

} // namespace test
} // namespace openvino_ep
} // namespace onnxruntime
```
**Build / run:** Build target: onnxruntime_test_all (or the per-component EP test target that includes openvino provider tests). Filter: --gtest_filter=BinManagerDeserializeTest.*. Expected pre-fix failure: std::bad_alloc caught and rethrown as OnnxRuntimeException (line 336), or OOM in the allocator. With ASan+LSan: potential large allocation report. Post-fix: ORT_ENFORCE throws OnnxRuntimeException with 'offset+size out of bounds' message before any allocation. TODO: confirm exact gtest binary name from onnxruntime/test/CMakeLists.txt and link the openvino EP test helpers.

## Suggested fix
Add the missing bounds check inside the loop before line 425, e.g. ORT_ENFORCE(blob_offset <= file_size && blob_size <= file_size - blob_offset, ...) using overflow-safe (SafeInt/uint64) arithmetic, matching the plugin's IsWithinBounds(blob_offset, blob_size, file_size). Compute file_size once from the stream length and reject any blob whose offset+size exceeds it.

## Models used

| Role | Provider / model |
| --- | --- |
| Idea | `claude-4-6-sonnet` |
| Tech lead | `claude-4-6-sonnet` |
| Researcher | `claude-4-8-opus` |
| Validator | `claude-4-6-sonnet` |


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #743.
