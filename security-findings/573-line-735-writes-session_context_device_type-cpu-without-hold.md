# Security finding #573: Line 735 writes `session_context_.device_type = "CPU"` without hold…

**Summary:** Line 735 writes `session_context_.device_type = "CPU"` without hold…

**CWE IDs:** CWE-366: Race Condition within a Thread
**Severity / Impact:** In a model with multiple subgraphs dispatched to the OpenVINO NPU EP (common for large models), if one subgraph triggers the NPU-to-CPU fallback, the concurrent write to `device_type` races with reads in other BackendManagers. The corrupted or torn `device_type` string causes a wrong cache key in another BackendManager's `backend_map_` lookup at line 700, potentially matching a cached backend compiled for different input shapes. When that backend's `Infer()` is called at line 773 with the actual runtime input tensors (whose shapes differ from what the cached backend was compiled for), OpenVINO reads/writes out-of-bounds into its inference buffers — heap corruption, potential RCE.
**Affected location:** `targets/onnxruntime/onnxruntime/core/providers/openvino/backend_manager.cc:735` — `BackendManager::Compute()`
**Validated for repos:** onnxruntime
**Trust boundary:** ORT parallel thread pool executing multiple subgraph Compute() calls concurrently; all BackendManagers in the same EP instance share session_context_ by reference (openvino_execution_provider.cc:143)

## Description / Root cause
Line 735 writes `session_context_.device_type = "CPU"` without holding any lock. `session_context_` is a shared reference (all BackendManagers in one EP hold a reference to the same `SessionContext`). Concurrent `Compute()` calls on other BackendManagers read `session_context_.device_type` at line 700 (`MakeMapKeyString(tensor_shapes, session_context_.device_type)`) while this thread is writing it — a C++ data race causing undefined behavior. The `mutex_` only protects `backend_map_`, not the `SessionContext` fields.

**Validator analysis:** The data race is real: SessionContext& session_context_ (backend_manager.h:60) is a single object shared by reference among every BackendManager (openvino_execution_provider.cc:143). backend_manager.cc:735-737 writes session_context_.device_type/precision with no lock, while :700 reads session_context_.device_type unlocked in other BackendManagers; the only lock (mutex_ at :704/:769) is per-BackendManager and guards backend_map_ only. Under ORT_PARALLEL execution with two independent OV fused-subgraphs, concurrent Compute() calls make the concurrent std::string write/read a genuine C++ data race → UB. So the core defect and its reachability (under parallel execution, dynamic-shapes-disabled path, NPU compile-failure fallback) hold. However the vuln_type and impact are overstated: (1) CWE-366 (race within a thread / signal reentrancy) is the wrong class — this is CWE-362 (improper synchronization of a shared resource across threads). (2) The RCE/heap-corruption chain (torn device_type string yielding a cache key that happens to collide with a backend compiled for other shapes) is speculative: the key encodes tensor_shapes too, so a corrupted device_type almost always fails to match and just creates a new map entry; the realistic worst case is a crash from reading a std::string mid-mutation or a logic bug where the whole EP silently switches all subgraphs to CPU permanently (device_type is mutated globally and never restored). The window is also narrow — the write only fires on first-time dynamic-backend creation when NPU compilation fails. The proposed fix is correct and sufficient in spirit: do not mutate shared session_context_ in Compute(); use a local device string for the key and pass a per-call/local SessionContext copy to MakeBackend. Even better, the fallback should also avoid permanently mutating shared precision/device_type for all other subgraphs, which is a correctness bug independent of threading; a local SessionContext copy fixes both the race and the global-state-mutation defect.

## Exploit / Proof of Concept
Prerequisites: (1) A multi-subgraph ONNX model dispatched to OpenVINO NPU EP, with `disable_dynamic_shapes=true`, `so_disable_cpu_ep_fallback=false`. (2) ORT's intra-op or inter-op parallelism enabled. Trigger: SubgraphA's Compute() hits NPU compilation failure, enters the fallback branch, and writes `session_context_.device_type = "CPU"` (line 735) while SubgraphB's Compute() concurrently reads `session_context_.device_type` at line 700 to build its cache key. SubgraphB may then look up or insert a backend under the wrong key. On the next inference call for SubgraphB (original device_type key still in the map, now shadowed by wrong key), `dynamic_backend->Infer(context)` runs a mismatched OpenVINO compiled model against the actual runtime inputs — OOB reads/writes in OpenVINO's tensor accessors.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for the data race / shared-state mutation at
// targets/onnxruntime/onnxruntime/core/providers/openvino/backend_manager.cc:735-737.
// Pre-fix: BackendManager::Compute() writes the shared SessionContext.device_type="CPU"
// (backend_manager.cc:735) on the NPU->CPU fallback path; because SessionContext is shared
// by reference across all BackendManagers (openvino_execution_provider.cc:143), this both
// races with concurrent reads at backend_manager.cc:700 (TSan data race) and permanently
// corrupts device_type for every other subgraph.
// Post-fix: the fallback must use a local device string for the cache key and a local
// SessionContext copy for MakeBackend, leaving the shared SessionContext.device_type intact.
//
// NOTE: No gtest harness in the unit-test table maps to onnxruntime/core/providers/openvino,
// and the trigger needs a real NPU compile failure + ORT_PARALLEL execution, so this is a
// SKELETON. Fill in the TODOs before use.

#include <gtest/gtest.h>
#include <thread>
#include <string>
// TODO: include the real headers for SessionContext, SubGraphContext, SharedContext,
//       and BackendManager once their include paths are confirmed by reading the repo
//       (core/providers/openvino/contexts.h, backend_manager.h).

TEST(OpenVINOBackendManager, FallbackDoesNotMutateSharedSessionContext) {
  // TODO: construct a SessionContext with device_type = "NPU" and
  //       so_disable_cpu_ep_fallback = false, disable_dynamic_shapes = true.
  // TODO: build a fused ONNX subgraph with a dynamic input shape whose NPU compilation
  //       fails (forces the fallback branch at backend_manager.cc:730-737).
  // TODO: create two BackendManagers sharing the SAME SessionContext& (mirroring
  //       openvino_execution_provider.cc:143).
  //
  // std::string device_before = session_context.device_type;
  // BackendManager bmA(session_context, ...); // subgraph that hits NPU fallback
  // BackendManager bmB(session_context, ...); // independent subgraph
  //
  // Drive bmA.Compute() (fallback) so that any device_type change is observed.
  // ASSERT: the shared SessionContext must be unchanged after the fallback.
  // EXPECT_EQ(session_context.device_type, device_before);  // fails pre-fix (becomes "CPU")
  //
  // Concurrency check (build with -fsanitize=thread): run bmA.Compute() (fallback) and
  // bmB.Compute() (reads device_type at :700) on two threads; TSan must report no data
  // race on SessionContext.device_type once the fix uses a local copy.
  GTEST_SKIP() << "Skeleton: fill in SessionContext/subgraph fixtures and NPU-failure stub.";
}
```
**Build / run:** No table harness maps to onnxruntime/core/providers/openvino. Best effort: build the ORT OpenVINO EP unit tests with ThreadSanitizer (e.g. build_dir/onnxruntime_test_all or the provider's ctest target, compiled with --use_tsan / -fsanitize=thread) and run --gtest_filter=OpenVINOBackendManager.FallbackDoesNotMutateSharedSessionContext. Expected pre-fix signal: EXPECT_EQ on device_type fails (mutated to "CPU"), and under TSan a 'data race' report on SessionContext::device_type between backend_manager.cc:735 (write) and :700 (read). Post-fix (local device string + local SessionContext copy) both go away.

## Suggested fix
Avoid mutating the shared `session_context_` in `Compute()`. Replace `session_context_.device_type = "CPU"` and the subsequent `key = MakeMapKeyString(tensor_shapes, session_context_.device_type)` with a local variable: `std::string local_device = "CPU"; key = MakeMapKeyString(tensor_shapes, local_device);`. Then pass `local_device` to `BackendFactory::MakeBackend` via a modified per-call context or a local SessionContext copy, rather than mutating the shared SessionContext. Alternatively, if mutation of the shared SessionContext is required, protect `session_context_.device_type` accesses in `Compute()` with the existing `mutex_` or a dedicated `session_context_mutex_`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #573.
